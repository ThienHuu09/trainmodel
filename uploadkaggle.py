import json
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# ==================== CẤU HÌNH KẾT NỐI ====================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "dfn5b_images"
JSON_FILE_PATH = "semanticbatchone_full.json"
VECTOR_SIZE = 1024
BATCH_SIZE = 800


def ensure_collection(client: QdrantClient, recreate: bool):
    exists = client.collection_exists(collection_name=COLLECTION_NAME)

    if exists and not recreate:
        print(f"[INFO] Collection '{COLLECTION_NAME}' đã tồn tại, sẽ upsert thêm/ghi đè theo id.")
        return

    if exists and recreate:
        print(f"[WARN] Xóa collection cũ '{COLLECTION_NAME}' theo yêu cầu (--recreate)...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"[INFO] Tạo collection mới '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance=COSINE)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )


def load_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file {path}.")
        raise SystemExit(1)


def to_point(item, idx):
    embedding = item.get("embedding")
    if embedding is None or len(embedding) != VECTOR_SIZE:
        print(f"[WARN] Bỏ qua record id={item.get('id', idx)} vì embedding thiếu hoặc sai chiều "
              f"(có {len(embedding) if embedding else 0}, cần {VECTOR_SIZE}).")
        return None

    return models.PointStruct(
        id=item.get("id", idx),
        vector=embedding,
        payload={
            "video_name": item.get("video_name"),
            "frame_id": item.get("frame_id"),
            "pts_time": item.get("pts_time"),
            "image_path": item.get("image_path"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                         help="Xóa collection cũ (nếu có) trước khi tạo mới. Mặc định KHÔNG xóa.")
    args = parser.parse_args()

    print("[INFO] Đang kết nối tới Qdrant server...")
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    ensure_collection(client, recreate=args.recreate)

    print(f"[INFO] Đang đọc dữ liệu từ {JSON_FILE_PATH} ...")
    data = load_data(JSON_FILE_PATH)
    total_items = len(data)
    print(f"[INFO] Tổng số bản ghi: {total_items}")

    uploaded = 0
    skipped = 0
    batch = []

    for idx, item in enumerate(tqdm(data, desc="Upload lên Qdrant")):
        point = to_point(item, idx)
        if point is None:
            skipped += 1
            continue

        batch.append(point)

        if len(batch) >= BATCH_SIZE:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
                uploaded += len(batch)
            except Exception as e:
                print(f"[ERROR] Lỗi upsert batch tại vị trí ~{idx}: {e}")
            batch = []

    # upload phần còn dư cuối cùng
    if batch:
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            uploaded += len(batch)
        except Exception as e:
            print(f"[ERROR] Lỗi upsert batch cuối: {e}")

    print(f"[DONE] Đã upload {uploaded}/{total_items} bản ghi. Bỏ qua {skipped} bản ghi lỗi.")


if __name__ == "__main__":
    main()