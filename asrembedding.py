import os
import time
import ijson
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# CẤU HÌNH KẾT NỐI & DỮ LIỆU
QDRANT_HOST = "localhost"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334
USE_GRPC = True

COLLECTION_NAME = "embeddinggemma_audio"
JSON_FILE_PATH = r"C:\AIC2026\FinalASR.json"
VECTOR_SIZE = 768

BATCH_SIZE = 2000
PARALLEL = max(4, (os.cpu_count() or 4))
MAX_RETRIES = 5


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"[INFO] Tạo collection mới '{COLLECTION_NAME}' (size={VECTOR_SIZE})...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1_000_000),
        )
    else:
        print(f"[INFO] Collection '{COLLECTION_NAME}' đã tồn tại, tiến hành nạp thêm dữ liệu...")


def enable_indexing_after_upload(client: QdrantClient, total_points: int):
    print("[INFO] Bật lại index (HNSW)...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    while True:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        if str(info.status).lower() == "green" or (info.indexed_vectors_count or 0) >= total_points:
            break
        time.sleep(3)
    print("[DONE] Index vector đã sẵn sàng.")


def ensure_payload_index(client: QdrantClient):
    print("[INFO] Đang tạo payload index cho 'video_name'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="video_name",
        field_schema=models.PayloadSchemaType.KEYWORD
    )


def iter_points_from_json(path: str, counters: dict):
    with open(path, "rb") as f:
        items_stream = ijson.items(f, "item", use_float=True)
        for idx, item in enumerate(items_stream):
            embedding = item.get("embedding")
            if embedding is None or len(embedding) != VECTOR_SIZE:
                counters["skipped"] += 1
                continue

            counters["read"] += 1
            yield models.PointStruct(
                id=item.get("id", idx),
                vector=embedding,
                payload={
                    "video_name": item.get("video_name", ""),
                    "asr_start": float(item.get("asr_start", 0.0)),
                    "asr_end": float(item.get("asr_end", 0.0)),
                    "text": item.get("text", ""),
                    "pts_time": float(item.get("pts_time", 0.0)),
                    "frame_id": int(item.get("frame_id", 0)),
                    "image_path": item.get("image_path", "")
                },
            )


def main():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=USE_GRPC,
        timeout=120,
    )

    ensure_collection(client)

    counters = {"read": 0, "skipped": 0}
    points_gen = iter_points_from_json(JSON_FILE_PATH, counters)

    def progress_wrapped_gen():
        pbar = tqdm(desc="Stream ASR Gemma qua gRPC", unit=" điểm")
        for p in points_gen:
            pbar.update(1)
            yield p
        pbar.close()

    _t0 = time.time()
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=progress_wrapped_gen(),
        batch_size=BATCH_SIZE,
        parallel=PARALLEL,
        max_retries=MAX_RETRIES,
        wait=False,
    )

    uploaded = counters["read"]
    print(f"[INFO] Đã nạp {uploaded} bản ghi ASR.")

    while True:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        if (info.points_count or 0) >= uploaded:
            break
        time.sleep(2)

    enable_indexing_after_upload(client, uploaded)
    ensure_payload_index(client)
    print(f"[DONE] Hoàn tất ASR trong {time.time() - _t0:.1f}s.")


if __name__ == "__main__":
    main()