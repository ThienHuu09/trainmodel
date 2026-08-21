import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "trake_collection"  # Tên collection cho TraKE
JSON_FILE_PATH = r"C:\AIC2026\embeddings_fixed_finalss.json"

def main():
    print("⏳ Đang kết nối Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    vector_size = 512  # TraKE có embedding 512 chiều (dựa theo ảnh 1)

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"🗑️ Xóa collection cũ '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"✨ Tạo mới collection '{COLLECTION_NAME}' (size={vector_size}, Cosine)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

    print(f"📂 Đang đọc file JSON: {JSON_FILE_PATH}...")
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📦 Tổng số bản ghi cần đẩy: {len(data)}")
    points = []
    for idx, item in enumerate(data):
        point_id = int(item.get("id", idx))
        vector_data = item["embedding"]

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector_data,
                payload={
                    "video_name": str(item.get("video_name", "")),
                    "frame_id": int(item.get("frame_id", 0)),
                    "n_index": int(item.get("n_index", 1)),
                    "pts_time": int(item.get("pts_time", 0)),
                    "image_path": str(item.get("image_path", ""))
                }
            )
        )

    batch_size = 730
    print("🚀 Bắt đầu upload TraKE lên Qdrant...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"✅ Đã upload batch {i} -> {min(i + batch_size, len(points))}")

    print("🎉 Hoàn tất upload TraKE!")

if __name__ == "__main__":
    main()
