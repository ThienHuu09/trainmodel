import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ocr_collection"  # Tên collection cho OCR
JSON_FILE_PATH = r"C:\AIC2026\ocr_data.json"

def main():
    print("⏳ Đang kết nối Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    vector_size = 1024  # OCR có embedding 1024 chiều (dựa theo ảnh 2)

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

    print(f"📦 Tổng số bản ghi trong file: {len(data)}")
    points = []
    skipped_count = 0
    
    for idx, item in enumerate(data):
        point_id = idx  # File OCR không có sẵn id nên dùng index chạy tự động
        vector_data = item.get("embedding")
        
        # Kiểm tra nếu embedding bị thiếu (None) hoặc rỗng thì bỏ qua để tránh sập code
        if vector_data is None or len(vector_data) == 0:
            skipped_count += 1
            continue

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector_data,
                payload={
                    "video_name": str(item.get("video_name", "")),
                    "text": str(item.get("text", "")),
                    "pts_time": int(item.get("pts_time", 0)),
                    "frame_id": int(item.get("frame_id", 0)),
                    "image_path": str(item.get("image_path", ""))
                }
            )
        )
    
    if skipped_count > 0:
        print(f"⚠️ Đã bỏ qua {skipped_count} bản ghi bị thiếu embedding.")
    print(f"📦 Số bản ghi hợp lệ sẽ đẩy lên Qdrant: {len(points)}")

    batch_size = 500
    print("🚀 Bắt đầu upload OCR lên Qdrant...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"✅ Đã upload batch {i} -> {min(i + batch_size, len(points))}")

    print("🎉 Hoàn tất upload OCR!")

if __name__ == "__main__":
    main()
