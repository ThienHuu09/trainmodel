import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Kết nối tới Qdrant Server (đảm bảo qdrant.exe đang chạy)
client = QdrantClient(host="127.0.0.1", port=6333)

COLLECTION_NAME = "mfusion_vr"  # Đổi tên collection theo ý bạn

# 2. Tạo collection nếu chưa tồn tại (giả sử vector chiều dài 512, tuỳ mô hình embedding của bạn)
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=512,  # Số chiều của vector (ví dụ: CLIP vit-b-32 dùng 512, v.v.)
            distance=models.Distance.COSINE
        )
    )
    print(f"Đã tạo collection mới: {COLLECTION_NAME}")

# 3. Đọc dữ liệu từ file JSON (ví dụ: embeddings.json)
# Cấu trúc file JSON mong đợi: list các object, mỗi object có 'id', 'vector', và 'payload'
json_file_path = "embeddings.json" 

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

points = []
for idx, item in enumerate(data):
    # item bao gồm: vector, video_name, pts_time, image_path, frame_id,...
    points.append(
        models.PointStruct(
            id=item.get("id", idx),  # ID định danh cho điểm (số nguyên hoặc UUID)
            vector=item["vector"],    # Mảng vector embedding
            payload={
                "video_name": item.get("video_name"),
                "pts_time": item.get("pts_time"),
                "image_path": item.get("image_path"),
                "frame_id": item.get("frame_id")
            }
        )
    )

# 4. Upload (Upsert) dữ liệu lên Qdrant theo từng batch để tối ưu tốc độ
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"Đã lưu batch từ {i} đến {i + len(batch)} vào Qdrant...")

print("Hoàn tất đẩy dữ liệu JSON vào Qdrant Server thành công!")
