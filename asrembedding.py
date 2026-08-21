import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Kết nối với Qdrant (thay đổi url nếu Qdrant của bạn chạy trên cloud hoặc port khác)
client = QdrantClient("http://localhost:6333") 
collection_name = "bge-m3audio"

# 2. Đọc file JSON đã gom từ Kaggle
input_json_path = "FinalASR.json" 

print(f"📂 Đang đọc dữ liệu từ {input_json_path}...")
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. Chuyển đổi dữ liệu sang định dạng PointStruct để đẩy lên Qdrant
points = []
for item in data:
    points.append(
        models.PointStruct(
            id=item["id"],
            vector=item["embedding"],
            payload={
                "video_name": item["video_name"],
                "asr_start": item["asr_start"],
                "asr_end": item["asr_end"],
                "text": item["text"],
                "pts_time": item["pts_time"],
                "frame_id": item["frame_id"],
                "image_path": item["image_path"]
            }
        )
    )

# 4. Upsert theo batch (mỗi batch 100-500 điểm để tránh treo kết nối)
batch_size = 500
print(f"🚀 Đang bắt đầu đẩy {len(points)} bản ghi lên collection '{collection_name}'...")

for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    client.upsert(
        collection_name=collection_name,
        points=batch
    )
    print(f"✅ Đã đẩy batch từ {i} đến {min(i + batch_size, len(points))}")

print("🎉 Hoàn tất toàn bộ dữ liệu!")
