import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# 1. KẾT NỐI QDRANT CHẾ ĐỘ LOCAL
QDRANT_URL = "http://localhost:6333"
client = QdrantClient(url=QDRANT_URL)

# ✅ ĐỔI TÊN COLLECTION 
collection_name = "embeddinggemma_audio"

# 2. KIỂM TRA VÀ TẠO COLLECTION (✅ ĐỔI THÀNH 768 CHIỀU CHO GEMMA)
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"[*] Đã tạo mới collection: {collection_name}")
else:
    print(f"[*] Collection '{collection_name}' đã tồn tại, sẽ nạp thêm data vào.")

# 3. ĐỌC FILE JSON TỪ KAGGLE
# (Bác nhớ trỏ đúng tới file JSON vừa được tạo bằng Gemma)
input_json_path =r"C:\AIC2026\jsonupload\FinalASR.json"

print(f"📂 Đang đọc dữ liệu từ {input_json_path}...")
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 4. CHUẨN BỊ PAYLOAD
points = []
for item in data:
    points.append(
        PointStruct(
            id=item["id"],
            vector=item["embedding"],
            payload={
                "video_name": item.get("video_name", ""),
                "asr_start": item.get("asr_start", 0.0),
                "asr_end": item.get("asr_end", 0.0),
                "text": item.get("text", ""),
                "pts_time": item.get("pts_time", 0.0),
                "frame_id": item.get("frame_id", 0),
                "image_path": item.get("image_path", "")
            }
        )
    )

# 5. ĐẨY VÀO QDRANT LOCAL BẰNG BATCH
batch_size = 500
print(f"🚀 Đang bắt đầu đẩy {len(points)} bản ghi lên collection '{collection_name}'...")

for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    client.upsert(
        collection_name=collection_name,
        points=batch
    )
    print(f"✅ Đã đẩy batch từ {i} đến {min(i + batch_size, len(points))}")

print("🎉 Hoàn tất nạp toàn bộ dữ liệu ASR của Gemma vào ổ cứng!")
