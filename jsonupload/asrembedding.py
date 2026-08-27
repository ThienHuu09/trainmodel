import ijson
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bge-m3audio"
JSON_FILE_PATH = r"C:\AIC2026\jsonupload\FinalASR.json"

def main():
    print("⏳ Đang kết nối Qdrant...")
    # Thêm check_compatibility=False để bỏ qua cảnh báo lệch phiên bản client/server
    client = QdrantClient(url=QDRANT_URL, timeout=60, check_compatibility=False) 

    vector_size = 768  # bge-m3 dense embedding = 768 chiều

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"🗑️ Xóa collection cũ '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"✨ Tạo mới collection '{COLLECTION_NAME}' (size={vector_size}, Cosine)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

    print(f"📂 Đang đọc file JSON bằng ijson (streaming): {JSON_FILE_PATH}...")
    
    batch_size = 500
    points = []
    skipped_count = 0
    total_processed = 0

    # Mở file ở chế độ nhị phân ('rb') để ijson xử lý streaming không tốn RAM
    with open(JSON_FILE_PATH, 'rb') as f:
        # ijson.items duyệt qua từng item trong mảng JSON lớn [...]
        items = ijson.items(f, 'item')

        for item in items:
            total_processed += 1
            vector_data = item.get("embedding")

            # Bỏ qua bản ghi thiếu embedding hoặc thiếu ID để tránh sập code
            if vector_data is None or len(vector_data) == 0 or item.get("id") is None:
                skipped_count += 1
                continue

            points.append(
                models.PointStruct(
                    id=int(item.get("id")),  # dùng id có sẵn trong file
                    vector=vector_data,
                    payload={
                        "video_name": str(item.get("video_name", "")),
                        "text": str(item.get("text", "")),
                        "asr_start": float(item.get("asr_start", 0)),
                        "asr_end": float(item.get("asr_end", 0)),
                        "pts_time": float(item.get("pts_time", 0)),
                        "frame_id": int(item.get("frame_id", 0)),
                        "image_path": str(item.get("image_path", "")),
                    }
                )
            )

            # Khi đủ batch_size thì tiến hành đẩy lên Qdrant luôn để giải phóng RAM
            if len(points) >= batch_size:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"✅ Đã upload batch tới bản ghi thứ {total_processed}...")
                points = []  # Reset mảng batch

        # Đẩy nốt phần dữ liệu dư còn lại cuối cùng
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

    if skipped_count > 0:
        print(f"⚠️ Đã bỏ qua {skipped_count} bản ghi bị thiếu embedding hoặc thiếu ID.")

    print(f"🎉 Hoàn tất upload ASR! Đã xử lý tổng cộng {total_processed} bản ghi.")

if __name__ == "__main__":
    main()