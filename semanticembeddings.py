import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==================== CẤU HÌNH KẾT NỐI ====================
# Đảm bảo Qdrant server của bạn đang chạy ở cổng này (mặc định là 6333)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "dfn5b_images"  # Đổi tên collection nếu bạn muốn (khớp với server search)
JSON_FILE_PATH = "all_embeddings_and_metadata.json"

def main():
    print("⏳ Đang kết nối tới Qdrant Server...")
    client = QdrantClient(url=QDRANT_URL)

    # 1. Tạo hoặc cấu hình lại Collection trên Qdrant
    # Vector của OpenCLIP ViT-H-14 có kích thước chuẩn là 1024 chiều, dùng khoảng cách COSINE
    vector_size = 1024
    
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"🗑️ Phát hiện collection cũ '{COLLECTION_NAME}', đang tiến hành xóa để làm mới...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"✨ Đang tạo mới collection '{COLLECTION_NAME}' với size = {vector_size} (Cosine distance)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    # 2. Đọc dữ liệu từ file JSON
    print(f"📂 Đang đọc dữ liệu từ file: {JSON_FILE_PATH}...")
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {JSON_FILE_PATH}. Hãy kiểm tra lại đường dẫn!")
        return

    total_items = len(data)
    print(f"📦 Tổng số lượng bản ghi cần đẩy lên Qdrant: {total_items}")

    # 3. Đóng gói dữ liệu thành các PointStruct
    print("📦 Đang chuẩn bị các điểm dữ liệu (points)...")
    points = []
    for idx, item in enumerate(data):
        # Lấy ID từ item, nếu không có thì dùng index tăng dần
        point_id = item.get("id", idx)
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=item["embedding"],  # Mảng vector 1024 chiều
                payload={
                    "video_name": item.get("video_name"),
                    "frame_id": item.get("frame_id"),
                    "pts_time": item.get("pts_time"),
                    "image_path": item.get("image_path")
                }
            )
        )

    # 4. Upload (Upsert) lên Qdrant theo từng batch để tối ưu hiệu suất
    batch_size = 200  # Có thể điều chỉnh từ 100 đến 500 tùy cấu hình máy
    print(f"🚀 Bắt đầu đẩy dữ liệu lên Qdrant theo batch (mỗi batch {batch_size} items)...")

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"✅ Đã upload thành công batch từ bản ghi {i} đến {min(i + batch_size, total_items)} / {total_items}")

    print("🎉 Hoàn tất toàn bộ! Dữ liệu đã được nạp thành công vào Qdrant Database.")

if __name__ == "__main__":
    main()
