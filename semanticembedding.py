import os
import json
import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Cấu hình thiết bị và thông số kết nối
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Đang sử dụng thiết bị: {device}")

client = QdrantClient("http://localhost:6333")
IMAGE_COLLECTION_NAME = "dfn5b_images"

# Đường dẫn thư mục chứa ảnh keyframe trên máy của bạn (khớp với server.py)
BASE_IMAGE_DIR = r"C:\AIC2026\filtered_keyframes"

# 2. Khởi tạo mô hình OpenCLIP (DFN5B)
print("⏳ Đang tải mô hình OpenCLIP (ViT-H-14-quickgelu, dfn5b)...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_model.eval()
print("✅ Mô hình OpenCLIP đã sẵn sàng!")

# 3. Tạo lại collection trên Qdrant (Kích thước vector của ViT-H-14 thường là 1024 chiều)
if client.collection_exists(IMAGE_COLLECTION_NAME):
    client.delete_collection(IMAGE_COLLECTION_NAME)
    print(f"🗑️ Đã xóa collection cũ '{IMAGE_COLLECTION_NAME}'.")

client.create_collection(
    collection_name=IMAGE_COLLECTION_NAME,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
print(f"✨ Đã tạo mới collection '{IMAGE_COLLECTION_NAME}' thành công!")

# 4. Duyệt qua thư mục ảnh để trích xuất vector và đẩy lên Qdrant
points = []
global_id = 0
batch_size = 100

print(f"📂 Đang quét thư mục ảnh: {BASE_IMAGE_DIR}")

if os.path.exists(BASE_IMAGE_DIR):
    # Duyệt qua từng thư mục video (vd: L21_V001, L21_V002...)
    for video_name in sorted(os.listdir(BASE_IMAGE_DIR)):
        video_dir = os.path.join(BASE_IMAGE_DIR, video_name)
        
        if os.path.isdir(video_dir):
            print(f"🎬 Đang xử lý video folder: {video_name}")
            
            for img_file in sorted(os.listdir(video_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(video_dir, img_file)
                    
                    # Lấy frame_id từ tên file (vd: 100.webp -> 100)
                    frame_id_str = os.path.splitext(img_file)[0]
                    try:
                        frame_id = int(frame_id_str)
                    except ValueError:
                        frame_id = 0
                        
                    pts_time = float(frame_id / 25.0) # Giả lập 25 fps
                    image_path_rel = f"{video_name}/{img_file}"
                    
                    try:
                        # Đọc ảnh và trích xuất vector đặc trưng bằng OpenCLIP
                        image = Image.open(img_path).convert("RGB")
                        image_tensor = preprocess(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image_tensor)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            embedding_vector = image_features.cpu().numpy().flatten().tolist()
                            
                        # Đóng gói điểm dữ liệu
                        points.append(
                            models.PointStruct(
                                id=global_id,
                                vector=embedding_vector,
                                payload={
                                    "video_name": video_name,
                                    "frame_id": frame_id,
                                    "pts_time": pts_time,
                                    "image_path": image_path_rel
                                }
                            )
                        )
                        global_id += 1
                        
                        # Thực hiện upsert theo batch để tiết kiệm bộ nhớ
                        if len(points) >= batch_size:
                            client.upsert(collection_name=IMAGE_COLLECTION_NAME, points=points)
                            print(f"✅ Đã đẩy batch lên tới ID: {global_id - 1}")
                            points = []
                            
                    except Exception as e:
                        print(f"⚠️ Lỗi xử lý ảnh {img_path}: {e}")

    # Đẩy nốt số lượng điểm dư còn lại trong batch cuối cùng
    if points:
        client.upsert(collection_name=IMAGE_COLLECTION_NAME, points=points)
        print(f"✅ Đã đẩy nốt batch cuối cùng. Tổng số lượng ảnh: {global_id}")

else:
    print(f"❌ Không tìm thấy đường dẫn thư mục ảnh: {BASE_IMAGE_DIR}")

print("🎉 Hoàn tất toàn bộ quá trình đưa dữ liệu hình ảnh DFN5B lên Qdrant!")
