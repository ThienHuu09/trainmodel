import os
import torch
import open_clip
from PIL import Image
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

device = "cpu"
print("🖥️ Khởi tạo hệ thống nhúng Vector (QDRANT ENGINE - CPU)")

DATASET_DIR = r"C:\AIC2026\dataset"
COLLECTION_NAME = "dfn5b_images"
VECTOR_SIZE = 1024 # Vector đầu ra của mô hình ViT-H-14 DFN5B

# 1. Tải mô hình
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='dfn5b', device=device)
model.eval()

# 2. Kết nối tới Qdrant Local (.exe đang chạy cổng 6333)
qdrant = QdrantClient(host="localhost", port=6333)

# Kiểm tra và tạo Collection nếu chưa có
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Đã tạo mới Collection: {COLLECTION_NAME}")

print(f"⏳ Đang quét danh sách file ảnh trong {DATASET_DIR}...")
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
image_paths = [os.path.join(root, file) for root, _, files in os.walk(DATASET_DIR) for file in files if file.lower().endswith(valid_extensions)]
print(f"📊 Tổng số lượng ảnh tìm thấy: {len(image_paths)}")

# 3. Tiến hành trích xuất dữ liệu theo Batch
BATCH_SIZE = 32 
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Đẩy dữ liệu lên Qdrant"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_images = []
    valid_paths = []
    
    for path in batch_paths:
        try:
            img = Image.open(path).convert('RGB')
            batch_images.append(preprocess(img))
            valid_paths.append(path)
        except Exception:
            continue
            
    if not batch_images:
        continue
        
    image_tensors = torch.stack(batch_images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings = image_features.numpy().tolist()
        
    points = []
    for path, vector in zip(valid_paths, embeddings):
        # Tạo ID UUID cố định dựa trên đường dẫn ảnh
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path))
        clean_path = path.replace("\\", "/") # Chuẩn hóa cho web
        
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={"image_path": clean_path, "filename": os.path.basename(path)}
        ))
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("🎉 XONG! Dữ liệu ảnh đã được lưu trong Qdrant Local. Bạn có thể check tại: http://localhost:6333/dashboard")