import os
import torch
import open_clip
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

device = "cpu"

# Định nghĩa các Collection riêng biệt
IMAGE_COLLECTION_NAME = "dfn5b_images"
ASR_COLLECTION_NAME = "bge-m3audio"

# Đường dẫn thư mục chứa ảnh keyframe gốc và thư mục chứa video trên máy bạn
BASE_IMAGE_DIR = r"C:\AIC2026\filtered_keyframes" 
VIDEO_DIR = r"C:\AIC2026\video"                 

app = FastAPI(title="MFusion-VR Full Core API (Semantic + ASR)")

# Kích hoạt CORS để Frontend kết nối không bị chặn
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

print("⏳ Đang kết nối Qdrant Server...")
qdrant_client = QdrantClient(host="localhost", port=6333)
print("✅ Đã kết nối Qdrant thành công!")

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH SEMANTIC (OpenCLIP)
# ==========================================
print("⏳ Đang tải mô hình OpenCLIP (DFN5B)...")
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng!")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH ASR (BGE-M3)
# ==========================================
print("⏳ Đang tải mô hình BGE-M3...")
bge_model = SentenceTransformer('BAAI/bge-m3', device=device)
print("✅ BGE-M3 đã sẵn sàng!")

# ==========================================
# 3. CÁC API ENDPOINTS
# ==========================================

# API 1: Tìm kiếm Semantic Hình ảnh/Keyframe
@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text cho Image"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
        text_tokens = clip_tokenizer([prompt]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.numpy().flatten().tolist()
            
        search_result = qdrant_client.search(
            collection_name=IMAGE_COLLECTION_NAME, query_vector=query_embedding, limit=top_k
        )
        
        output = []
        for hit in search_result:
            payload = hit.payload
            output.append({
                "image_path": payload.get("image_path"),
                "score": round(hit.score, 4),
                "video_name": payload.get("video_name"),
                "frame_id": payload.get("frame_id"),
                "pts_time": payload.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


# API 2: Tìm kiếm ASR (Lời thoại qua BGE-M3)
@app.get("/api/search-asr")
def search_asr(prompt: str = Query(..., description="Query Text cho ASR"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
        # Mã hóa câu query thành vector bằng BGE-M3 SentenceTransformer
        query_vector = bge_model.encode(prompt).tolist()
            
        search_response = qdrant_client.query_points(
            collection_name=ASR_COLLECTION_NAME, 
            query=query_vector, 
            limit=top_k
        )
        
        output = []
        for hit in search_response.points:
            payload = hit.payload
            output.append({
                "text": payload.get("text"),
                "score": round(hit.score, 4),
                "video_name": payload.get("video_name"),
                "image_path": payload.get("image_path"),
                "audio_path": payload.get("audio_path"),
                "pts_time": payload.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


# API 3: Lấy danh sách ngẫu nhiên keyframe
@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
    try:
        random_vector = np.random.uniform(-1, 1, 1024).astype(np.float32)
        random_vector /= np.linalg.norm(random_vector)
        
        search_result = qdrant_client.search(
            collection_name=IMAGE_COLLECTION_NAME, query_vector=random_vector.tolist(), limit=limit
        )
        
        output = []
        for hit in search_result:
            payload = hit.payload
            output.append({
                "image_path": payload.get("image_path"),
                "score": "RAND",
                "video_name": payload.get("video_name"),
                "frame_id": payload.get("frame_id"),
                "pts_time": payload.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


# API 4: Trả về file ảnh tĩnh
@app.get("/api/image")
def get_local_image(path: str):
    if not os.path.isabs(path):
        win_path = os.path.join(BASE_IMAGE_DIR, path)
    else:
        win_path = path.replace("/", "\\")
        
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}


# API 5: Trả về file video MP4
@app.get("/api/video")
def get_local_video(video_name: str):
    filename = f"{video_name}.mp4" if not video_name.endswith(".mp4") else video_name
    video_path = os.path.join(VIDEO_DIR, filename)
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": f"Video not found at {video_path}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
