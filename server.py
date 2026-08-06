import os
import torch
import open_clip
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from qdrant_client import QdrantClient

device = "cpu"
COLLECTION_NAME = "dfn5b_images"

# Đường dẫn thư mục chứa ảnh keyframe gốc và thư mục chứa video trên máy bạn
BASE_IMAGE_DIR = r"C:\AIC2026\filtered_keyframes"  # hoặc thư mục chứa dataset_webp
VIDEO_DIR = r"C:\AIC2026\video"                   # Thư mục chứa các file L21_V001.mp4,...

app = FastAPI(title="MFusion-VR Qdrant Core API")

# Kích hoạt CORS để Frontend kết nối không bị chặn
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

print("⏳ Loading DFN5B Model & Connecting Qdrant...")
model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
model.eval()

# KẾT NỐI QDRANT LOCAL
qdrant = QdrantClient(host="localhost", port=6333)
print("✅ Backend thông suốt với Qdrant Server!")

@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
        text_tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            query_features = model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.numpy().flatten().tolist()
            
        # Tìm kiếm trong Qdrant
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=top_k
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

@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
    try:
        random_vector = np.random.uniform(-1, 1, 1024).astype(np.float32)
        random_vector /= np.linalg.norm(random_vector)
        
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=random_vector.tolist(), limit=limit
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

@app.get("/api/image")
def get_local_image(path: str):
    # Hỗ trợ cả đường dẫn tuyệt đối hoặc tương đối từ thư mục gốc
    if not os.path.isabs(path):
        win_path = os.path.join(BASE_IMAGE_DIR, path)
    else:
        win_path = path.replace("/", "\\")
        
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}

@app.get("/api/video")
def get_local_video(video_name: str):
    # Trả về file video mp4 tương ứng (vd: L21_V001.mp4)
    filename = f"{video_name}.mp4" if not video_name.endswith(".mp4") else video_name
    video_path = os.path.join(VIDEO_DIR, filename)
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": f"Video not found at {video_path}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
