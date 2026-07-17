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
model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='dfn5b', device=device)
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.eval()

# KẾT NỐI QDRANT LOCAL (Tuyệt đối không dùng dòng chroamba cũ gây lỗi)
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
            output.append({
                "image_path": hit.payload["image_path"],
                "score": round(hit.score, 4),
                "filename": hit.payload["filename"]
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
            output.append({
                "image_path": hit.payload["image_path"],
                "score": "RAND",
                "filename": hit.payload["filename"]
            })
        return {"results": output}
    except Exception as e:
        return {"results": []}

@app.get("/api/image")
def get_local_image(path: str):
    win_path = path.replace("/", "\\")
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
