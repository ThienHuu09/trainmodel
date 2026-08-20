import os
import csv
import torch
import open_clip
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

device = "cpu"

# Định nghĩa các Collection riêng biệt
IMAGE_COLLECTION_NAME = "dfn5b_images"
ASR_COLLECTION_NAME = "bge-m3audio"
OCR_COLLECTION_NAME = "ocrtoi"

# Đường dẫn thư mục chứa ảnh keyframe gốc và thư mục chứa video trên máy bạn
BASE_IMAGE_DIR = r"C:\AIC2026\dataset_webp" 
VIDEO_DIR = r"C:\AIC2026\video"                 

app = FastAPI(title="MFusion-VR Full Core API (Semantic + ASR + OCR + TraKE)")

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
# 1. KHỞI TẠO MÔ HÌNH SEMANTIC & TraKE (OpenCLIP)
# ==========================================
print("⏳ Đang tải mô hình OpenCLIP (DFN5B)...")
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng!")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH ASR & OCR (BGE-M3)
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


# API 2: Tìm kiếm ASR (Logic từ file pipeline - Lời thoại qua BGE-M3)
@app.get("/api/search-asr")
def search_asr(prompt: str = Query(..., description="Query Text cho ASR"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
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


# API 3: Tìm kiếm OCR (Logic từ file server2 - Văn bản trong hình qua collection ocrtoi)
@app.get("/api/search-ocr")
def search_ocr_text(prompt: str = Query(..., description="Query Text cho OCR"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
        records, _ = qdrant_client.scroll(
            collection_name=OCR_COLLECTION_NAME,
            scroll_filter=qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key="text", match=qmodels.MatchText(text=prompt)),
                    qmodels.FieldCondition(key="ocr_text", match=qmodels.MatchText(text=prompt)),
                    qmodels.FieldCondition(key="ocr", match=qmodels.MatchText(text=prompt))
                ]
            ),
            limit=top_k,
            with_payload=True
        )
        
        if not records:
            query_vector = bge_model.encode(prompt).tolist()
            search_response = qdrant_client.query_points(
                collection_name=OCR_COLLECTION_NAME, 
                query=query_vector, 
                limit=top_k
            )
            points_list = search_response.points
        else:
            points_list = records

        output = []
        for hit in points_list:
            payload = getattr(hit, 'payload', {}) or {}
            score = getattr(hit, 'score', 1.0)
            output.append({
                "ocr_text": payload.get("text") or payload.get("ocr_text") or payload.get("ocr", ""),
                "score": round(score, 4) if isinstance(score, float) else 1.0,
                "video_name": payload.get("video_name"),
                "image_path": payload.get("image_path"),
                "frame_id": payload.get("frame_id"),
                "pts_time": payload.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


# API 4: Tìm kiếm TraKE (Logic từ file server1 - Temporal/Sequential Search)
@app.get("/api/trake")
def search_trake(
    q1: str = Query(""), q2: str = Query(""), 
    q3: str = Query(""), q4: str = Query(""), 
    top_k: int = 50
):
    try:
        queries = [q for q in [q1, q2, q3, q4] if q.strip()]
        if not queries:
            return {"results": []}
        
        combined_prompt = " -> ".join(queries)
        text_tokens = clip_tokenizer([combined_prompt]).to(device)
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


# API 5: Lấy danh sách ngẫu nhiên keyframe
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


# API 6: Trả về file ảnh tĩnh
@app.get("/api/image")
def get_local_image(path: str):
    if not os.path.isabs(path):
        win_path = os.path.join(BASE_IMAGE_DIR, path)
    else:
        win_path = path.replace("/", "\\")
        
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}


# API 7: Trả về file video MP4
@app.get("/api/video")
def get_local_video(video_name: str):
    filename = f"{video_name}.mp4" if not video_name.endswith(".mp4") else video_name
    video_path = os.path.join(VIDEO_DIR, filename)
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": f"Video not found at {video_path}"}


# API 8: Tạo và lưu file CSV Submission (Hỗ trợ Semantic/OCR/ASR, VQA và TraKE)
@app.post("/api/submit-csv")
def submit_to_csv(
    mode: str = Query("semantic", description="Chế độ hiện tại: semantic, vqa, trake"),
    video_name: str = Query(..., description="Tên video (vd: L30_V057)"),
    frame_id: str = Query(..., description="Frame ID chính (hoặc chuỗi frame cho TraKE)"),
    filename: str = Query(..., description="Tên file CSV muốn lưu"),
    vqa_answer: str = Query("", description="Đáp án VQA nếu có")
):
    try:
        output_dir = r"C:\Users\XPS 15 9570\Downloads\submission"
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename.endswith(".csv"):
            filename += ".csv"
            
        file_path = os.path.join(output_dir, filename)
        clean_video_name = video_name.strip()
        
        row_data = []
        
        if mode == "trake":
            frames = [f.replace("#", "").strip() for f in frame_id.split(",")]
            row_data = [clean_video_name] + frames
            
        elif mode == "vqa":
            clean_frame_id = frame_id.replace("#", "").strip()
            answer = vqa_answer.strip()
            
            if "," in answer or '"' in answer:
                escaped_answer = answer.replace('"', '""')
                formatted_answer = f'"{escaped_answer}"'
            else:
                formatted_answer = answer
                
            row_data = [clean_video_name, clean_frame_id, formatted_answer]
            
        else:
            clean_frame_id = frame_id.replace("#", "").strip()
            row_data = [clean_video_name, clean_frame_id]
            
        with open(file_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row_data)
            
        return {"status": "success", "message": f"Đã lưu thành công vào {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
