import os
import csv
import traceback
import torch
import open_clip
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from PIL import Image
from google import genai

# ==========================================
# CẤU HÌNH THIẾT BỊ VÀ GEMINI API (Full CPU)
# ==========================================
device = "cpu"
print(f"[INFO] Hệ thống đang chạy hoàn toàn trên thiết bị: {device.upper()}")

# Cấu hình Gemini API Key của bạn tại đây
GEMINI_API_KEY = ""
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY.strip():
    print("⚠️  [CẢNH BÁO] Bạn chưa thay GEMINI_API_KEY bằng key thật! "
          "Mọi query sẽ KHÔNG được dịch/tối ưu và sẽ rơi vào fallback (dùng nguyên câu gốc).")

# Định nghĩa các Collection riêng biệt trong Qdrant
IMAGE_COLLECTION_NAME = "jinaV2_images"
ASR_COLLECTION_NAME = "bge-m3audio"
OCR_COLLECTION_NAME = "ocr_collection"
TRAKE_COLLECTION_NAME = "trake_collection"

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
# 1. KHỞI TẠO MÔ HÌNH SEMANTIC (OpenCLIP DFN5B - CPU)
# ==========================================
print("⏳ Đang tải mô hình OpenCLIP (DFN5B) lên CPU...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng trên CPU!")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH ASR & OCR (Embedding Gemma - CPU)
# ==========================================
print("⏳ Đang tải mô hình Embedding Gemma lên CPU...")
GEMMA_EMBEDDING_MODEL_NAME = "google/embedding-gemma"  
bge_model = SentenceTransformer(GEMMA_EMBEDDING_MODEL_NAME, device=device)
print("✅ Embedding Gemma đã sẵn sàng trên CPU!")

# Khởi tạo riêng mô hình OpenCLIP ViT-B-32 cho TraKE (CPU)
print("⏳ Đang tải mô hình OpenCLIP (ViT-B-32) cho TraKE lên CPU...")
trake_model, _, trake_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
trake_tokenizer = open_clip.get_tokenizer("ViT-B-32")
trake_model.eval()
print("✅ OpenCLIP ViT-B-32 cho TraKE đã sẵn sàng trên CPU!")

# ==========================================
# HÀM HỖ TRỢ: DỊCH, TÓM TẮT & TỐI ƯU QUERY BẰNG GEMINI
# ==========================================
def optimize_query_for_clip(raw_query: str) -> str:
    if not raw_query or not raw_query.strip():
        return ""
        
    prompt_instruction = f"""
    Bạn là một chuyên gia tối ưu hóa và dịch thuật câu lệnh tìm kiếm hình ảnh/video cho mô hình OpenCLIP.
    Nhiệm vụ: 
    1. Kiểm tra ngôn ngữ của câu query gốc dưới đây.
    2. Nếu câu query là tiếng Việt, hãy **dịch sang tiếng Anh** chuẩn xác. Nếu câu query đã là tiếng Anh, hãy giữ nguyên tiếng Anh.
    3. Tóm tắt, cô đọng câu đó lại thành một câu ngắn gọn, súc tích, giữ lại toàn bộ các đặc trưng quan trọng nhất (hành động, bối cảnh, đối tượng, màu sắc, trang phục, văn bản xuất hiện).
    
    Yêu cầu bắt buộc:
    - Kết quả trả về PHẢI LÀ TIẾNG ANH.
    - Độ dài tối đa khoảng 40-60 từ.
    - Chỉ trả về duy nhất chuỗi query đã hoàn thiện, tuyệt đối không kèm theo lời giải thích.
    
    Query gốc: "{raw_query}"
    """
    
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt_instruction
        )
        optimized_text = response.text.strip().replace('"', '')
        token_ids = clip_tokenizer([optimized_text])
        num_tokens = int((token_ids != 0).sum().item())
        print(f"[Gemini Translator & Optimizer] Gốc: '{raw_query}' -> Kết quả: '{optimized_text}'")
        return optimized_text
    except Exception as e:
        print(f"[Gemini Error] Không thể xử lý query, dùng tạm query gốc: {e}")
        return raw_query

# ==========================================
# 3. CÁC API ENDPOINTS & NHIỆM VỤ CHI TIẾT
# ==========================================

@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text cho Image"), top_k: int = 50):
    """
    [NHIỆM VỤ]: Semantic Text-to-Image Search.
    - Nhận câu truy vấn văn bản (tiếng Việt hoặc tiếng Anh).
    - Sử dụng Gemini để dịch và tối ưu hóa câu lệnh sang tiếng Anh chuẩn OpenCLIP.
    - Chuyển văn bản thành vector đặc trưng và tìm kiếm các keyframe tương đồng cao nhất trên Qdrant.
    """
    if not prompt.strip():
        return {"results": []}
    try:
        refined_prompt = optimize_query_for_clip(prompt)
        text_tokens = clip_tokenizer([refined_prompt]).to(device)
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


@app.post("/api/search/image")
async def search_image_by_upload(file: UploadFile = File(...), top_k: int = 50):
    """
    [NHIỆM VỤ]: Image-to-Image Search.
    - Nhận một file hình ảnh được tải lên từ phía người dùng.
    - Trích xuất vector đặc trưng hình ảnh bằng OpenCLIP.
    - Truy vấn các khung hình video có nội dung hoặc bối cảnh trực quan tương tự.
    """
    try:
        image_bytes = await file.read()
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.numpy().flatten().tolist()
            
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


@app.get("/api/search-asr")
def search_asr(prompt: str = Query(..., description="Query Text cho ASR"), top_k: int = 50):
    """
    [NHIỆM VỤ]: ASR (Speech-to-Text) Search.
    - Nhận từ khóa hoặc nội dung câu nói cần tìm kiếm.
    - Sử dụng mô hình Embedding Gemma để chuyển câu query thành vector và tìm kiếm trong cơ sở dữ liệu lời thoại video.
    """
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


@app.get("/api/search/ocr")
def search_ocr_text(prompt: str = Query(..., description="Query Text cho OCR"), top_k: int = 50):
    """
    [NHIỆM VỤ]: OCR (Text-in-Video) Search.
    - Tìm kiếm các đoạn văn bản xuất hiện trực tiếp bên trong khung hình video (bảng hiệu, chữ viết, phụ đề).
    - Hỗ trợ tìm kiếm khớp chuỗi trực tiếp (Text Match) và kết hợp tìm kiếm ngữ nghĩa qua vector khi cần thiết.
    """
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


class TrakeRequest(BaseModel):
    trek1: str = ""
    trek2: str = ""
    trek3: str = ""
    trek4: str = ""
    trek5: str = ""


@app.post("/api/search/trake")
def search_trake(req: TrakeRequest):
    """
    [NHIỆM VỤ]: TraKE (Sequential Action Search).
    - Nhận danh sách các bước hành động tuần tự theo thời gian (từ Trek 1 đến Trek 5).
    - Kết hợp các bước thành một chuỗi ngữ cảnh hành động duy nhất và mã hóa qua OpenCLIP ViT-B-32.
    - Truy tìm các đoạn video chứa chuỗi hành động diễn ra liên tiếp đúng thứ tự yêu cầu.
    """
    try:
        queries = [req.trek1, req.trek2, req.trek3, req.trek4, req.trek5]
        active_queries = [q for q in queries if q.strip()]
        
        if not active_queries:
            return {"results": []}
        
        combined_prompt = " -> ".join(active_queries)
        text_tokens = trake_tokenizer([combined_prompt]).to(device)
        
        with torch.no_grad():
            feat = trake_model.encode_text(text_tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            emb = feat.numpy().flatten().tolist()
                
        res = qdrant_client.search(
            collection_name=TRAKE_COLLECTION_NAME, 
            query_vector=emb, 
            limit=50
        )
        
        output = []
        for hit in res:
            p = hit.payload
            output.append({
                "image_path": p.get("image_path"),
                "score": round(hit.score, 4),
                "video_name": p.get("video_name"),
                "frame_id": p.get("frame_id"),
                "pts_time": p.get("pts_time", 0.0)
            })
            
        sorted_output = sorted(output, key=lambda x: x.get('score', 0), reverse=True)
        return {"results": sorted_output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
    """
    [NHIỆM VỤ]: Random Exploration.
    - Tạo một vector ngẫu nhiên chuẩn hóa và truy vấn Qdrant để trả về danh sách các keyframe ngẫu nhiên.
    - Phục vụ mục đích khám phá tập dữ liệu nhanh hoặc kiểm tra giao diện.
    """
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


@app.get("/api/image")
def get_local_image(path: str):
    """
    [NHIỆM VỤ]: Image File Server.
    - Trả về tệp hình ảnh keyframe thực tế lưu trữ trên ổ cứng dựa theo đường dẫn truyền vào.
    """
    if not os.path.isabs(path):
        win_path = os.path.join(BASE_IMAGE_DIR, path)
    else:
        win_path = path.replace("/", "\\")
        
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}


@app.get("/api/video")
def get_local_video(video_name: str):
    """
    [NHIỆM VỤ]: Video Streaming / File Server.
    - Trả về tệp video gốc tương ứng với tên video để xem lại phân cảnh.
    """
    filename = f"{video_name}.mp4" if not video_name.endswith(".mp4") else video_name
    video_path = os.path.join(VIDEO_DIR, filename)
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": f"Video not found at {video_path}"}


@app.post("/api/submit-csv")
def submit_to_csv(
    mode: str = Query("semantic", description="Chế độ hiện tại: semantic, vqa, trake"),
    video_name: str = Query(..., description="Tên video (vd: L30_V057)"),
    frame_id: str = Query(..., description="Frame ID chính (hoặc chuỗi frame cho TraKE)"),
    filename: str = Query(..., description="Tên file CSV muốn lưu"),
    vqa_answer: str = Query("", description="Đáp án VQA nếu có")
):
    """
    [NHIỆM VỤ]: Competition Submission Helper.
    - Tự động đóng gói kết quả tìm kiếm (tên video, mã frame hoặc câu trả lời VQA).
    - Ghi định dạng chuẩn vào file `.csv` để chuẩn bị nộp bài cho các vòng đấu.
    """
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
