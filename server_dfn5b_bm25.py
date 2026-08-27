import os
import csv
import time
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
GEMINI_API_KEY = "AQ.Ab8RN6JtY-EGSIMresVONNUBhTpNPLeDjK0lgfivTaUSg8mmVw"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY.strip():
    print("⚠️  [CẢNH BÁO] Bạn chưa thay GEMINI_API_KEY bằng key thật! "
          "Mọi query sẽ KHÔNG được dịch/tối ưu và sẽ rơi vào fallback (dùng nguyên câu gốc).")

# Định nghĩa các Collection (Chỉ nạp dfn5b_images và trake_collection vào RAM theo yêu cầu)
IMAGE_COLLECTION_NAME = "dfn5b_images"
ASR_COLLECTION_NAME = "bge-m3audio"
OCR_COLLECTION_NAME = "ocr_collection"
TRAKE_COLLECTION_NAME = "trake_collection"

# Đường dẫn thư mục chứa ảnh keyframe gốc và thư mục chứa video trên máy bạn
BASE_IMAGE_DIR = r"C:\AIC2026\dataset_webp" 
VIDEO_DIR = r"C:\AIC2026\video"              

app = FastAPI(title="MFusion-VR Core API (DFN5B & TraKE on RAM)")

# Kích hoạt CORS để Frontend kết nối không bị chặn
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

print("⏳ Đang kết nối Qdrant Server...")
# Tăng timeout lên 60.0 giây để tránh lỗi ReadTimeout khi load dữ liệu lớn
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=60.0)
print("✅ Đã kết nối Qdrant thành công!")


# ==========================================
# ⚡ HÀM NẠP COLLECTION VÀO RAM (Đã tối ưu batch & timeout)
# ==========================================
def load_collection_to_ram(collection_name: str):
    print(f"⏳ Đang tải collection '{collection_name}' vào RAM...")
    t0 = time.time()

    if not qdrant_client.collection_exists(collection_name=collection_name):
        print(f"[WARN] Collection '{collection_name}' không tồn tại, bỏ qua.")
        return np.zeros((0, 0), dtype=np.float32), []

    vector_batches = []
    payloads = []
    next_offset = None
    loaded = 0

    while True:
        try:
            # Giảm limit xuống 1000 để gói dữ liệu nhẹ hơn, tránh nghẽn mạng và timeout
            points, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            print(f"[ERROR] Lỗi khi scroll collection '{collection_name}' tại offset {next_offset}: {e}")
            break

        if not points:
            break

        batch_matrix = np.asarray([p.vector for p in points], dtype=np.float32)
        vector_batches.append(batch_matrix)
        for p in points:
            payloads.append(p.payload or {})

        loaded += len(points)
        print(f"   Đã tải {loaded} điểm của '{collection_name}'...", end="\r")

        if next_offset is None:
            break

    print() # Xuống dòng sau khi load xong batch
    if not vector_batches:
        print(f"[WARN] Collection '{collection_name}' rỗng.")
        return np.zeros((0, 0), dtype=np.float32), []

    matrix = np.concatenate(vector_batches, axis=0)
    del vector_batches  

    ram_mb = matrix.nbytes / (1024 ** 2)
    print(f"✅ '{collection_name}': {matrix.shape[0]} điểm, {matrix.shape[1]} chiều, "
          f"{ram_mb:.1f} MB, tải trong {time.time() - t0:.1f}s")
    return matrix, payloads


def top_k_indices(sims: np.ndarray, k: int):
    n = len(sims)
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])]


print("⏳ Đang nạp 2 collection (dfn5b_images & trake_collection) vào RAM...")
SEMANTIC_MATRIX, SEMANTIC_PAYLOADS = load_collection_to_ram(IMAGE_COLLECTION_NAME)
TRAKE_MATRIX, TRAKE_PAYLOADS = load_collection_to_ram(TRAKE_COLLECTION_NAME)

_total_ram_mb = sum(
    m.nbytes / (1024 ** 2) for m in [SEMANTIC_MATRIX, TRAKE_MATRIX]
)
print(f"🚀 Đã nạp xong vào RAM (~{_total_ram_mb:.0f} MB). Các API Semantic và TraKE chạy trực tiếp trên RAM tốc độ cao.")

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH SEMANTIC (OpenCLIP DFN5B - CPU)
# ==========================================
print("⏳ Đang tải mô hình OpenCLIP (DFN5B) lên CPU...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng trên CPU!")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH ASR & OCR (EmbeddingGemma 300M - CPU)
# ==========================================
print("⏳ Đang tải mô hình EmbeddingGemma (300M) lên CPU...")
GEMMA_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"  
bge_model = SentenceTransformer(GEMMA_EMBEDDING_MODEL_NAME, device=device)
print("✅ EmbeddingGemma đã sẵn sàng trên CPU!")

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
    [NHIỆM VỤ]: Semantic Text-to-Image Search (Chạy trực tiếp trên RAM với dfn5b).
    """
    if not prompt.strip():
        return {"results": []}
    try:
        refined_prompt = optimize_query_for_clip(prompt)
        text_tokens = clip_tokenizer([refined_prompt]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.numpy().flatten().astype(np.float32)

        if SEMANTIC_MATRIX.shape[0] == 0:
            return {"results": [], "error": f"Collection '{IMAGE_COLLECTION_NAME}' rỗng hoặc chưa nạp."}

        sims = SEMANTIC_MATRIX @ query_embedding  
        top_idx = top_k_indices(sims, top_k)

        output = []
        for i in top_idx:
            p = SEMANTIC_PAYLOADS[i]
            output.append({
                "image_path": p.get("image_path"),
                "score": round(float(sims[i]), 4),
                "video_name": p.get("video_name"),
                "frame_id": p.get("frame_id"),
                "pts_time": p.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/search/image")
async def search_image_by_upload(file: UploadFile = File(...), top_k: int = 50):
    """
    [NHIỆM VỤ]: Image-to-Image Search (Chạy trực tiếp trên RAM với dfn5b).
    """
    try:
        image_bytes = await file.read()
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.numpy().flatten().astype(np.float32)

        if SEMANTIC_MATRIX.shape[0] == 0:
            return {"results": [], "error": f"Collection '{IMAGE_COLLECTION_NAME}' rỗng hoặc chưa nạp."}

        sims = SEMANTIC_MATRIX @ query_embedding
        top_idx = top_k_indices(sims, top_k)

        output = []
        for i in top_idx:
            p = SEMANTIC_PAYLOADS[i]
            output.append({
                "image_path": p.get("image_path"),
                "score": round(float(sims[i]), 4),
                "video_name": p.get("video_name"),
                "frame_id": p.get("frame_id"),
                "pts_time": p.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/search-asr")
def search_asr(prompt: str = Query(..., description="Query Text cho ASR"), top_k: int = 50):
    """
    [NHIỆM VỤ]: ASR (Speech-to-Text) Search (Truy vấn trực tiếp Qdrant do không nạp RAM).
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
    [NHIỆM VỤ]: OCR (Text-in-Video) Search (Truy vấn trực tiếp Qdrant do không nạp RAM).
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
    [NHIỆM VỤ]: TraKE (Sequential Action Search - Chạy trực tiếp trên RAM).
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
            emb = feat.numpy().flatten().astype(np.float32)

        if TRAKE_MATRIX.shape[0] == 0:
            return {"results": [], "error": f"Collection '{TRAKE_COLLECTION_NAME}' rỗng hoặc chưa nạp."}

        sims = TRAKE_MATRIX @ emb
        top_idx = top_k_indices(sims, 50)

        output = []
        for i in top_idx:
            p = TRAKE_PAYLOADS[i]
            output.append({
                "image_path": p.get("image_path"),
                "score": round(float(sims[i]), 4),
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
    [NHIỆM VỤ]: Random Exploration (Lấy ngẫu nhiên từ RAM của dfn5b).
    """
    try:
        n = len(SEMANTIC_PAYLOADS)
        if n == 0:
            return {"results": [], "error": f"Collection '{IMAGE_COLLECTION_NAME}' rỗng hoặc chưa nạp."}

        k = min(limit, n)
        random_idx = np.random.choice(n, size=k, replace=False)

        output = []
        for i in random_idx:
            p = SEMANTIC_PAYLOADS[i]
            output.append({
                "image_path": p.get("image_path"),
                "score": "RAND",
                "video_name": p.get("video_name"),
                "frame_id": p.get("frame_id"),
                "pts_time": p.get("pts_time", 0.0)
            })
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/image")
def get_local_image(path: str):
    """
    [NHIỆM VỤ]: Image File Server.
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
