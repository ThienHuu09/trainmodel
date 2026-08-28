import os
import csv
import time
import bisect
import traceback
import torch
import open_clip
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Định nghĩa các Collection riêng biệt trong Qdrant
IMAGE_COLLECTION_NAME = "dfn5b_images"
ASR_COLLECTION_NAME = "embeddinggemma_audio"
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
print("✅ Đã kết nối Qdrant thành công! (Không lưu trữ collection trong RAM)")

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
GEMMA_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"  
try:
    # Đổi tên biến bge_model thành embedding_model cho hợp lý
    embedding_model = SentenceTransformer(GEMMA_EMBEDDING_MODEL_NAME, device=device)
    print("✅ Embedding Gemma đã sẵn sàng trên CPU!")
except Exception as e:
    print(f"❌ Lỗi khi tải Embedding Gemma: {e}")
    print("💡 Lưu ý: Hãy đảm bảo bạn đã đăng nhập Hugging Face (`huggingface-cli login`) nếu mô hình yêu cầu quyền truy cập.")

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH TraKE — ViT-L-14 (768 chiều), CHỈ DÙNG NHÁNH TEXT
# ==========================================
# ⚠️ QUAN TRỌNG: collection TRAKE_COLLECTION_NAME BẮT BUỘC phải được embed
# sẵn bằng đúng model ViT-L-14/datacomp_xl_s13b_b90k (768 chiều). Nếu
# collection hiện tại vẫn là dữ liệu cũ (embed bằng ViT-B-32/512 chiều),
# search sẽ lỗi sai số chiều vector. Xác nhận lại trước khi chạy!
TRAKE_MODEL_NAME = "ViT-L-14"
TRAKE_PRETRAINED = "datacomp_xl_s13b_b90k"
TRAKE_VECTOR_SIZE = 768

CANDIDATES_PER_VIDEO_PER_EVENT = 30  # số điểm tốt nhất lấy ra / video / sự kiện
EXACT_SEARCH = True                   # True = duyệt chính xác (chậm hơn ANN nhưng đúng 100%)
HNSW_EF = 256                          # chỉ có tác dụng khi EXACT_SEARCH = False

trake_device = "cpu"  # ép CPU thuần, không dùng CUDA nữa
print(f"⏳ Đang tải mô hình TraKE {TRAKE_MODEL_NAME} ({TRAKE_PRETRAINED}) trên CPU...")
trake_model, _, trake_preprocess = open_clip.create_model_and_transforms(
    TRAKE_MODEL_NAME, pretrained=TRAKE_PRETRAINED, device="cpu"
)
trake_tokenizer = open_clip.get_tokenizer(TRAKE_MODEL_NAME)
trake_model.eval()

print("🗑️  Đang xóa nhánh ảnh (visual tower) vì server chỉ cần encode text cho TraKE...")
del trake_model.visual

trake_model = trake_model.to(trake_device)  # giữ fp32, không .half() vì fp16 trên CPU không được hỗ trợ tốt
print(f"✅ TraKE ({TRAKE_MODEL_NAME}, nhánh text) đã sẵn sàng trên {trake_device} "
      f"({TRAKE_VECTOR_SIZE} chiều)!")

# Chỉ quét tên video (không tải vector) để biết số video tối đa cho Groups API
print("⏳ Đang quét danh sách tên video trong TRAKE collection (không tải vector)...")
_t0 = time.time()
_distinct_videos = set()
_next_offset = None
try:
    while True:
        _points, _next_offset = qdrant_client.scroll(
            collection_name=TRAKE_COLLECTION_NAME,
            limit=10000,
            offset=_next_offset,
            with_payload=["video_name"],
            with_vectors=False
        )
        for _p in _points:
            _distinct_videos.add(_p.payload.get("video_name", "unknown"))
        if _next_offset is None:
            break
    TOTAL_TRAKE_VIDEOS = len(_distinct_videos)
except Exception as e:
    print(f"[WARN] Không quét được TraKE collection (có thể chưa tồn tại): {e}")
    TOTAL_TRAKE_VIDEOS = 0

MAX_TRAKE_GROUPS = max(TOTAL_TRAKE_VIDEOS + 50, 100)
print(f"✅ Có {TOTAL_TRAKE_VIDEOS} video trong '{TRAKE_COLLECTION_NAME}' "
      f"(quét xong trong {time.time() - _t0:.1f}s).")

try:
    qdrant_client.create_payload_index(
        collection_name=TRAKE_COLLECTION_NAME,
        field_name="video_name",
        field_schema=qmodels.PayloadSchemaType.KEYWORD
    )
except Exception:
    pass

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
# THUẬT TOÁN DP CHO TraKE (khớp chuỗi sự kiện tuần tự theo thời gian)
# ==========================================
def find_best_trake_dynamic(
    candidates_by_video,
    num_events,
    top_k=5,
    max_duration_sec=300.0,
    gap_penalty=0.0008,
    max_seq_per_video=2
):
    all_sequences = []

    for video_name, candidates in candidates_by_video.items():
        event_lists = []
        ok = True
        for e in range(num_events):
            items = candidates.get(e, [])
            if not items:
                ok = False
                break
            dedup = {}
            for pts, score, fid, path in items:
                if fid not in dedup or score > dedup[fid][1]:
                    dedup[fid] = (pts, score, fid, path)
            sorted_items = sorted(dedup.values(), key=lambda x: x[2])
            event_lists.append(sorted_items)

        if not ok:
            continue

        dp_all = []
        first_list = event_lists[0]
        dp_all.append([
            {"score": score, "prev": -1, "start_time": pts}
            for (pts, score, fid, path) in first_list
        ])

        for e in range(1, num_events):
            prev_list = event_lists[e - 1]
            prev_dp = dp_all[e - 1]
            prev_frame_ids = [item[2] for item in prev_list]

            prefix_max_score, prefix_max_idx = [], []
            best_score, best_idx = float("-inf"), -1
            for i, node in enumerate(prev_dp):
                if node["score"] > best_score:
                    best_score, best_idx = node["score"], i
                prefix_max_score.append(best_score)
                prefix_max_idx.append(best_idx)

            cur_list = event_lists[e]
            cur_dp = []
            for (pts, score, fid, path) in cur_list:
                pos = bisect.bisect_left(prev_frame_ids, fid) - 1
                if pos < 0:
                    cur_dp.append({"score": float("-inf"), "prev": -1, "start_time": pts})
                    continue

                best_prev_idx = prefix_max_idx[pos]
                best_prev_score = prefix_max_score[pos]
                prev_pts = prev_list[best_prev_idx][0]
                prev_start_time = prev_dp[best_prev_idx]["start_time"]

                gap = max(0.0, pts - prev_pts)
                total_score = best_prev_score + score - gap_penalty * gap

                cur_dp.append({"score": total_score, "prev": best_prev_idx, "start_time": prev_start_time})

            dp_all.append(cur_dp)

        last_list = event_lists[-1]
        last_dp = dp_all[-1]
        finalists = []
        for i, node in enumerate(last_dp):
            if node["score"] == float("-inf"):
                continue
            last_time = last_list[i][0]
            if (last_time - node["start_time"]) > max_duration_sec:
                continue
            finalists.append((node["score"], i))

        finalists.sort(key=lambda x: x[0], reverse=True)

        used_frame_ids = set()
        accepted = 0
        for score, end_idx in finalists:
            if accepted >= max_seq_per_video:
                break
            seq = []
            idx = end_idx
            for e in range(num_events - 1, -1, -1):
                pts, sc, fid, path = event_lists[e][idx]
                seq.append({
                    "video_name": video_name,
                    "pts_time": pts,
                    "score": float(sc),
                    "frame_id": fid,
                    "image_path": path
                })
                idx = dp_all[e][idx]["prev"]
            seq.reverse()

            seq_frame_ids = {item["frame_id"] for item in seq}
            if seq_frame_ids & used_frame_ids:
                continue

            used_frame_ids |= seq_frame_ids
            accepted += 1
            all_sequences.append({
                "video_name": video_name,
                "total_score": float(score),
                "sequence": seq
            })

    all_sequences.sort(key=lambda x: x["total_score"], reverse=True)
    return all_sequences[:top_k]


def _query_trake_event_groups(event_idx: int, vector: list):
    """Gọi Qdrant Groups API cho 1 sự kiện TraKE, trả về (event_idx, list các group theo video)."""
    search_params = qmodels.SearchParams(
        exact=EXACT_SEARCH,
        hnsw_ef=None if EXACT_SEARCH else HNSW_EF
    )
    result = qdrant_client.query_points_groups(
        collection_name=TRAKE_COLLECTION_NAME,
        query=vector,
        group_by="video_name",
        limit=MAX_TRAKE_GROUPS,
        group_size=CANDIDATES_PER_VIDEO_PER_EVENT,
        with_payload=True,
        with_vectors=False,
        search_params=search_params,
    )
    return event_idx, result.groups


# ==========================================
# 3. CÁC API ENDPOINTS & NHIỆM VỤ CHI TIẾT
# ==========================================

@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text cho Image"), top_k: int = 50):
    """
    [NHIỆM VỤ]: Semantic Text-to-Image Search.
    - Nhận câu truy vấn văn bản (tiếng Việt hoặc tiếng Anh).
    - Sử dụng Gemini để dịch và tối ưu hóa câu lệnh sang tiếng Anh chuẩn OpenCLIP.
    - Chuyển văn bản thành vector đặc trưng và tìm kiếm trực tiếp trên Qdrant Server.
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

        search_result = qdrant_client.query_points(
            collection_name=IMAGE_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        ).points

        output = []
        for p in search_result:
            payload = p.payload or {}
            output.append({
                "image_path": payload.get("image_path"),
                "score": round(float(p.score), 4),
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
    - Truy vấn trực tiếp trên Qdrant Server.
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

        search_result = qdrant_client.query_points(
            collection_name=IMAGE_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        ).points

        output = []
        for p in search_result:
            payload = p.payload or {}
            output.append({
                "image_path": payload.get("image_path"),
                "score": round(float(p.score), 4),
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
    - Sử dụng mô hình Embedding Gemma để chuyển câu query thành vector và tìm kiếm trực tiếp trên Qdrant Server.
    """
    if not prompt.strip():
        return {"results": []}
    try:
        # Dùng encode_query() thay vì encode() thường -> tự động thêm đúng
        # prompt "task: search result | query: " theo chuẩn EmbeddingGemma,
        # khớp với cách corpus được encode lúc index (encode_document()).
        query_vector = embedding_model.encode_query(prompt).astype(np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_list = query_vector.tolist()

        search_result = qdrant_client.query_points(
            collection_name=ASR_COLLECTION_NAME,
            query=query_list,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        ).points

        output = []
        for p in search_result:
            payload = p.payload or {}
            output.append({
                "text": payload.get("text"),
                "score": round(float(p.score), 4),
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
    - Tìm kiếm các đoạn văn bản xuất hiện trực tiếp bên trong khung hình video trên Qdrant Server.
    """
    if not prompt.strip():
        return {"results": []}
    try:
        prompt_lower = prompt.lower()

        # Thử tìm kiếm chính xác bằng Qdrant filter
        try:
            scroll_result, _ = qdrant_client.scroll(
                collection_name=OCR_COLLECTION_NAME,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="text",
                            match=qmodels.MatchText(text=prompt)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            if scroll_result:
                output = []
                for p in scroll_result:
                    payload = p.payload or {}
                    output.append({
                        "ocr_text": payload.get("text") or payload.get("ocr_text") or payload.get("ocr", ""),
                        "score": 1.0,
                        "video_name": payload.get("video_name"),
                        "image_path": payload.get("image_path"),
                        "frame_id": payload.get("frame_id"),
                        "pts_time": payload.get("pts_time", 0.0)
                    })
                return {"results": output}
        except Exception:
            pass

        # Fallback: Dense search trên Qdrant — dùng encode_query() thay vì
        # encode() thường, khớp chuẩn EmbeddingGemma (xem giải thích ở /api/search-asr)
        query_vector = embedding_model.encode_query(prompt).astype(np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_list = query_vector.tolist()

        search_result = qdrant_client.query_points(
            collection_name=OCR_COLLECTION_NAME,
            query=query_list,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        ).points

        output = []
        for p in search_result:
            payload = p.payload or {}
            output.append({
                "ocr_text": payload.get("text") or payload.get("ocr_text") or payload.get("ocr", ""),
                "score": round(float(p.score), 4),
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
    [NHIỆM VỤ]: TraKE (Sequential Action Search) — bản nâng cấp.
    - Encode từng sự kiện (trek1..trek5) RIÊNG BIỆT bằng ViT-L-14.
    - Với mỗi sự kiện, gọi Qdrant Groups API (group theo video_name) song song
      bằng ThreadPoolExecutor để lấy top candidate MỖI VIDEO.
    - Đưa candidates vào thuật toán DP (find_best_trake_dynamic) để tìm chuỗi
      sự kiện đúng thứ tự thời gian, khớp nhất trong từng video.
    """
    try:
        _t0 = time.time()
        queries = [req.trek1, req.trek2, req.trek3, req.trek4, req.trek5]
        active_queries = [q for q in queries if q.strip()]

        if not active_queries:
            return {"results": []}

        num_events = len(active_queries)
        text_tokens = trake_tokenizer(active_queries).to(trake_device)

        with torch.no_grad():
            text_feats = trake_model.encode_text(text_tokens)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            text_vecs = text_feats.float().cpu().numpy()  # (num_events, TRAKE_VECTOR_SIZE)

        video_candidates = {}
        with ThreadPoolExecutor(max_workers=num_events) as executor:
            futures = [
                executor.submit(_query_trake_event_groups, e_idx, text_vecs[e_idx].tolist())
                for e_idx in range(num_events)
            ]
            for fut in as_completed(futures):
                e_idx, groups = fut.result()
                for g in groups:
                    v_name = g.id
                    video_candidates.setdefault(v_name, {})
                    items = []
                    for hit in g.hits:
                        payload = hit.payload or {}
                        items.append((
                            float(payload.get("pts_time", 0.0)),
                            float(hit.score),
                            int(payload.get("frame_id", 0)),
                            payload.get("image_path", "")
                        ))
                    video_candidates[v_name][e_idx] = items

        top_seqs = find_best_trake_dynamic(video_candidates, num_events, top_k=5)

        output = []
        seen_paths = set()
        for seq in top_seqs:
            for item in seq["sequence"]:
                if item["image_path"] in seen_paths:
                    continue
                seen_paths.add(item["image_path"])
                output.append({
                    "video_name": item["video_name"],
                    "frame_id": item["frame_id"],
                    "pts_time": item["pts_time"],
                    "image_path": item["image_path"],
                    "score": round(item["score"], 4)
                })

        print(f"⏱️  TraKE search hoàn tất trong {time.time() - _t0:.2f}s "
              f"({num_events} sự kiện, exact={EXACT_SEARCH}).")

        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
    """
    [NHIỆM VỤ]: Random Exploration.
    - Lấy danh sách các keyframe ngẫu nhiên từ Qdrant Server.
    """
    try:
        scroll_result, _ = qdrant_client.scroll(
            collection_name=IMAGE_COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        output = []
        for p in scroll_result:
            payload = p.payload or {}
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
