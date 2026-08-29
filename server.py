import os
import csv
import time
import bisect
import hashlib
import pickle
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

# ==========================================
# CẤU HÌNH THIẾT BỊ (Full CPU)
# ==========================================
device = "cpu"
print(f"[INFO] Hệ thống đang chạy hoàn toàn trên thiết bị: {device.upper()}")

# Định nghĩa các Collection riêng biệt trong Qdrant
IMAGE_COLLECTION_NAME = "dfn5b_images"   # dùng chung cho cả Semantic Search VÀ TraKE
ASR_COLLECTION_NAME = "embeddinggemma_audio"

# File cache lưu sẵn hash index (build 1 lần, các lần chạy sau load lại cho nhanh)
HASH_INDEX_CACHE_PATH = "hash_index.pkl"

# Đường dẫn thư mục chứa ảnh keyframe gốc và thư mục chứa video trên máy bạn
BASE_IMAGE_DIR = r"C:\AIC2026\dataset_webp" 
VIDEO_DIR = r"C:\AIC2026\video"                 

app = FastAPI(title="MFusion-VR Full Core API (Semantic + ASR + Reverse Lookup + TraKE)")

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
# HASH INDEX CHO REVERSE IMAGE LOOKUP (thay thế tính năng OCR)
# ==========================================
# Mục đích: người dùng đưa lên 1 ảnh keyframe LẤY NGUYÊN VẸN từ dataset
# (không chỉnh sửa/nén lại) -> tra cứu lại đúng metadata (video_name,
# frame_id, image_path, pts_time) đã gắn sẵn trong Qdrant.
# Vì ảnh không đổi -> so khớp SHA-256 là chính xác 100%, không cần AI.
def compute_sha256(file_path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_or_load_hash_index() -> dict:
    """
    Build (lần đầu) hoặc cập nhật gia tăng (incremental) hash index.
    - Lần đầu (chưa có cache): quét TOÀN BỘ collection, hash hết.
    - Các lần sau (đã có cache, có thêm batch mới trong Qdrant): CHỈ hash
      những image_path CHƯA từng xuất hiện trong cache cũ — không hash lại
      những ảnh batch cũ đã xử lý, tiết kiệm thời gian đáng kể.
    """
    index = {}
    processed_paths = set()

    if os.path.exists(HASH_INDEX_CACHE_PATH):
        print("⏳ Đang tải hash index đã build sẵn từ cache...")
        try:
            with open(HASH_INDEX_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            index = cached.get("index", {})
            processed_paths = set(cached.get("processed_paths", []))
            print(f"✅ Đã tải {len(index)} hash từ cache "
                  f"({len(processed_paths)} ảnh đã xử lý trước đó).")
        except Exception as e:
            print(f"⚠️  Cache hash index bị lỗi, sẽ build lại từ đầu: {e}")
            index, processed_paths = {}, set()

    print("⏳ Đang quét collection "
          f"'{IMAGE_COLLECTION_NAME}' để tìm ảnh MỚI cần hash (bỏ qua ảnh đã xử lý)...")
    _t0 = time.time()
    next_offset = None
    scanned, newly_hashed, skipped, missing = 0, 0, 0, 0

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=IMAGE_COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        for p in points:
            payload = p.payload or {}
            image_path = payload.get("image_path")
            if not image_path:
                continue
            scanned += 1

            # Bỏ qua ảnh đã hash từ lần chạy trước (đây là chỗ tiết kiệm
            # thời gian khi thêm batch2, batch3... về sau)
            if image_path in processed_paths:
                skipped += 1
                continue

            abs_path = image_path if os.path.isabs(image_path) \
                else os.path.join(BASE_IMAGE_DIR, image_path.replace("/", os.sep))

            if not os.path.exists(abs_path):
                missing += 1
                continue

            try:
                file_hash = compute_sha256(abs_path)
                index[file_hash] = {
                    "video_name": payload.get("video_name"),
                    "frame_id": payload.get("frame_id"),
                    "image_path": image_path,
                    "pts_time": payload.get("pts_time", 0.0)
                }
                processed_paths.add(image_path)
                newly_hashed += 1
            except Exception:
                continue

        if next_offset is None:
            break

    print(f"✅ Cập nhật hash index xong trong {time.time() - _t0:.1f}s: "
          f"{newly_hashed} ảnh MỚI vừa hash, {skipped} ảnh cũ được bỏ qua "
          f"(không hash lại), {missing} file không tìm thấy trên ổ cứng. "
          f"Tổng hiện có: {len(index)} hash.")

    try:
        with open(HASH_INDEX_CACHE_PATH, "wb") as f:
            pickle.dump({"index": index, "processed_paths": processed_paths}, f)
        print(f"💾 Đã lưu cache hash index vào '{HASH_INDEX_CACHE_PATH}'.")
    except Exception as e:
        print(f"⚠️  Không lưu được cache hash index: {e}")

    return index


HASH_INDEX = build_or_load_hash_index()


@app.post("/api/admin/reload-hash-index")
def reload_hash_index():
    """
    [NHIỆM VỤ]: Gọi API này SAU KHI upsert xong 1 batch mới (batch2, batch3...)
    vào Qdrant, để cập nhật hash index mà KHÔNG cần restart cả server
    (tránh phải load lại DFN5B/TraKE/EmbeddingGemma vốn tốn thời gian).
    Chỉ hash các ảnh MỚI, không hash lại ảnh batch cũ (xem build_or_load_hash_index).
    """
    global HASH_INDEX
    try:
        HASH_INDEX = build_or_load_hash_index()
        return {"status": "success", "total_hashes": len(HASH_INDEX)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH SEMANTIC (OpenCLIP DFN5B - CPU)
# ==========================================
print("⏳ Đang tải mô hình OpenCLIP (DFN5B) lên CPU...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b', device=device)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng trên CPU!")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH ASR (Embedding Gemma - CPU)
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
# 3. TraKE (Sequential Action Search) — NẠP TOÀN BỘ EMBEDDING VÀO RAM
# ==========================================
# ⚡ TỐI ƯU TỐC ĐỘ TỐI ĐA: thay vì gọi Qdrant qua mạng mỗi lần search (dù
# là ANN hay brute-force), toàn bộ embedding của collection "dfn5b_images"
# (không đổi giữa các lần search) được tải 1 LẦN DUY NHẤT vào RAM dạng ma
# trận numpy ngay lúc khởi động. Mỗi lần search TraKE sau đó chỉ còn là
# 1 phép nhân ma trận numpy (BLAS, rất nhanh) — hoàn toàn không gọi Qdrant
# qua mạng nữa. Dùng lại clip_model/clip_tokenizer (DFN5B) đã tải ở bước 1,
# không tải thêm model thứ 2 cho cùng 1 bộ trọng số.
#
# ⚠️ LƯU Ý: nếu bạn re-upload/cập nhật dữ liệu trong Qdrant sau khi server
# đã khởi động, RAM cache này KHÔNG tự cập nhật — phải restart server.
print("⏳ Đang tải toàn bộ embedding vào RAM cho TraKE (chỉ chạy 1 lần lúc khởi động)...")
_load_start = time.time()

try:
    qdrant_client.create_payload_index(
        collection_name=IMAGE_COLLECTION_NAME,
        field_name="video_name",
        field_schema=qmodels.PayloadSchemaType.KEYWORD
    )
except Exception:
    pass  # index có thể đã tồn tại, bỏ qua an toàn

ALL_VECTORS = []
ALL_VIDEO_NAMES = []
ALL_FRAME_IDS = []
ALL_PTS_TIMES = []
ALL_IMAGE_PATHS = []

_next_offset = None
_loaded_count = 0
while True:
    points, _next_offset = qdrant_client.scroll(
        collection_name=IMAGE_COLLECTION_NAME,
        limit=5000,               # batch lớn để giảm số lượt gọi mạng lúc load
        offset=_next_offset,
        with_payload=True,
        with_vectors=True
    )
    for p in points:
        ALL_VECTORS.append(p.vector)
        ALL_VIDEO_NAMES.append(p.payload.get("video_name", "unknown"))
        ALL_FRAME_IDS.append(p.payload.get("frame_id", 0))
        ALL_PTS_TIMES.append(p.payload.get("pts_time", 0.0))
        ALL_IMAGE_PATHS.append(p.payload.get("image_path", ""))

    _loaded_count += len(points)
    if _loaded_count % 20000 < 5000:  # in tiến độ định kỳ, không spam log
        print(f"   ... đã tải {_loaded_count} điểm")

    if _next_offset is None:
        break

if len(ALL_VECTORS) == 0:
    print(f"❌ CẢNH BÁO: Không tải được điểm nào từ collection '{IMAGE_COLLECTION_NAME}'! "
          f"Kiểm tra lại tên collection có đúng và đã upload dữ liệu chưa.")

EMBEDDING_MATRIX = np.array(ALL_VECTORS, dtype=np.float32)  # shape: (N, 1024)
FRAME_IDS_ARR = np.array(ALL_FRAME_IDS, dtype=np.int64)
PTS_TIMES_ARR = np.array(ALL_PTS_TIMES, dtype=np.float64)

# Gom sẵn chỉ số (index) theo từng video -> khi search chỉ cần "cắt lát"
# (slice) mảng numpy theo các index này, không cần lọc/tìm kiếm gì thêm.
VIDEO_TO_INDICES = {}
for i, v_name in enumerate(ALL_VIDEO_NAMES):
    VIDEO_TO_INDICES.setdefault(v_name, []).append(i)
for v_name in VIDEO_TO_INDICES:
    VIDEO_TO_INDICES[v_name] = np.array(VIDEO_TO_INDICES[v_name], dtype=np.int64)

_ram_mb = EMBEDDING_MATRIX.nbytes / (1024 ** 2) if len(ALL_VECTORS) > 0 else 0.0
print(f"✅ Đã nạp {len(ALL_VECTORS)} điểm ({len(VIDEO_TO_INDICES)} video) vào RAM cho TraKE "
      f"({_ram_mb:.1f} MB) trong {time.time() - _load_start:.1f} giây.")

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


# ==========================================
# 3. CÁC API ENDPOINTS & NHIỆM VỤ CHI TIẾT
# ==========================================

@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text cho Image"), top_k: int = 50):
    """
    [NHIỆM VỤ]: Semantic Text-to-Image Search — bản KHÔNG gọi Qdrant lúc search.
    - Nhận câu truy vấn văn bản, encode thẳng bằng clip_model (không qua Gemini
      dịch/tối ưu nữa — dùng nguyên câu gốc).
    - Tìm kiếm bằng phép nhân ma trận numpy trên EMBEDDING_MATRIX đã nạp sẵn
      trong RAM lúc khởi động (dùng chung với TraKE) — KHÔNG round-trip mạng
      tới Qdrant, nên nhanh hơn nhiều so với qdrant_client.query_points.
    """
    if not prompt.strip():
        return {"results": []}
    try:
        text_tokens = clip_tokenizer([prompt]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_vec = query_features.float().cpu().numpy().flatten()  # (1024,)

        # 1 phép nhân ma trận duy nhất trên TOÀN BỘ dữ liệu đã cache trong RAM
        sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T  # (N,)

        k = min(top_k, sim.shape[0])
        top_idx_unsorted = np.argpartition(-sim, k - 1)[:k]
        top_idx = top_idx_unsorted[np.argsort(-sim[top_idx_unsorted])]

        output = []
        for i in top_idx:
            output.append({
                "image_path": ALL_IMAGE_PATHS[i],
                "score": round(float(sim[i]), 4),
                "video_name": ALL_VIDEO_NAMES[i],
                "frame_id": int(FRAME_IDS_ARR[i]),
                "pts_time": float(PTS_TIMES_ARR[i])
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


@app.post("/api/lookup/image")
async def lookup_image_exact(file: UploadFile = File(...), top_k: int = 50):
    """
    [NHIỆM VỤ]: Reverse Image Lookup (thay thế tính năng OCR).
    - Dùng khi người dùng có sẵn 1 ảnh keyframe LẤY NGUYÊN VẸN từ dataset
      (chưa qua chỉnh sửa/nén lại), cần tra lại metadata gốc (video_name,
      frame_id, image_path, pts_time) đã gắn sẵn trong Qdrant.
    - Tầng 1 (chính xác 100%): so khớp SHA-256 của file upload với HASH_INDEX
      đã build sẵn từ toàn bộ ảnh trong dataset — vì ảnh không đổi nên khớp
      hash là chắc chắn đúng, không cần AI, gần như tức thì.
    - Tầng 2 (dự phòng, "match_type": "approximate"): nếu không tìm thấy hash
      khớp (VD ảnh đã bị nén lại/đổi định dạng ngoài ý muốn), fallback dùng
      CLIP encode_image + Qdrant search để tìm ảnh gần giống nhất.
    """
    try:
        image_bytes = await file.read()
        file_hash = hashlib.sha256(image_bytes).hexdigest()

        # --- Tầng 1: exact match bằng hash ---
        if file_hash in HASH_INDEX:
            meta = HASH_INDEX[file_hash]
            return {
                "match_type": "exact",
                "results": [{
                    "video_name": meta.get("video_name"),
                    "frame_id": meta.get("frame_id"),
                    "image_path": meta.get("image_path"),
                    "pts_time": meta.get("pts_time", 0.0),
                    "score": 1.0
                }]
            }

        # --- Tầng 2: fallback CLIP approximate search ---
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
        return {"match_type": "approximate", "results": output}
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
    [NHIỆM VỤ]: TraKE (Sequential Action Search) — bản KHÔNG gọi Qdrant lúc search.
    - Encode từng sự kiện (trek1..trek5) bằng clip_model (DFN5B) — model đã
      load sẵn trong RAM cho Semantic Search, không tải thêm model thứ 2.
    - Toàn bộ embedding của "dfn5b_images" đã nạp sẵn vào RAM (EMBEDDING_MATRIX)
      lúc khởi động -> search chỉ còn 1 phép nhân ma trận numpy duy nhất
      (num_events, 1024) @ (1024, N) = (num_events, N), KHÔNG round-trip
      mạng tới Qdrant nữa.
    - Gom theo video bằng VIDEO_TO_INDICES đã tính sẵn (chỉ "cắt lát" mảng).
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
        text_tokens = clip_tokenizer(active_queries).to(device)

        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            text_vecs = text_feats.float().cpu().numpy()  # (num_events, 1024)

        # 1 phép nhân ma trận duy nhất trên TOÀN BỘ dữ liệu đã cache trong RAM
        sim_all = text_vecs.astype(np.float32) @ EMBEDDING_MATRIX.T  # (num_events, N)

        video_candidates = {}
        for v_name, idxs in VIDEO_TO_INDICES.items():
            video_candidates[v_name] = {}
            v_frame_ids = FRAME_IDS_ARR[idxs]
            v_pts_times = PTS_TIMES_ARR[idxs]
            v_image_paths = [ALL_IMAGE_PATHS[i] for i in idxs]

            for e_idx in range(num_events):
                v_scores = sim_all[e_idx][idxs]
                items = [
                    (float(v_pts_times[i]), float(v_scores[i]), int(v_frame_ids[i]), v_image_paths[i])
                    for i in range(len(idxs))
                ]
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
              f"({num_events} sự kiện, {len(VIDEO_TO_INDICES)} video, không gọi Qdrant).")

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
