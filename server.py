import os
import csv
import time
import bisect
import hashlib
import pickle
import traceback
import argparse
import torch
import open_clip
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from PIL import Image
import io

# ==========================================
# ĐỌC THAM SỐ DÒNG LỆNH
# ==========================================
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument(
    "--new", action="store_true",
    help="Quét lại Qdrant để cập nhật hash index (dùng khi vừa thêm batch ảnh mới). "
         "Không truyền cờ này -> load thẳng pickle cache cũ cho khởi động nhanh."
)
_cli_args, _ = _arg_parser.parse_known_args()
FORCE_RESCAN_HASH_INDEX = _cli_args.new

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
# HASH INDEX CHO REVERSE IMAGE LOOKUP
# ==========================================
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
    index = {}
    processed_paths = set()

    # Nếu có cờ --new, ta xóa cache cũ đi để quét lại từ đầu
    if FORCE_RESCAN_HASH_INDEX and os.path.exists(HASH_INDEX_CACHE_PATH):
        try:
            os.remove(HASH_INDEX_CACHE_PATH)
            print(f"🗑️ Đã xóa cache cũ tại '{HASH_INDEX_CACHE_PATH}' do có cờ --new.")
        except Exception as e:
            print(f"⚠️ Không thể xóa file cache: {e}")

    if not FORCE_RESCAN_HASH_INDEX and os.path.exists(HASH_INDEX_CACHE_PATH):
        print("⏳ Đang tải hash index đã build sẵn từ cache...")
        try:
            with open(HASH_INDEX_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            index = cached.get("index", {})
            processed_paths = set(cached.get("processed_paths", []))
            print(f"✅ Đã tải {len(index)} hash từ cache "
                  f"({len(processed_paths)} ảnh đã xử lý trước đó).")
            return index
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

            if not FORCE_RESCAN_HASH_INDEX and image_path in processed_paths:
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
          f"{newly_hashed} ảnh MỚI vừa hash, {skipped} ảnh cũ được bỏ qua, "
          f"{missing} file không tìm thấy. Tổng hiện có: {len(index)} hash.")

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
    embedding_model = SentenceTransformer(GEMMA_EMBEDDING_MODEL_NAME, device=device)
    print("✅ Embedding Gemma đã sẵn sàng trên CPU!")
except Exception as e:
    print(f"❌ Lỗi khi tải Embedding Gemma: {e}")

# ==========================================
# 3. TraKE (Sequential Action Search) — NẠP TOÀN BỘ EMBEDDING VÀO RAM
# ==========================================
print("⏳ Đang tải toàn bộ embedding vào RAM cho TraKE (chỉ chạy 1 lần lúc khởi động)...")
_load_start = time.time()

try:
    qdrant_client.create_payload_index(
        collection_name=IMAGE_COLLECTION_NAME,
        field_name="video_name",
        field_schema=qmodels.PayloadSchemaType.KEYWORD
    )
except Exception:
    pass

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
        limit=3000,
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
    if _loaded_count % 20000 < 5000:
        print(f"   ... đã tải {_loaded_count} điểm")

    if _next_offset is None:
        break

if len(ALL_VECTORS) == 0:
    print(f"❌ CẢNH BÁO: Không tải được điểm nào từ collection '{IMAGE_COLLECTION_NAME}'!")

EMBEDDING_MATRIX = np.array(ALL_VECTORS, dtype=np.float32)
FRAME_IDS_ARR = np.array(ALL_FRAME_IDS, dtype=np.int64)
PTS_TIMES_ARR = np.array(ALL_PTS_TIMES, dtype=np.float64)

VIDEO_TO_INDICES = {}
for i, v_name in enumerate(ALL_VIDEO_NAMES):
    VIDEO_TO_INDICES.setdefault(v_name, []).append(i)
for v_name in VIDEO_TO_INDICES:
    VIDEO_TO_INDICES[v_name] = np.array(VIDEO_TO_INDICES[v_name], dtype=np.int64)

_ram_mb = EMBEDDING_MATRIX.nbytes / (1024 ** 2) if len(ALL_VECTORS) > 0 else 0.0
print(f"✅ Đã nạp {len(ALL_VECTORS)} điểm ({len(VIDEO_TO_INDICES)} video) vào RAM cho TraKE "
      f"({_ram_mb:.1f} MB) trong {time.time() - _load_start:.1f} giây.")

# ==========================================
# THUẬT TOÁN DP CHO TraKE
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
# CÁC API ENDPOINTS
# ==========================================

@app.get("/api/search")
def search_semantic(prompt: str = Query(..., description="Query Text cho Image"), top_k: int = 50):
    if not prompt.strip():
        return {"results": []}
    try:
        text_tokens = clip_tokenizer([prompt]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(text_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_vec = query_features.float().cpu().numpy().flatten()

        sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T

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
    if not prompt.strip():
        return {"results": []}
    try:
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


# ==========================================
# ENDPOINT: TÌM ẢNH LÂN CẬN (SEQUENTIAL FRAMES)
# ==========================================
@app.get("/api/search/sequential-frames")
async def search_sequential_frames(
    video_name: str = Query(..., description="Tên video, VD: L26_V105"),
    center_frame_id: str = Query(..., description="Số thứ tự đuôi của frame gốc, VD: 127")
):
    """
    API tìm 5 frame trước và 5 frame sau dựa vào số thứ tự đuôi image_path (nhập thủ công).
    """
    try:
        clean_fid = center_frame_id.replace("#", "").replace("img_", "").split(".")[0].strip()
        center_num = int(clean_fid)

        start_num = max(0, center_num - 5)
        end_num = center_num + 5

        results = []
        for f_id in range(start_num, end_num + 1):
            padded_id = f"{f_id:03d}" if f_id < 1000 else str(f_id)
            image_path = f"{video_name}/{padded_id}.webp"
            
            results.append({
                "video_name": video_name,
                "frame_id": f"#{padded_id}",
                "image_path": image_path,
                "pts_time": f"00:00:{(f_id * 1):02d}.000",
                "score": 100.0 if f_id == center_num else 90.0
            })

        return {"status": "success", "results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/search/sequential-frames-upload")
async def search_sequential_frames_by_upload(file: UploadFile = File(...)):
    """
    API tìm ảnh lân cận bằng cách upload trực tiếp file ảnh:
    Tự băm SHA-256 đối chiếu HASH_INDEX để lấy ra frame gốc rồi trả về dải 11 frame lân cận.
    """
    try:
        image_bytes = await file.read()
        file_hash = hashlib.sha256(image_bytes).hexdigest()

        if file_hash not in HASH_INDEX:
            return JSONResponse(
                status_code=404, 
                content={"status": "error", "message": "Không tìm thấy ảnh này trong dataset gốc qua mã băm."}
            )

        meta = HASH_INDEX[file_hash]
        video_name = meta.get("video_name")
        raw_frame_id = str(meta.get("frame_id", ""))
        
        clean_fid = raw_frame_id.replace("#", "").replace("img_", "").split(".")[0].strip()
        center_num = int(clean_fid)

        start_num = max(0, center_num - 5)
        end_num = center_num + 5

        results = []
        for f_id in range(start_num, end_num + 1):
            padded_id = f"{f_id:03d}" if f_id < 1000 else str(f_id)
            image_path = f"{video_name}/{padded_id}.webp"
            
            results.append({
                "video_name": video_name,
                "frame_id": f"#{padded_id}",
                "image_path": image_path,
                "pts_time": f"00:00:{(f_id * 1):02d}.000",
                "score": 100 if f_id == center_num else 90
            })

        return {"status": "success", "results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/lookup/image")
async def lookup_image_exact(file: UploadFile = File(...), top_k: int = 50):
    try:
        image_bytes = await file.read()
        file_hash = hashlib.sha256(image_bytes).hexdigest()

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
            text_vecs = text_feats.float().cpu().numpy()

        sim_all = text_vecs.astype(np.float32) @ EMBEDDING_MATRIX.T

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

        print(f"⏱️  TraKE search hoàn tất trong {time.time() - _t0:.2f}s.")
        return {"results": output}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
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
    if not os.path.isabs(path):
        win_path = os.path.join(BASE_IMAGE_DIR, path)
    else:
        win_path = path.replace("/", "\\")
        
    if os.path.exists(win_path):
        return FileResponse(win_path)
    return {"error": f"File not found at {win_path}"}


@app.get("/api/video")
def get_local_video(video_name: str):
    filename = f"{video_name}.mp4" if not video_name.endswith(".mp4") else video_name
    video_path = os.path.join(VIDEO_DIR, filename)
    
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": f"Video not found at {video_path}"}


@app.post("/api/submit-csv")
def submit_to_csv(
    mode: str = Query("semantic", description="Chế độ hiện tại: semantic, vqa, trake"),
    video_name: str = Query(..., description="Tên video"),
    frame_id: str = Query(..., description="Frame ID chính"),
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
