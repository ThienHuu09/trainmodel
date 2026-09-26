import argparse
import concurrent.futures
import csv
import functools
import hashlib
import io
import os
import pickle
import random as py_random
import re
import time
import bisect
from typing import Optional
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import open_clip
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import requests
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

# CẢI TIẾN: Sử dụng package chuẩn mới `google.genai` từ Google AI Studio
# Lấy API key tại: https://aistudio.google.com/apikey
try:
  from google import genai
  from google.genai import types as genai_types
except ImportError:
  genai = None
  genai_types = None

# Tự động đọc file .env đặt cùng thư mục (nếu có)
try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC774oM9uqXHXaVTFYjeLA-TDnTKWMJjzA")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TRANSLATE_TEMPERATURE = float(
    os.environ.get("GEMINI_TRANSLATE_TEMPERATURE", "0.2")
)

gemini_client = None
if genai is not None and GEMINI_API_KEY:
  try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
  except Exception as e:
    print(f"⚠️ Không thể khởi tạo Google GenAI Client: {e}")

# 🔍 DEBUG — thêm dòng này
print(f"[DEBUG] Đang dùng model = '{GEMINI_MODEL_NAME}'")
print(f"[DEBUG] Đang dùng key = '{GEMINI_API_KEY[:15]}...' (độ dài {len(GEMINI_API_KEY)})")
print(f"[DEBUG] gemini_client đã khởi tạo: {gemini_client is not None}")
# Giới hạn token áp dụng cho CLIP (context_length mặc định = 77 token).
MAX_QUERY_TOKENS = 70

# ==========================================
# CẤU HÌNH THIẾT BỊ VÀ THREADS (Full CPU Optimization)
# ==========================================
device = "cpu"
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

print(
    f"[INFO] Hệ thống đang chạy hoàn toàn trên CPU. Số threads được cấu hình:"
    f" {num_cores}"
)

# ==========================================
# ĐỌC THAM SỐ DÒNG LỆNH
# ==========================================
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument(
    "--new",
    action="store_true",
    help="Bỏ qua cache pickle cũ, hash lại TOÀN BỘ ảnh từ đầu.",
)
_cli_args, _ = _arg_parser.parse_known_args()
FORCE_RESCAN_HASH_INDEX = _cli_args.new

IMAGE_COLLECTION_NAME = "dfn5b_images"
ASR_COLLECTION_NAME = "embeddinggemma_audio"
OCR_COLLECTION_NAME = "ocr_collection"

HASH_INDEX_CACHE_PATH = "hash_index.pkl"

# ==========================================
# ĐỒNG BỘ NHIỀU Ổ ĐĨA / BATCH DATASET (batch1 + batch2)
# ==========================================
# Do giới hạn dung lượng, dataset được tách ra nhiều thư mục/ổ đĩa. Thử lần
# lượt từng thư mục trong danh sách cho tới khi tìm thấy file — không cần
# gộp file vật lý, không sửa lại đường dẫn đã lưu trong Qdrant.
BASE_IMAGE_DIRS = [
    r"C:\AIC2026\dataset_webp",        # batch 1 (cũ)
    r"E:\dataset batch 2",     # batch 2 (mới)
]
# Các đuôi video hỗ trợ hiển thị — thử theo đúng thứ tự này. .mov (QuickTime)
# thêm vào theo yêu cầu, media_type tương ứng khai báo ở VIDEO_MIME_TYPES.
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".webm", ".avi"]
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}

VIDEO_DIRS = [
    r"C:\AIC2026\video",           # batch 1 (cũ)
    r"F:\videos",               # batch 2 (mới)
]

# Giữ lại 2 biến đơn (BASE_IMAGE_DIR/VIDEO_DIR) để tương thích ngược, luôn
# trỏ về thư mục batch1 — chỉ dùng làm fallback cuối khi resolve_*() không
# tìm thấy file ở đâu cả (để log lỗi có đường dẫn dễ hiểu).
# Số luồng song song khi hash ảnh lúc khởi động (I/O-bound nên để cao hơn số
# core CPU vẫn lợi). Máy ổ SSD/NVMe có thể tăng lên 64; ổ HDD nên giữ thấp
# hơn (~8-16) để tránh thrashing đầu đọc.
HASH_SCAN_WORKERS = 32

BASE_IMAGE_DIR = BASE_IMAGE_DIRS[0]
VIDEO_DIR = VIDEO_DIRS[0]


def resolve_image_abs_path(image_path: str) -> str:
  """Trả về đường dẫn tuyệt đối tồn tại đầu tiên khi thử qua từng thư mục
  trong BASE_IMAGE_DIRS (batch1 rồi batch2, ...). Nếu image_path đã là
  đường dẫn tuyệt đối thì dùng luôn. Nếu không thư mục nào có file, trả về
  path ghép với thư mục đầu tiên để log lỗi vẫn có ý nghĩa."""
  if os.path.isabs(image_path):
    return image_path
  rel = image_path.replace("/", os.sep)
  for base in BASE_IMAGE_DIRS:
    candidate = os.path.join(base, rel)
    if os.path.exists(candidate):
      return candidate
  return os.path.join(BASE_IMAGE_DIRS[0], rel)


def resolve_video_abs_path(filename: str) -> str:
  """Trả về đường dẫn video tồn tại đầu tiên khi thử qua từng thư mục trong
  VIDEO_DIRS (batch1 rồi batch2, ...), và thử lần lượt các đuôi file phổ
  biến (mp4, mov, mkv, webm, avi) nếu tên chưa có đuôi hoặc đuôi không
  đúng thực tế trên đĩa."""
  base = filename
  for ext in VIDEO_EXTENSIONS:
    if base.lower().endswith(ext):
      base = base[: -len(ext)]
      break
  for base_dir in VIDEO_DIRS:
    for ext in VIDEO_EXTENSIONS:
      candidate = os.path.join(base_dir, base + ext)
      if os.path.exists(candidate):
        return candidate
  return os.path.join(VIDEO_DIRS[0], base + VIDEO_EXTENSIONS[0])

app = FastAPI(
    title=(
        "MFusion-VR Full Core API (servernew.py — Rocchio + Pre-filter + TraKE Context + Vector Cache)"
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Đang kết nối Qdrant Server...")
qdrant_client = QdrantClient(host="localhost", port=6333)
print("✅ Đã kết nối Qdrant thành công!")

try:
  qdrant_client.create_payload_index(
      collection_name=IMAGE_COLLECTION_NAME,
      field_name="video_name",
      field_schema=qmodels.PayloadSchemaType.KEYWORD,
  )
except Exception:
  pass


def compute_sha256(file_path: str, chunk_size: int = 65536) -> str:
  h = hashlib.sha256()
  with open(file_path, "rb") as f:
    while True:
      chunk = f.read(chunk_size)
      if not chunk:
        break
      h.update(chunk)
  return h.hexdigest()


# ==========================================
# ⚡ NẠP RAM CACHE (SEMANTIC IMAGE MATRIX)
# ==========================================
def build_all_ram_caches(force_rescan_hash: bool):
  _load_start = time.time()

  old_hash_index = {}
  processed_paths = set()

  if force_rescan_hash and os.path.exists(HASH_INDEX_CACHE_PATH):
    try:
      os.remove(HASH_INDEX_CACHE_PATH)
      print(
          f"🗑️ Đã xóa cache hash cũ tại '{HASH_INDEX_CACHE_PATH}' do có cờ"
          " --new."
      )
    except Exception as e:
      print(f"⚠️ Không thể xóa file cache: {e}")

  if not force_rescan_hash and os.path.exists(HASH_INDEX_CACHE_PATH):
    print("⏳ Đang tải hash index đã build sẵn từ cache...")
    try:
      with open(HASH_INDEX_CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
      old_hash_index = cached.get("index", {})
      processed_paths = set(cached.get("processed_paths", []))
      print(
          f"✅ Đã tải {len(old_hash_index)} hash từ cache"
          f" ({len(processed_paths)} ảnh đã xử lý trước đó)."
      )
    except Exception as e:
      print(f"⚠️  Cache hash index bị lỗi, sẽ build lại từ đầu: {e}")
      old_hash_index, processed_paths = {}, set()

  hash_index = dict(old_hash_index)

  print(
      f"⏳ Đang quét collection '{IMAGE_COLLECTION_NAME}' — 1 LẦN DUY NHẤT —"
      " vừa nạp vector cho Semantic/TraKE, vừa cập nhật hash index (chỉ ảnh"
      " mới)..."
  )

  all_vectors = []
  frame_ids_tmp = []
  pts_times_tmp = []
  all_image_paths = []
  video_name_idx_tmp = []
  video_name_to_idx = {}
  video_name_unique = []
  video_to_indices_tmp = {}
  image_path_to_idx = {}

  newly_hashed, skipped_hash, missing_hash = 0, 0, 0
  next_offset = None
  loaded_count = 0
  to_hash = []  # (image_path, video_name, frame_id, pts_time) cần hash — xử lý song song sau vòng lặp

  while True:
    points, next_offset = qdrant_client.scroll(
        collection_name=IMAGE_COLLECTION_NAME,
        limit=4000,
        offset=next_offset,
        with_payload=True,
        with_vectors=True,
    )
    for p in points:
      payload = p.payload or {}
      v_name = payload.get("video_name", "unknown")
      frame_id = payload.get("frame_id", 0)
      pts_time = payload.get("pts_time", 0.0)
      image_path = payload.get("image_path", "").replace("\\", "/")

      point_idx = len(all_vectors)
      all_vectors.append(p.vector)
      frame_ids_tmp.append(frame_id)
      pts_times_tmp.append(pts_time)
      all_image_paths.append(image_path)

      if image_path:
        image_path_to_idx[image_path] = point_idx

      if v_name not in video_name_to_idx:
        video_name_to_idx[v_name] = len(video_name_unique)
        video_name_unique.append(v_name)
      video_name_idx_tmp.append(video_name_to_idx[v_name])
      video_to_indices_tmp.setdefault(v_name, []).append(point_idx)

      if image_path:
        if not force_rescan_hash and image_path in processed_paths:
          skipped_hash += 1
        else:
          # Không hash ngay tại đây (I/O tuần tự rất chậm) — gom lại rồi
          # chạy song song đa luồng bên dưới, sau khi tải hết điểm.
          to_hash.append((image_path, v_name, frame_id, pts_time))

    loaded_count += len(points)
    if loaded_count % 20000 < 3000:
      print(f"   ... đã tải {loaded_count} điểm")

    if next_offset is None:
      break

  # --- Hash song song đa luồng (I/O-bound -> ThreadPoolExecutor rất hiệu quả) ---
  if to_hash:
    print(
        f"⏳ Đang hash song song {len(to_hash)} ảnh mới bằng"
        f" {HASH_SCAN_WORKERS} luồng..."
    )
    _hash_t0 = time.time()

    def _hash_one(item):
      image_path, v_name, frame_id, pts_time = item
      abs_path = resolve_image_abs_path(image_path)
      if not os.path.exists(abs_path):
        return (image_path, None)
      try:
        file_hash = compute_sha256(abs_path)
        return (
            image_path,
            {
                "hash": file_hash,
                "video_name": v_name,
                "frame_id": frame_id,
                "image_path": image_path,
                "pts_time": pts_time,
            },
        )
      except Exception:
        return (image_path, None)

    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=HASH_SCAN_WORKERS
    ) as executor:
      for image_path, result in executor.map(_hash_one, to_hash):
        done_count += 1
        if result is None:
          missing_hash += 1
        else:
          file_hash = result.pop("hash")
          hash_index[file_hash] = result
          processed_paths.add(image_path)
          newly_hashed += 1
        if done_count % 5000 == 0:
          print(f"   ... đã hash {done_count}/{len(to_hash)} ảnh")

    print(
        f"✅ Hash song song xong {len(to_hash)} ảnh trong"
        f" {time.time() - _hash_t0:.1f} giây."
    )

  if len(all_vectors) == 0:
    print(
        f"❌ CẢNH BÁO: Không tải được điểm nào từ collection"
        f" '{IMAGE_COLLECTION_NAME}'!"
    )

  embedding_matrix = np.array(all_vectors, dtype=np.float32)
  del all_vectors

  if embedding_matrix.size > 0:
    print(
        "⚡ Đang chuẩn hóa L2 cho ma trận vector để tính chính xác Cosine"
        " Similarity..."
    )
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    embedding_matrix = embedding_matrix / norms
    print("✅ Chuẩn hóa L2 hoàn tất!")

  frame_ids_arr = np.array(frame_ids_tmp, dtype=np.int32)
  del frame_ids_tmp

  pts_times_arr = np.array(pts_times_tmp, dtype=np.float32)
  del pts_times_tmp

  video_name_idx_arr = np.array(video_name_idx_tmp, dtype=np.int32)
  del video_name_idx_tmp
  del video_name_to_idx

  video_to_indices = {
      name: np.array(idxs, dtype=np.int32)
      for name, idxs in video_to_indices_tmp.items()
  }
  del video_to_indices_tmp

  try:
    with open(HASH_INDEX_CACHE_PATH, "wb") as f:
      pickle.dump(
          {"index": hash_index, "processed_paths": processed_paths}, f
      )
    print(f"💾 Đã lưu cache hash index vào '{HASH_INDEX_CACHE_PATH}'.")
  except Exception as e:
    print(f"⚠️  Không lưu được cache hash index: {e}")

  ram_mb = (
      (embedding_matrix.nbytes / (1024 ** 2))
      if embedding_matrix.size > 0
      else 0.0
  )
  print(
      f"✅ Đã nạp {len(all_image_paths)} điểm ({len(video_to_indices)} video) vào"
      f" RAM ({ram_mb:.1f} MB embedding) trong {time.time() - _load_start:.1f}"
      f" giây. Hash: {newly_hashed} mới, {skipped_hash} bỏ qua (đã có cache),"
      f" {missing_hash} thiếu file."
  )

  return {
      "embedding_matrix": embedding_matrix,
      "frame_ids_arr": frame_ids_arr,
      "pts_times_arr": pts_times_arr,
      "all_image_paths": all_image_paths,
      "video_name_unique": video_name_unique,
      "video_name_idx_arr": video_name_idx_arr,
      "video_to_indices": video_to_indices,
      "hash_index": hash_index,
      "image_path_to_idx": image_path_to_idx,
  }


_caches = build_all_ram_caches(force_rescan_hash=FORCE_RESCAN_HASH_INDEX)
EMBEDDING_MATRIX = _caches["embedding_matrix"]
FRAME_IDS_ARR = _caches["frame_ids_arr"]
PTS_TIMES_ARR = _caches["pts_times_arr"]
ALL_IMAGE_PATHS = _caches["all_image_paths"]
VIDEO_NAME_UNIQUE = _caches["video_name_unique"]
VIDEO_NAME_IDX_ARR = _caches["video_name_idx_arr"]
VIDEO_TO_INDICES = _caches["video_to_indices"]
HASH_INDEX = _caches["hash_index"]
IMAGE_PATH_TO_IDX = _caches["image_path_to_idx"]
del _caches


# ==========================================
# ⚡ NẠP RAM CACHE CHO ASR (ASR MATRIX)
# ==========================================
def build_asr_ram_cache():
  _t0 = time.time()
  print(f"⏳ Đang nạp collection '{ASR_COLLECTION_NAME}' (ASR) vào RAM...")

  all_vectors = []
  texts_tmp = []
  video_names_tmp = []
  image_paths_tmp = []
  audio_paths_tmp = []
  pts_times_tmp = []
  frame_ids_tmp = []
  asr_image_path_to_idx = {}

  next_offset = None
  loaded_count = 0
  while True:
    points, next_offset = qdrant_client.scroll(
        collection_name=ASR_COLLECTION_NAME,
        limit=5000,
        offset=next_offset,
        with_payload=True,
        with_vectors=True,
    )
    for p in points:
      payload = p.payload or {}
      idx = len(all_vectors)
      all_vectors.append(p.vector)
      texts_tmp.append(payload.get("text"))
      video_names_tmp.append(payload.get("video_name"))
      ipath = (payload.get("image_path") or "").replace("\\", "/")
      image_paths_tmp.append(ipath)
      if ipath:
        asr_image_path_to_idx[ipath] = idx
      audio_paths_tmp.append(payload.get("audio_path"))
      pts_times_tmp.append(payload.get("pts_time", 0.0))
      frame_ids_tmp.append(payload.get("frame_id", 0))

    loaded_count += len(points)
    if loaded_count % 20000 < 2300:
      print(f"   ... đã tải {loaded_count} đoạn ASR")

    if next_offset is None:
      break

  if len(all_vectors) == 0:
    print(
        f"⚠️  CẢNH BÁO: Không tải được điểm nào từ collection"
        f" '{ASR_COLLECTION_NAME}'! ASR search sẽ luôn trả về rỗng."
    )
    matrix = np.zeros((0, 0), dtype=np.float32)
  else:
    matrix = np.array(all_vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    matrix = matrix / norms

  ram_mb = (matrix.nbytes / (1024 ** 2)) if matrix.size > 0 else 0.0
  print(
      f"✅ Đã nạp {len(texts_tmp)} đoạn ASR vào RAM ({ram_mb:.1f} MB) trong"
      f" {time.time() - _t0:.1f} giây."
  )

  return {
      "matrix": matrix,
      "texts": texts_tmp,
      "video_names": video_names_tmp,
      "image_paths": image_paths_tmp,
      "audio_paths": audio_paths_tmp,
      "pts_times": pts_times_tmp,
      "frame_ids": frame_ids_tmp,
      "image_path_to_idx": asr_image_path_to_idx,
  }


_asr_caches = build_asr_ram_cache()
ASR_EMBEDDING_MATRIX = _asr_caches["matrix"]
ASR_TEXTS = _asr_caches["texts"]
ASR_VIDEO_NAMES = _asr_caches["video_names"]
ASR_IMAGE_PATHS = _asr_caches["image_paths"]
ASR_AUDIO_PATHS = _asr_caches["audio_paths"]
ASR_PTS_TIMES = _asr_caches["pts_times"]
ASR_FRAME_IDS = _asr_caches["frame_ids"]
ASR_IMAGE_PATH_TO_IDX = _asr_caches["image_path_to_idx"]
del _asr_caches


def get_video_name(i: int) -> str:
  return VIDEO_NAME_UNIQUE[VIDEO_NAME_IDX_ARR[i]]


QUERY_VECTOR_CACHE = {}


def parse_video_filter(video_filter: str):
  if not video_filter or not video_filter.strip():
    return None
  prefixes = [p.strip().upper() for p in video_filter.split(",") if p.strip()]
  if not prefixes:
    return None
  collected = []
  for v_name, idxs in VIDEO_TO_INDICES.items():
    v_upper = v_name.upper()
    if any(v_upper.startswith(px) for px in prefixes):
      collected.append(idxs)
  if not collected:
    return np.array([], dtype=np.int32)
  return np.concatenate(collected)


def parse_asr_video_filter(video_filter: str):
  if not video_filter or not video_filter.strip():
    return None
  prefixes = [p.strip().upper() for p in video_filter.split(",") if p.strip()]
  if not prefixes:
    return None
  mask = np.zeros(len(ASR_VIDEO_NAMES), dtype=bool)
  for i, vn in enumerate(ASR_VIDEO_NAMES):
    if vn and any(vn.upper().startswith(px) for px in prefixes):
      mask[i] = True
  return mask


@app.post("/api/admin/reload-index")
def reload_index():
  global EMBEDDING_MATRIX, FRAME_IDS_ARR, PTS_TIMES_ARR, ALL_IMAGE_PATHS
  global VIDEO_NAME_UNIQUE, VIDEO_NAME_IDX_ARR, VIDEO_TO_INDICES, HASH_INDEX
  global IMAGE_PATH_TO_IDX, QUERY_VECTOR_CACHE
  try:
    caches = build_all_ram_caches(force_rescan_hash=False)
    EMBEDDING_MATRIX = caches["embedding_matrix"]
    FRAME_IDS_ARR = caches["frame_ids_arr"]
    PTS_TIMES_ARR = caches["pts_times_arr"]
    ALL_IMAGE_PATHS = caches["all_image_paths"]
    VIDEO_NAME_UNIQUE = caches["video_name_unique"]
    VIDEO_NAME_IDX_ARR = caches["video_name_idx_arr"]
    VIDEO_TO_INDICES = caches["video_to_indices"]
    HASH_INDEX = caches["hash_index"]
    IMAGE_PATH_TO_IDX = caches["image_path_to_idx"]
    QUERY_VECTOR_CACHE.clear()
    return {
        "status": "success",
        "total_points": len(ALL_IMAGE_PATHS),
        "total_videos": len(VIDEO_TO_INDICES),
        "total_hashes": len(HASH_INDEX),
    }
  except Exception as e:
    return {"status": "error", "message": str(e)}


print("⏳ Đang tải mô hình OpenCLIP (DFN5B) lên CPU...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14-quickgelu", pretrained="dfn5b", device=device
)
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
clip_model.eval()
print("✅ OpenCLIP đã sẵn sàng trên CPU!")


def count_clip_tokens(text: str) -> int:
  if not text:
    return 0
  tokens = clip_tokenizer([text])[0]
  return int((tokens != 0).sum().item())


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
  words = text.split()
  while words:
    candidate = " ".join(words)
    if count_clip_tokens(candidate) <= max_tokens:
      return candidate
    words.pop()
  return text[:200]


def gemini_translate_and_compress(vietnamese_text: str, max_tokens: int) -> str:
  if gemini_client is None:
    raise RuntimeError(
        "Gemini client chưa được khởi tạo. Kiểm tra lại GEMINI_API_KEY."
    )

  prompt = (
      "Translate the following Vietnamese text into English, preserving its"
            " full meaning as closely as possible. The result will be tokenized by a"
            f" CLIP text encoder with a hard limit of ~77 tokens, so the English"
            f" translation MUST stay under {max_tokens} tokens — rewrite/compress it"
            " if needed while keeping every important detail. Return ONLY the final"
            " English text — no explanation, no quotes, no extra commentary.\n\n"
            f"Vietnamese text: {vietnamese_text}"
  )

  response = gemini_client.models.generate_content(
      model=GEMINI_MODEL_NAME,
      contents=prompt,
      config=genai_types.GenerateContentConfig(
          temperature=GEMINI_TRANSLATE_TEMPERATURE,
      ) if genai_types is not None else None,
  )
  translated = (getattr(response, "text", "") or "").strip().strip('"')
  if not translated:
    raise RuntimeError("Gemini không trả về kết quả dịch (response rỗng).")
  return translated


gemini_translate_and_compress = functools.lru_cache(maxsize=2048)(
    gemini_translate_and_compress
)


class TranslateRequest(BaseModel):
  text: str


@app.post("/api/translate")
def translate_query_vi_to_en(payload: TranslateRequest):
  text = (payload.text or "").strip()
  if not text:
    return {"original": "", "translated": "", "token_count": 0}

  _t0 = time.time()
  _hits_before = gemini_translate_and_compress.cache_info().hits
  try:
    translated = gemini_translate_and_compress(text, MAX_QUERY_TOKENS)
    _cache_hit = gemini_translate_and_compress.cache_info().hits > _hits_before
    if _cache_hit:
      print(f"⚡ [Gemini Cache HIT] '{text[:60]}' -> trả kết quả cũ, không gọi API.")
  except Exception as e:
    print(
        f"[CẢNH BÁO] Lỗi dịch Gemini, giữ nguyên tiếng Việt làm fallback: {e}"
    )
    translated = text

  token_count = count_clip_tokens(translated)
  if token_count > MAX_QUERY_TOKENS:
    translated = truncate_to_token_limit(translated, MAX_QUERY_TOKENS)
    token_count = count_clip_tokens(translated)

  print(f"⏱️  Translate (Gemini) hoàn tất trong {time.time() - _t0:.3f}s.")
  return {
      "original": text,
      "translated": translated,
      "token_count": token_count,
  }


@app.get("/api/admin/translate-cache-stats")
def translate_cache_stats():
  info = gemini_translate_and_compress.cache_info()
  return {
      "hits": info.hits,
      "misses": info.misses,
      "current_size": info.currsize,
      "max_size": info.maxsize,
  }


print("⏳ Đang tải mô hình Embedding Gemma lên CPU...")
GEMMA_EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
try:
  embedding_model = SentenceTransformer(
      GEMMA_EMBEDDING_MODEL_NAME, device=device
  )
  print("✅ Embedding Gemma đã sẵn sàng trên CPU!")
except Exception as e:
  print(f"❌ Lỗi khi tải Embedding Gemma: {e}")
  embedding_model = None

print("🚀 SERVER SẴN SÀNG.")


def find_best_trake_dynamic(
    candidates_by_video,
    num_events,
    top_k=5,
    max_duration_sec=300.0,
    gap_penalty=0.0008,
    max_seq_per_video=2,
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
      # ⚙️ FIX: sắp xếp theo pts_time (thời gian thực trong video), KHÔNG
      # phải theo frame_id. Toàn bộ công thức DP + gap_penalty bên dưới
      # được suy ra trên biến pts (giây), nên ràng buộc "sự kiện trước xảy
      # ra trước sự kiện sau" cũng phải được kiểm tra bằng pts, không phải
      # bằng frame_id. frame_id thường tăng cùng chiều với pts, nhưng
      # không có gì đảm bảo tuyệt đối 1-1 (VD: frame bị đánh số lại, video
      # ghép nhiều đoạn, hoặc lệch do làm tròn) — dùng frame_id làm khóa
      # sắp xếp/bisect có thể khiến DP chọn nhầm một candidate "trước" có
      # frame_id nhỏ hơn nhưng pts_time lại LỚN HƠN (đi ngược thời gian),
      # hoặc bỏ sót một candidate hợp lệ có frame_id lớn hơn nhưng pts_time
      # nhỏ hơn. Đây chính là phần còn thiếu/sai của thuật toán TraKE.
      sorted_items = sorted(dedup.values(), key=lambda x: x[0])
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
      # ⚙️ FIX: dùng pts_time (đã sort tăng dần ở trên) để tìm ranh giới
      # "trước thời điểm hiện tại", thay vì frame_id.
      prev_pts_times = [item[0] for item in prev_list]

      # ⚙️ FIX ĐỘ CHÍNH XÁC: prefix-max phải tính trên
      # (score + gap_penalty * pts), KHÔNG phải trên score đơn thuần.
      #
      # Bản cũ: chọn candidate trước đó có score tích lũy cao nhất (bất kể
      # cách xa thời điểm hiện tại bao nhiêu), rồi mới trừ gap_penalty
      # "hậu kiểm" -> đây là xấp xỉ (heuristic), có thể bỏ lỡ 1 candidate
      # điểm thấp hơn chút nhưng gần thời gian hơn, mà sau khi trừ phạt lại
      # cho tổng điểm cao hơn.
      #
      # Về toán học:
      #   total = prev_score - gap_penalty*(cur_pts - prev_pts) + cur_score
      #         = (prev_score + gap_penalty*prev_pts) - gap_penalty*cur_pts + cur_score
      # -> phần phụ thuộc vào "prev" chỉ còn là (prev_score + gap_penalty*prev_pts),
      # nên prefix-max trên đại lượng NÀY mới cho lựa chọn tối ưu THẬT SỰ,
      # đồng thời vẫn giữ nguyên độ phức tạp O(n log n) (không tốn thêm
      # chi phí tính toán nào so với bản cũ).
      prefix_max_val, prefix_max_idx = [], []
      best_val, best_idx = float("-inf"), -1
      for i, node in enumerate(prev_dp):
        weighted = node["score"] + gap_penalty * prev_list[i][0]
        if weighted > best_val:
          best_val, best_idx = weighted, i
        prefix_max_val.append(best_val)
        prefix_max_idx.append(best_idx)

      cur_list = event_lists[e]
      cur_dp = []
      for pts, score, fid, path in cur_list:
        # ⚙️ FIX: điều kiện hợp lệ đúng về mặt thời gian là
        # prev_pts_time < cur_pts_time (không phải prev_fid < cur_fid).
        pos = bisect.bisect_left(prev_pts_times, pts) - 1
        if pos < 0:
          cur_dp.append(
              {"score": float("-inf"), "prev": -1, "start_time": pts}
          )
          continue

        best_prev_idx = prefix_max_idx[pos]
        best_weighted = prefix_max_val[pos]
        prev_start_time = prev_dp[best_prev_idx]["start_time"]

        total_score = best_weighted - gap_penalty * pts + score

        cur_dp.append({
            "score": total_score,
            "prev": best_prev_idx,
            "start_time": prev_start_time,
        })

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
            "image_path": path,
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
          "sequence": seq,
      })

  all_sequences.sort(key=lambda x: x["total_score"], reverse=True)
  return all_sequences[:top_k]


# ==========================================
# API: SEMANTIC SEARCH (Đã tối ưu ma trận)
# ==========================================
@app.get("/api/search")
def search_semantic(
    prompt: str = Query(..., description="Query Text cho Image"),
    top_k: int = 50,
    video_filter: str = Query("", description="Lọc theo prefix video, VD: L21,L22"),
):
  if not prompt.strip():
    return {"results": []}
  try:
    _t0 = time.time()
    cache_key = (prompt.strip(), "semantic")

    _t_enc = time.time()
    if cache_key in QUERY_VECTOR_CACHE:
      query_vec = QUERY_VECTOR_CACHE[cache_key]
    else:
      text_tokens = clip_tokenizer([prompt]).to(device)
      with torch.no_grad():
        query_features = clip_model.encode_text(text_tokens)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_vec = query_features.float().cpu().numpy().flatten()
      QUERY_VECTOR_CACHE[cache_key] = query_vec
    _enc_ms = (time.time() - _t_enc) * 1000

    filtered_idxs = parse_video_filter(video_filter)

    _t_mat = time.time()
    if filtered_idxs is not None:
      if len(filtered_idxs) == 0:
        return {"results": []}
      sub_matrix = EMBEDDING_MATRIX[filtered_idxs]
      sim_sub = query_vec.astype(np.float32) @ sub_matrix.T
      k = min(top_k, sim_sub.shape[0])
      top_sub_unsorted = np.argpartition(-sim_sub, k - 1)[:k]
      top_sub = top_sub_unsorted[np.argsort(-sim_sub[top_sub_unsorted])]
      top_idx = filtered_idxs[top_sub]
      top_sim_vals = sim_sub[top_sub]
    else:
      # ⚡ Sử dụng cú pháp rút gọn, không ép kiểu .astype(np.float32) lại trên ma trận gốc
      sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T
      k = min(top_k, sim.shape[0])
      top_idx_unsorted = np.argpartition(-sim, k - 1)[:k]
      top_idx = top_idx_unsorted[np.argsort(-sim[top_idx_unsorted])]
      top_sim_vals = sim[top_idx]
    _mat_ms = (time.time() - _t_mat) * 1000

    output = []
    for i, sv in zip(top_idx, top_sim_vals):
      output.append({
          "image_path": ALL_IMAGE_PATHS[i],
          "score": round(float(sv), 4),
          "video_name": get_video_name(i),
          "frame_id": int(FRAME_IDS_ARR[i]),
          "pts_time": float(PTS_TIMES_ARR[i]),
      })
    print(
        f"⏱️  Semantic search {time.time() - _t0:.3f}s "
        f"[encode {_enc_ms:.0f}ms | matmul {_mat_ms:.0f}ms | filter='{video_filter or 'none'}' {len(output)} kết quả]."
    )
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: ASR SEARCH (Đã tối ưu ma trận)
# ==========================================
@app.get("/api/search-asr")
def search_asr(
    prompt: str = Query(..., description="Query Text cho ASR"),
    top_k: int = 50,
    video_filter: str = Query("", description="Lọc theo prefix video"),
):
  if not prompt.strip():
    return {"results": []}
  try:
    _t0 = time.time()
    if embedding_model is None:
      return {"results": [], "error": "Embedding Gemma chưa tải được."}

    cache_key = (prompt.strip(), "asr")
    _t_enc = time.time()
    if cache_key in QUERY_VECTOR_CACHE:
      query_vector = QUERY_VECTOR_CACHE[cache_key]
    else:
      query_vector = embedding_model.encode_query(prompt).astype(np.float32)
      query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
      QUERY_VECTOR_CACHE[cache_key] = query_vector
    _enc_ms = (time.time() - _t_enc) * 1000

    if ASR_EMBEDDING_MATRIX.size == 0:
      return {"results": []}

    _t_mat = time.time()
    asr_mask = parse_asr_video_filter(video_filter)

    if asr_mask is not None:
      valid_idx = np.where(asr_mask)[0]
      if len(valid_idx) == 0:
        return {"results": []}
      sub_matrix = ASR_EMBEDDING_MATRIX[valid_idx]
      sim_sub = query_vector @ sub_matrix.T
      k = min(top_k, sim_sub.shape[0])
      top_sub = np.argpartition(-sim_sub, k - 1)[:k]
      top_sub = top_sub[np.argsort(-sim_sub[top_sub])]
      top_idx_arr = valid_idx[top_sub]
      sim_vals = sim_sub[top_sub]
    else:
      # ⚡ Sử dụng cú pháp rút gọn
      sim = query_vector @ ASR_EMBEDDING_MATRIX.T
      k = min(top_k, sim.shape[0])
      top_idx_arr = np.argpartition(-sim, k - 1)[:k]
      top_idx_arr = top_idx_arr[np.argsort(-sim[top_idx_arr])]
      sim_vals = sim[top_idx_arr]
    _mat_ms = (time.time() - _t_mat) * 1000

    output = []
    for idx, sv in zip(top_idx_arr, sim_vals):
      output.append({
          "text": ASR_TEXTS[idx],
          "score": round(float(sv), 4),
          "video_name": ASR_VIDEO_NAMES[idx],
          "image_path": ASR_IMAGE_PATHS[idx],
          "audio_path": ASR_AUDIO_PATHS[idx],
          "pts_time": float(ASR_PTS_TIMES[idx]),
          "frame_id": ASR_FRAME_IDS[idx],
      })
    print(
        f"⏱️  ASR search (RAM) {time.time() - _t0:.3f}s "
        f"[encode {_enc_ms:.0f}ms | matmul {_mat_ms:.0f}ms]."
    )
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: SEMANTIC + ASR (RRF, Đã tối ưu ma trận)
# ==========================================
@app.get("/api/search/semantic+asr")
def search_semantic_asr(
    prompt: str = Query(..., description="Query Text"),
    top_k: int = 50,
    video_filter: str = Query("", description="Lọc theo prefix video"),
):
  if not prompt.strip():
    return {"results": []}
  try:
    _t0 = time.time()
    filtered_idxs = parse_video_filter(video_filter)

    cache_key_sem = (prompt.strip(), "semantic")
    if cache_key_sem in QUERY_VECTOR_CACHE:
      query_vec = QUERY_VECTOR_CACHE[cache_key_sem]
    else:
      text_tokens = clip_tokenizer([prompt]).to(device)
      with torch.no_grad():
        query_features = clip_model.encode_text(text_tokens)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_vec = query_features.float().cpu().numpy().flatten()
      QUERY_VECTOR_CACHE[cache_key_sem] = query_vec

    if filtered_idxs is not None:
      if len(filtered_idxs) == 0:
        return {"results": []}
      sub_matrix = EMBEDDING_MATRIX[filtered_idxs]
      sim = query_vec.astype(np.float32) @ sub_matrix.T
      k_sem = min(200, sim.shape[0])
      top_sub = np.argpartition(-sim, k_sem - 1)[:k_sem]
      top_sub = top_sub[np.argsort(-sim[top_sub])]
      top_sem_idx = filtered_idxs[top_sub]
      sim_vals = sim[top_sub]
    else:
      # ⚡ Sử dụng cú pháp rút gọn
      sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T
      k_sem = min(200, sim.shape[0])
      top_sem_idx = np.argpartition(-sim, k_sem - 1)[:k_sem]
      top_sem_idx = top_sem_idx[np.argsort(-sim[top_sem_idx])]
      sim_vals = sim[top_sem_idx]

    semantic_ranks = {}
    semantic_data = {}
    for rank, (idx, sv) in enumerate(zip(top_sem_idx, sim_vals)):
      path = ALL_IMAGE_PATHS[idx]
      semantic_ranks[path] = rank
      semantic_data[path] = {
          "image_path": path,
          "video_name": get_video_name(idx),
          "frame_id": int(FRAME_IDS_ARR[idx]),
          "pts_time": float(PTS_TIMES_ARR[idx]),
      }

    asr_ranks = {}
    asr_data = {}
    if embedding_model is not None and ASR_EMBEDDING_MATRIX.size > 0:
      cache_key_asr = (prompt.strip(), "asr")
      if cache_key_asr in QUERY_VECTOR_CACHE:
        query_vector = QUERY_VECTOR_CACHE[cache_key_asr]
      else:
        query_vector = embedding_model.encode_query(prompt).astype(np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        QUERY_VECTOR_CACHE[cache_key_asr] = query_vector

      asr_mask = parse_asr_video_filter(video_filter)
      if asr_mask is not None:
        valid_idx = np.where(asr_mask)[0]
        if len(valid_idx) > 0:
          asr_sim = query_vector @ ASR_EMBEDDING_MATRIX[valid_idx].T
          k_asr = min(200, asr_sim.shape[0])
          top_asr_sub = np.argpartition(-asr_sim, k_asr - 1)[:k_asr]
          top_asr_sub = top_asr_sub[np.argsort(-asr_sim[top_asr_sub])]
          top_asr_idx = valid_idx[top_asr_sub]
          for rank, idx in enumerate(top_asr_idx):
            path = ASR_IMAGE_PATHS[idx]
            if path:
              asr_ranks[path] = rank
              asr_data[path] = {
                  "image_path": path,
                  "video_name": ASR_VIDEO_NAMES[idx],
                  "pts_time": float(ASR_PTS_TIMES[idx]),
                  "frame_id": ASR_FRAME_IDS[idx],
              }
      else:
        # ⚡ Sử dụng cú pháp rút gọn
        asr_sim = query_vector @ ASR_EMBEDDING_MATRIX.T
        k_asr = min(200, asr_sim.shape[0])
        top_asr_idx = np.argpartition(-asr_sim, k_asr - 1)[:k_asr]
        top_asr_idx = top_asr_idx[np.argsort(-asr_sim[top_asr_idx])]
        for rank, idx in enumerate(top_asr_idx):
          path = ASR_IMAGE_PATHS[idx]
          if path:
            asr_ranks[path] = rank
            asr_data[path] = {
                "image_path": path,
                "video_name": ASR_VIDEO_NAMES[idx],
                "pts_time": float(ASR_PTS_TIMES[idx]),
                "frame_id": ASR_FRAME_IDS[idx],
            }

    all_paths = set(semantic_ranks.keys()) | set(asr_ranks.keys())
    rrf_scores = []
    for path in all_paths:
      r_sem = semantic_ranks.get(path, 9999)
      r_asr = asr_ranks.get(path, 9999)

      score = 1.0 / (60.0 + r_sem) + 1.0 / (60.0 + r_asr)
      meta = semantic_data.get(path) or asr_data.get(path)
      rrf_scores.append((score, path, meta))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)

    output = []
    for score, path, meta in rrf_scores[:top_k]:
      output.append({
          "image_path": path,
          "score": round(score * 100, 4),
          "video_name": meta.get("video_name"),
          "frame_id": meta.get("frame_id"),
          "pts_time": meta.get("pts_time"),
      })
    print(f"⏱️  Semantic+ASR (RRF) hoàn tất trong {time.time() - _t0:.2f}s "
          f"({len(output)} kết quả).")
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: SEMANTIC + OCR (RRF, Đã tối ưu ma trận)
# ==========================================
@app.get("/api/search/semantic+ocr")
def search_semantic_ocr(
    prompt: str = Query(..., description="Query Text"), top_k: int = 50
):
  if not prompt.strip():
    return {"results": []}
  try:
    _t0 = time.time()
    cache_key_sem = (prompt.strip(), "semantic")
    if cache_key_sem in QUERY_VECTOR_CACHE:
      query_vec = QUERY_VECTOR_CACHE[cache_key_sem]
    else:
      text_tokens = clip_tokenizer([prompt]).to(device)
      with torch.no_grad():
        query_features = clip_model.encode_text(text_tokens)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_vec = query_features.float().cpu().numpy().flatten()
      QUERY_VECTOR_CACHE[cache_key_sem] = query_vec

    # ⚡ Sử dụng cú pháp rút gọn
    sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T

    k_sem = min(200, sim.shape[0])
    top_sem_idx = np.argpartition(-sim, k_sem - 1)[:k_sem]
    top_sem_idx = top_sem_idx[np.argsort(-sim[top_sem_idx])]

    semantic_ranks = {}
    semantic_data = {}
    for rank, idx in enumerate(top_sem_idx):
      path = ALL_IMAGE_PATHS[idx]
      semantic_ranks[path] = rank
      semantic_data[path] = {
          "image_path": path,
          "video_name": get_video_name(idx),
          "frame_id": int(FRAME_IDS_ARR[idx]),
          "pts_time": float(PTS_TIMES_ARR[idx]),
      }

    records = []
    try:
      records, _ = qdrant_client.scroll(
          collection_name=OCR_COLLECTION_NAME,
          scroll_filter=qmodels.Filter(
              should=[
                  qmodels.FieldCondition(
                      key="text", match=qmodels.MatchText(text=prompt)
                  ),
                  qmodels.FieldCondition(
                      key="ocr_text", match=qmodels.MatchText(text=prompt)
                  ),
                  qmodels.FieldCondition(
                      key="ocr", match=qmodels.MatchText(text=prompt)
                  ),
              ]
          ),
          limit=200,
          with_payload=True,
      )

      if not records and embedding_model is not None:
        query_vector = embedding_model.encode_query(prompt).astype(np.float32)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_list = query_vector.tolist()
        search_response = qdrant_client.query_points(
            collection_name=OCR_COLLECTION_NAME,
            query=query_list,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        records = search_response.points
      elif records:

        def _ocr_relevance_key(point):
          payload = getattr(point, "payload", {}) or {}
          prompt_lower = prompt.strip().lower()
          best_pos, best_len = None, None
          for field in ("text", "ocr_text", "ocr"):
            val = payload.get(field)
            if not val:
              continue
            val_lower = str(val).lower()
            pos = val_lower.find(prompt_lower)
            if pos != -1:
              if best_pos is None or pos < best_pos:
                best_pos = pos
                best_len = len(val_lower)
          if best_pos is None:
            return (1, 0, 0)
          return (0, best_pos, best_len or 0)

        records = sorted(records, key=_ocr_relevance_key)
    except Exception as e:
      print(f"[OCR Collection Exception] {e}")
      records = []

    ocr_ranks = {}
    ocr_data = {}
    for rank, p in enumerate(records):
      payload = getattr(p, "payload", {}) or {}
      path = payload.get("image_path")
      if path:
        path_clean = path.replace("\\", "/")
        ocr_ranks[path_clean] = rank
        ocr_data[path_clean] = {
            "image_path": path_clean,
            "video_name": payload.get("video_name"),
            "pts_time": payload.get("pts_time", 0.0),
            "frame_id": payload.get("frame_id", 0),
        }

    all_paths = set(semantic_ranks.keys()) | set(ocr_ranks.keys())
    rrf_scores = []
    for path in all_paths:
      r_sem = semantic_ranks.get(path, 9999)
      r_ocr = ocr_ranks.get(path, 9999)

      score = 1.0 / (60.0 + r_sem) + 1.0 / (60.0 + r_ocr)
      meta = semantic_data.get(path) or ocr_data.get(path)
      rrf_scores.append((score, path, meta))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)

    output = []
    for score, path, meta in rrf_scores[:top_k]:
      output.append({
          "image_path": path,
          "score": round(score * 100, 4),
          "video_name": meta.get("video_name"),
          "frame_id": meta.get("frame_id"),
          "pts_time": meta.get("pts_time"),
      })
    print(f"⏱️  Semantic+OCR (RRF) hoàn tất trong {time.time() - _t0:.2f}s "
          f"({len(output)} kết quả).")
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# ⭐ ROCCHIO RELEVANCE FEEDBACK (Đã tối ưu ma trận)
# ==========================================
class RocchioRequest(BaseModel):
  prompt: str = ""
  query_vector: Optional[list[float]] = None
  positive_paths: list[str] = []
  negative_paths: list[str] = []
  alpha: float = 1.0
  beta: float = 0.75
  gamma: float = 0.15
  top_k: int = 50
  embedding_type: str = "semantic"
  video_filter: str = ""


@app.post("/api/search/rocchio")
def search_rocchio(req: RocchioRequest):
  try:
    _t0 = time.time()
    use_asr = (req.embedding_type == "asr")

    matrix = ASR_EMBEDDING_MATRIX if use_asr else EMBEDDING_MATRIX
    path_to_idx = ASR_IMAGE_PATH_TO_IDX if use_asr else IMAGE_PATH_TO_IDX

    if matrix.size == 0:
      return {"results": [], "error": "Matrix rỗng."}

    dim = matrix.shape[1]

    cache_key = (req.prompt.strip(), req.embedding_type)
    if req.query_vector is not None and len(req.query_vector) == dim:
      v_query = np.array(req.query_vector, dtype=np.float32)
    elif cache_key in QUERY_VECTOR_CACHE:
      v_query = QUERY_VECTOR_CACHE[cache_key]
    elif req.prompt.strip():
      if use_asr and embedding_model is not None:
        v_query = embedding_model.encode_query(req.prompt).astype(np.float32)
        v_query = v_query / (np.linalg.norm(v_query) + 1e-8)
      else:
        text_tokens = clip_tokenizer([req.prompt]).to(device)
        with torch.no_grad():
          feats = clip_model.encode_text(text_tokens)
          feats /= feats.norm(dim=-1, keepdim=True)
          v_query = feats.float().cpu().numpy().flatten()
      QUERY_VECTOR_CACHE[cache_key] = v_query
    else:
      v_query = np.zeros(dim, dtype=np.float32)

    def _lookup_vectors(paths: list[str]) -> Optional[np.ndarray]:
      idxs = []
      for p in paths:
        p_clean = p.replace("\\", "/")
        idx = path_to_idx.get(p_clean)
        if idx is not None:
          idxs.append(idx)
      if not idxs:
        return None
      return matrix[np.array(idxs, dtype=np.int32)]

    _t_lookup = time.time()
    pos_vecs = _lookup_vectors(req.positive_paths)
    neg_vecs = _lookup_vectors(req.negative_paths)
    _lookup_ms = (time.time() - _t_lookup) * 1000

    v_new = req.alpha * v_query
    if pos_vecs is not None:
      v_new = v_new + req.beta * pos_vecs.mean(axis=0)
    if neg_vecs is not None:
      v_new = v_new - req.gamma * neg_vecs.mean(axis=0)

    norm = np.linalg.norm(v_new)
    if norm > 1e-8:
      v_new /= norm

    filtered_idxs = parse_video_filter(req.video_filter)

    _t_mat = time.time()
    if filtered_idxs is not None:
      if len(filtered_idxs) == 0:
        return {"results": [], "rocchio_vector": v_new.tolist()}
      sub_matrix = matrix[filtered_idxs]
      sim_sub = v_new @ sub_matrix.T
      k = min(req.top_k, sim_sub.shape[0])
      top_sub = np.argpartition(-sim_sub, k - 1)[:k]
      top_sub = top_sub[np.argsort(-sim_sub[top_sub])]
      top_idx = filtered_idxs[top_sub]
      sim_vals = sim_sub[top_sub]
    else:
      # ⚡ Sử dụng cú pháp rút gọn
      sim = v_new @ matrix.T
      k = min(req.top_k, sim.shape[0])
      top_idx = np.argpartition(-sim, k - 1)[:k]
      top_idx = top_idx[np.argsort(-sim[top_idx])]
      sim_vals = sim[top_idx]
    _mat_ms = (time.time() - _t_mat) * 1000

    if use_asr:
      output = []
      for idx, sv in zip(top_idx, sim_vals):
        output.append({
            "image_path": ASR_IMAGE_PATHS[idx],
            "score": round(float(sv), 4),
            "video_name": ASR_VIDEO_NAMES[idx],
            "frame_id": ASR_FRAME_IDS[idx],
            "pts_time": float(ASR_PTS_TIMES[idx]),
            "text": ASR_TEXTS[idx],
        })
    else:
      output = []
      for idx, sv in zip(top_idx, sim_vals):
        output.append({
            "image_path": ALL_IMAGE_PATHS[idx],
            "score": round(float(sv), 4),
            "video_name": get_video_name(idx),
            "frame_id": int(FRAME_IDS_ARR[idx]),
            "pts_time": float(PTS_TIMES_ARR[idx]),
        })

    total_ms = (time.time() - _t0) * 1000
    print(
        f"⏱️  Rocchio search {total_ms:.0f}ms total "
        f"[lookup {_lookup_ms:.0f}ms | matmul {_mat_ms:.0f}ms | "
        f"D+={len(req.positive_paths)} D-={len(req.negative_paths)}]"
    )
    return {
        "results": output,
        "rocchio_vector": v_new.tolist(),
        "stats": {
            "positive_found": pos_vecs.shape[0] if pos_vecs is not None else 0,
            "negative_found": neg_vecs.shape[0] if neg_vecs is not None else 0,
            "total_ms": round(total_ms, 1),
        },
    }
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: TÌM KIẾM CHÍNH XÁC (SHA-256, Đã tối ưu ma trận)
# ==========================================
@app.post("/api/lookup/image")
async def lookup_image_exact(file: UploadFile = File(...), top_k: int = 50):
  try:
    _t0 = time.time()
    image_bytes = await file.read()
    file_hash = hashlib.sha256(image_bytes).hexdigest()

    if file_hash in HASH_INDEX:
      meta = HASH_INDEX[file_hash]
      print(f"⏱️  Reverse Lookup (EXACT hash) hoàn tất trong "
            f"{time.time() - _t0:.3f}s.")
      return {
          "match_type": "exact",
          "results": [{
              "video_name": meta.get("video_name"),
              "frame_id": meta.get("frame_id"),
              "image_path": meta.get("image_path"),
              "pts_time": meta.get("pts_time", 0.0),
              "score": 1.0,
          }],
      }

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
      image_features = clip_model.encode_image(image_tensor)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      query_vec = image_features.float().cpu().numpy().flatten()

    # ⚡ Sử dụng cú pháp rút gọn
    sim = query_vec.astype(np.float32) @ EMBEDDING_MATRIX.T
    k = min(top_k, sim.shape[0])
    top_idx_unsorted = np.argpartition(-sim, k - 1)[:k]
    top_idx = top_idx_unsorted[np.argsort(-sim[top_idx_unsorted])]

    output = []
    for i in top_idx:
      output.append({
          "image_path": ALL_IMAGE_PATHS[i],
          "score": round(float(sim[i]), 4),
          "video_name": get_video_name(i),
          "frame_id": int(FRAME_IDS_ARR[i]),
          "pts_time": float(PTS_TIMES_ARR[i]),
      })
    print(f"⏱️  Reverse Lookup (APPROXIMATE, CLIP fallback) hoàn tất trong "
          f"{time.time() - _t0:.2f}s ({len(output)} kết quả).")
    return {"match_type": "approximate", "results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: TÌM KIẾM LÂN CẬN (Sequential Frames)
# ==========================================
@app.get("/api/search/sequential-frames")
def search_sequential_frames(
    video_name: str = Query(..., description="Tên video, VD: L21_V003"),
    center_frame_id: str = Query(
        ..., description="Số thứ tự (n) của frame gốc, VD: 010"
    ),
):
  try:
    _t0 = time.time()
    clean_fid = (
        center_frame_id.replace("#", "").replace("img_", "").split(".")[0].strip()
    )
    try:
      center_num = int(clean_fid)
    except ValueError:
      return JSONResponse(
          status_code=400,
          content={
              "status": "error",
              "message": (
                  f"center_frame_id '{center_frame_id}' không phải số hợp lệ."
              ),
          },
      )

    idxs = VIDEO_TO_INDICES.get(video_name)
    if idxs is None or len(idxs) == 0:
      return JSONResponse(
          status_code=404,
          content={
              "status": "error",
              "message": (
                  f"Không tìm thấy video '{video_name}' trong dữ liệu đã nạp."
              ),
          },
      )

    # ±10 khung quanh frame trung tâm -> tổng 20 khung hình hiển thị timeline
    # (trước là ±5 = 10 khung).
    low, high = center_num - 10, center_num + 10

    matched = []
    for i in idxs:
      path = ALL_IMAGE_PATHS[i]
      if not path:
        continue
      fname = path.split("/")[-1]
      num_part = fname.split(".")[0]
      try:
        n = int(num_part)
      except ValueError:
        continue
      if low <= n <= high:
        matched.append((n, i))

    matched.sort(key=lambda x: x[0])

    results = []
    for n, i in matched:
      results.append({
          "video_name": video_name,
          "frame_id": int(FRAME_IDS_ARR[i]),
          "image_path": ALL_IMAGE_PATHS[i],
          "pts_time": float(PTS_TIMES_ARR[i]),
          "score": 100.0 if n == center_num else 90.0,
      })

    if not results:
      return JSONResponse(
          status_code=404,
          content={
              "status": "error",
              "message": (
                  f"Không có frame nào trong khoảng [{low}, {high}] của video"
                  f" '{video_name}'."
              ),
          },
      )

    print(f"⏱️  Sequential-frames hoàn tất trong {time.time() - _t0:.3f}s "
          f"({len(results)} kết quả, video='{video_name}').")
    return {"status": "success", "results": results}

  except Exception as e:
    return JSONResponse(
        status_code=500, content={"status": "error", "message": str(e)}
    )


# ==========================================
# API: TraKE Search (Đã tối ưu ma trận)
# ==========================================
class TrakeRequest(BaseModel):
  trek1: str = ""
  trek2: str = ""
  trek3: str = ""
  trek4: str = ""
  context: str = ""
  context_weight: float = 0.3  # 0 = bỏ qua context, 1 = chỉ dùng context
  top_k: int = 5
  video_filter: str = ""


@app.post("/api/search/trake")
def search_trake(req: TrakeRequest):
  try:
    _t0 = time.time()
    queries = [req.trek1, req.trek2, req.trek3, req.trek4]
    active_queries = [q for q in queries if q.strip()]

    if not active_queries:
      return {"results": []}

    num_events = len(active_queries)
    text_tokens = clip_tokenizer(active_queries).to(device)

    with torch.no_grad():
      text_feats = clip_model.encode_text(text_tokens)
      text_feats /= text_feats.norm(dim=-1, keepdim=True)

      # 🆕 CONTEXT: nếu người dùng có nhập bối cảnh chung (vd: "một buổi nấu
      # ăn trong bếp"), encode riêng bằng CLIP rồi TRỘN VECTOR (không nối
      # chuỗi text) vào từng event. Trộn vector tránh được rủi ro vượt giới
      # hạn 77 token của CLIP khi context dài — mỗi phần (context, event)
      # được encode độc lập trong giới hạn của chính nó, rồi mới kết hợp.
      context_text = req.context.strip()
      if context_text:
        ctx_tokens = clip_tokenizer([context_text]).to(device)
        ctx_feat = clip_model.encode_text(ctx_tokens)
        ctx_feat /= ctx_feat.norm(dim=-1, keepdim=True)

        w = max(0.0, min(1.0, req.context_weight))
        blended = (1.0 - w) * text_feats + w * ctx_feat  # broadcast (E,D)+(1,D)
        blended = blended / blended.norm(dim=-1, keepdim=True)
        text_feats = blended

      text_vecs = text_feats.float().cpu().numpy()

    if req.video_filter.strip():
      filtered_video_names = [
          v_name for v_name in VIDEO_TO_INDICES
          if any(v_name.upper().startswith(px.strip().upper())
                 for px in req.video_filter.split(",") if px.strip())
      ]
    else:
      filtered_video_names = list(VIDEO_TO_INDICES.keys())

    if not filtered_video_names:
      return {"results": []}

    # ✅ SỬA LỖI: build đúng 1 sub-matrix nối các video đã lọc theo ĐÚNG thứ
    # tự offset bên dưới, rồi mới nhân ma trận trên sub-matrix đó — thay vì
    # nhân trên FULL EMBEDDING_MATRIX rồi cắt theo offset "ảo" (bug cũ khiến
    # score bị lấy nhầm từ video/frame khác, làm kết quả TraKE sai lệch).
    concat_offset = {}
    offset = 0
    flat_idxs = []
    for v_name in filtered_video_names:
      idxs = VIDEO_TO_INDICES[v_name]
      concat_offset[v_name] = offset
      offset += len(idxs)
      flat_idxs.extend(idxs)

    flat_idxs = np.asarray(flat_idxs, dtype=np.int64)

    # ⚡ Chỉ nhân ma trận trên các frame thuộc video đã lọc -> đúng VÀ nhanh
    # hơn khi video_filter thu hẹp tập video (so với nhân trên full matrix).
    sub_matrix = EMBEDDING_MATRIX[flat_idxs]
    sim_all = text_vecs.astype(np.float32) @ sub_matrix.T

    video_candidates = {}
    for v_name in filtered_video_names:
      idxs = VIDEO_TO_INDICES[v_name]
      off = concat_offset[v_name]
      v_frame_ids = FRAME_IDS_ARR[idxs]
      v_pts_times = PTS_TIMES_ARR[idxs]
      v_image_paths = [ALL_IMAGE_PATHS[i] for i in idxs]

      video_candidates[v_name] = {}
      for e_idx in range(num_events):
        v_scores = sim_all[e_idx][off: off + len(idxs)]
        items = [
            (
                float(v_pts_times[i]),
                float(v_scores[i]),
                int(v_frame_ids[i]),
                v_image_paths[i],
            )
            for i in range(len(idxs))
        ]
        video_candidates[v_name][e_idx] = items

    top_seqs = find_best_trake_dynamic(
        video_candidates, num_events,
        top_k=req.top_k,
        max_duration_sec=300.0,
        gap_penalty=0.0008,
    )

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
            "score": round(item["score"], 4),
        })

    _ctx_info = (
        f"context weight={req.context_weight}" if req.context.strip() else "no context"
    )
    print(
        f"⏱️  TraKE search hoàn tất trong {time.time() - _t0:.2f}s"
        f" ({num_events} sự kiện, {len(filtered_video_names)} video, {_ctx_info})."
    )

    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


# ==========================================
# API: FIND SIMILAR (image-to-image, dùng lại vector CLIP đã lập chỉ mục)
# ==========================================
@app.get("/api/search/similar")
def search_similar(
    image_path: str = Query(
        ..., description="image_path của keyframe gốc (lấy từ Keyframe Inspector)"
    ),
    top_k: int = 50,
    video_filter: str = Query("", description="Lọc theo prefix video, VD: L21,L22"),
):
  try:
    _t0 = time.time()
    norm_path = image_path.replace("\\", "/")
    src_idx = IMAGE_PATH_TO_IDX.get(norm_path)
    if src_idx is None:
      return {
          "results": [],
          "error": f"Không tìm thấy keyframe với image_path='{image_path}' trong index.",
      }

    query_vec = EMBEDDING_MATRIX[src_idx]
    filtered_idxs = parse_video_filter(video_filter)

    # Lấy dư 1 kết quả để bù cho việc loại bỏ chính ảnh gốc khỏi output.
    if filtered_idxs is not None:
      if len(filtered_idxs) == 0:
        return {"results": []}
      sub_matrix = EMBEDDING_MATRIX[filtered_idxs]
      sim_sub = query_vec @ sub_matrix.T
      k = min(top_k + 1, sim_sub.shape[0])
      top_sub_unsorted = np.argpartition(-sim_sub, k - 1)[:k]
      top_sub = top_sub_unsorted[np.argsort(-sim_sub[top_sub_unsorted])]
      top_idx = filtered_idxs[top_sub]
      top_sim_vals = sim_sub[top_sub]
    else:
      sim = query_vec @ EMBEDDING_MATRIX.T
      k = min(top_k + 1, sim.shape[0])
      top_idx_unsorted = np.argpartition(-sim, k - 1)[:k]
      top_idx = top_idx_unsorted[np.argsort(-sim[top_idx_unsorted])]
      top_sim_vals = sim[top_idx]

    output = []
    for i, sv in zip(top_idx, top_sim_vals):
      if int(i) == src_idx:
        continue  # bỏ chính ảnh gốc khỏi kết quả Find Similar
      output.append({
          "image_path": ALL_IMAGE_PATHS[i],
          "score": round(float(sv), 4),
          "video_name": get_video_name(i),
          "frame_id": int(FRAME_IDS_ARR[i]),
          "pts_time": float(PTS_TIMES_ARR[i]),
      })
      if len(output) >= top_k:
        break

    print(
        f"⏱️  Find Similar (CLIP) hoàn tất trong {time.time() - _t0:.3f}s"
        f" ({len(output)} kết quả, filter='{video_filter or 'none'}')."
    )
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


@app.get("/api/random")
def get_random_keyframes(limit: int = 50):
  try:
    _t0 = time.time()
    n = len(ALL_IMAGE_PATHS)
    if n == 0:
      return {"results": []}
    rand_idxs = py_random.sample(range(n), min(limit, n))
    output = []
    for i in rand_idxs:
      output.append({
          "image_path": ALL_IMAGE_PATHS[i],
          "score": "RAND",
          "video_name": get_video_name(i),
          "frame_id": int(FRAME_IDS_ARR[i]),
          "pts_time": float(PTS_TIMES_ARR[i]),
      })
    print(f"⏱️  Random keyframes (RAM) hoàn tất trong {time.time() - _t0:.3f}s "
          f"({len(output)} kết quả, 0 Qdrant calls).")
    return {"results": output}
  except Exception as e:
    return {"results": [], "error": str(e)}


@app.get("/api/image")
def get_local_image(path: str):
  win_path = resolve_image_abs_path(path)

  if os.path.exists(win_path):
    return FileResponse(win_path)
  return {"error": f"File not found at {win_path} (đã thử qua: {BASE_IMAGE_DIRS})"}


@app.get("/api/video")
def get_local_video(video_name: str):
  video_path = resolve_video_abs_path(video_name)

  if os.path.exists(video_path):
    ext = os.path.splitext(video_path)[1].lower()
    mime = VIDEO_MIME_TYPES.get(ext, "video/mp4")
    return FileResponse(video_path, media_type=mime)
  return {"error": f"Video not found at {video_path} (đã thử qua: {VIDEO_DIRS} x {VIDEO_EXTENSIONS})"}


# ==========================================
# QUY TẮC NỘP BÀI CSV (AIC26)
# ==========================================
MAX_ROWS_PER_CSV = 100      # Giới hạn số dòng tối đa / 1 file CSV
MAX_QA_ANSWER_LENGTH = 100  # Giới hạn ký tự tối đa cho Answer (dạng Q&A)

_VIDEO_EXT_RE = re.compile(r"\.(mp4|avi|mkv|mov|webm|mpg|mpeg)$", re.IGNORECASE)


def _clean_video_name_for_csv(video_name: str) -> str:
  """Chuẩn hoá tên video theo quy tắc: dạng mã (VD: L01_V028), KHÔNG có
  phần mở rộng. Tự động cắt đuôi nếu người dùng lỡ gõ kèm (vd .mp4)."""
  name = video_name.strip()
  name = _VIDEO_EXT_RE.sub("", name)
  return name


def _clean_frame_id_for_csv(frame_id: str) -> str:
  """Chuẩn hoá Frame ID: bắt buộc là số nguyên, không khoảng trắng thừa,
  tự động bỏ số 0 dư ở đầu (vd '0100' -> '100'). Raise ValueError nếu
  không phải số nguyên hợp lệ."""
  raw = frame_id.replace("#", "").strip()
  if not raw:
    raise ValueError("Frame ID đang rỗng.")
  try:
    return str(int(raw))
  except ValueError:
    raise ValueError(f"Frame ID '{raw}' không phải số nguyên hợp lệ.")


def _count_existing_csv_rows(file_path: str) -> int:
  """Đếm đúng SỐ DÒNG DỮ LIỆU (logical rows) đã có trong file, dùng
  csv.reader thay vì đếm dòng thô — vì Answer của Q&A có thể chứa ký tự
  xuống dòng bên trong dấu ngoặc kép, đếm dòng thô sẽ bị sai lệch."""
  if not os.path.exists(file_path):
    return 0
  with open(file_path, "r", encoding="utf-8", newline="") as f:
    return sum(1 for _ in csv.reader(f))


@app.post("/api/submit-csv")
def submit_to_csv(
    mode: str = Query(
        "semantic", description="Chế độ hiện tại: semantic, vqa, trake"
    ),
    video_name: str = Query(..., description="Tên video (vd: L30_V057)"),
    frame_id: str = Query(
        ..., description="Frame ID chính (hoặc chuỗi frame cho TraKE)"
    ),
    filename: str = Query(..., description="Tên file CSV muốn lưu"),
    vqa_answer: str = Query("", description="Đáp án VQA nếu có"),
):
  try:
    _t0 = time.time()
    output_dir = r"C:\Users\XPS 15 9570\Downloads\submission"
    os.makedirs(output_dir, exist_ok=True)

    if not filename.endswith(".csv"):
      filename += ".csv"

    file_path = os.path.join(output_dir, filename)

    # ✅ QUY TẮC 8: tối đa 100 dòng / 1 file CSV.
    existing_rows = _count_existing_csv_rows(file_path)
    if existing_rows >= MAX_ROWS_PER_CSV:
      return {
          "status": "error",
          "message": (
              f"File '{filename}' đã đạt giới hạn tối đa {MAX_ROWS_PER_CSV}"
              " dòng theo quy định nộp bài. Vui lòng tạo file CSV khác."
          ),
      }

    try:
      clean_video_name = _clean_video_name_for_csv(video_name)
    except ValueError as e:
      return {"status": "error", "message": str(e)}

    row_data = []

    if mode == "trake":
      try:
        raw_frames = [f for f in frame_id.split(",") if f.strip()]
        frames = [_clean_frame_id_for_csv(f) for f in raw_frames]
      except ValueError as e:
        return {"status": "error", "message": str(e)}

      if len(frames) < 2:
        return {
            "status": "error",
            "message": (
                "TraKE cần ít nhất 2 Frame ID tương ứng với các sự kiện đã"
                " chọn."
            ),
        }

      # ✅ Bảo đảm đúng thứ tự thời gian (frame_id lớn hơn = thời điểm
      # muộn hơn trong cùng 1 video) — sắp xếp tăng dần để tránh sai thứ
      # tự nếu người dùng lỡ xác nhận không theo trình tự sự kiện.
      frames_sorted = sorted(frames, key=lambda x: int(x))
      row_data = [clean_video_name] + frames_sorted

    elif mode == "vqa":
      try:
        clean_frame_id = _clean_frame_id_for_csv(frame_id)
      except ValueError as e:
        return {"status": "error", "message": str(e)}

      answer = vqa_answer.strip()
      if len(answer) > MAX_QA_ANSWER_LENGTH:
        return {
            "status": "error",
            "message": (
                f"Answer dài {len(answer)} ký tự, vượt quá giới hạn"
                f" {MAX_QA_ANSWER_LENGTH} ký tự cho phép của Q&A."
            ),
        }

      # ⚠️ KHÔNG tự bọc dấu ngoặc kép / tự escape ở đây. csv.writer bên
      # dưới (quoting=QUOTE_MINIMAL) đã TỰ ĐỘNG thêm và escape dấu " đúng
      # chuẩn RFC4180 khi answer chứa dấu phẩy, dấu " hoặc xuống dòng —
      # tự bọc thêm ở tầng này sẽ khiến dấu " bị escape 2 LẦN (double-
      # escape), sinh ra file CSV sai định dạng so với thể lệ.
      row_data = [clean_video_name, clean_frame_id, answer]

    else:
      try:
        clean_frame_id = _clean_frame_id_for_csv(frame_id)
      except ValueError as e:
        return {"status": "error", "message": str(e)}
      row_data = [clean_video_name, clean_frame_id]

    with open(file_path, mode="a", encoding="utf-8", newline="") as f:
      writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
      writer.writerow(row_data)

    new_row_count = existing_rows + 1
    print(
        f"⏱️  Submit CSV hoàn tất trong {time.time() - _t0:.3f}s "
        f"(mode='{mode}', file='{filename}', dòng {new_row_count}/{MAX_ROWS_PER_CSV})."
    )
    return {
        "status": "success",
        "message": (
            f"Đã lưu thành công vào {file_path}"
            f" ({new_row_count}/{MAX_ROWS_PER_CSV} dòng)"
        ),
    }
  except Exception as e:
    return {"status": "error", "message": str(e)}


# ==========================================
# NỘP BÀI QUA DRES API (proxy — tránh lỗi CORS khi gọi thẳng từ trình
# duyệt sang server ban tổ chức, vd. https://eventretrieval.oj.io.vn/api/v2)
# ==========================================
class DresLoginRequest(BaseModel):
  base_url: str
  username: str
  password: str


class DresSubmitRequest(BaseModel):
  base_url: str
  session: str
  evaluation_id: str
  answer_sets: list


@app.post("/api/dres/login")
def dres_login(payload: DresLoginRequest):
  """Proxy POST {base_url}/login — trả về nguyên vẹn JSON + status code từ
  DRES (chứa sessionId khi thành công, hoặc description khi lỗi)."""
  try:
    url = f"{payload.base_url.rstrip('/')}/login"
    resp = requests.post(
        url,
        json={"username": payload.username, "password": payload.password},
        timeout=15,
    )
    try:
      data = resp.json()
    except Exception:
      data = {"description": resp.text}
    return JSONResponse(content=data, status_code=resp.status_code)
  except Exception as e:
    return JSONResponse(
        content={"description": f"Không kết nối được tới DRES: {e}"},
        status_code=502,
    )


@app.get("/api/dres/evaluations")
def dres_evaluations(
    base_url: str = Query(..., description="Base URL của server DRES"),
    session: str = Query(..., description="sessionId lấy được từ /api/dres/login"),
):
  """Proxy GET {base_url}/client/evaluation/list?session=... — trả về
  nguyên vẹn danh sách evaluation từ DRES."""
  try:
    url = f"{base_url.rstrip('/')}/client/evaluation/list"
    resp = requests.get(url, params={"session": session}, timeout=15)
    try:
      data = resp.json()
    except Exception:
      data = {"description": resp.text}
    return JSONResponse(content=data, status_code=resp.status_code)
  except Exception as e:
    return JSONResponse(
        content={"description": f"Không kết nối được tới DRES: {e}"},
        status_code=502,
    )


@app.post("/api/dres/submit")
def dres_submit(payload: DresSubmitRequest):
  """Proxy POST {base_url}/submit/{evaluation_id}?session=... với body
  {"answerSets": answer_sets} — dùng chung cho cả KIS/QA/TRAKE, vì phần
  phân loại + build answerSets đã được xử lý ở frontend (index.html)."""
  try:
    url = f"{payload.base_url.rstrip('/')}/submit/{payload.evaluation_id}"
    resp = requests.post(
        url,
        params={"session": payload.session},
        json={"answerSets": payload.answer_sets},
        timeout=15,
    )
    try:
      data = resp.json()
    except Exception:
      data = {"description": resp.text}
    return JSONResponse(content=data, status_code=resp.status_code)
  except Exception as e:
    return JSONResponse(
        content={"description": f"Không kết nối được tới DRES: {e}"},
        status_code=502,
    )
@app.get("/api/dres/state")
def dres_evaluation_state(
    base_url: str = Query(..., description="Base URL của server DRES"),
    session: str = Query(..., description="sessionId lấy được từ /api/dres/login"),
):
  """Proxy GET {base_url}/evaluation/state/list?session=... — trả về mảng
  ApiEvaluationState (mỗi evaluation đang chạy kèm timeElapsed/timeLeft
  tính bằng giây). Dùng để hiển thị đồng hồ đếm ngược giống BTC."""
  try:
    url = f"{base_url.rstrip('/')}/evaluation/state/list"
    resp = requests.get(url, params={"session": session}, timeout=15)
    try:
      data = resp.json()
    except Exception:
      data = {"description": resp.text}
    return JSONResponse(content=data, status_code=resp.status_code)
  except Exception as e:
    return JSONResponse(
        content={"description": f"Không kết nối được tới DRES: {e}"},
        status_code=502,
    )


@app.get("/api/dres/current-task")
def dres_current_task(
    base_url: str = Query(..., description="Base URL của server DRES"),
    session: str = Query(..., description="sessionId lấy được từ /api/dres/login"),
    evaluation_id: str = Query(..., description="Evaluation ID đang theo dõi"),
):
  """Proxy GET {base_url}/client/evaluation/currentTask/{evaluationId}?session=...
  — trả về thông tin task hiện tại (tên, loại truy vấn, thời lượng tối đa)."""
  try:
    url = f"{base_url.rstrip('/')}/client/evaluation/currentTask/{evaluation_id}"
    resp = requests.get(url, params={"session": session}, timeout=15)
    try:
      data = resp.json()
    except Exception:
      data = {"description": resp.text}
    return JSONResponse(content=data, status_code=resp.status_code)
  except Exception as e:
    return JSONResponse(
        content={"description": f"Không kết nối được tới DRES: {e}"},
        status_code=502,
    )


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)
