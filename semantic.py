import os
import time
import uuid
import argparse
import ijson
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

QDRANT_HOST = "localhost"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334
USE_GRPC = True

COLLECTION_NAME = "dfn5b_images"
JSON_FILE_PATH = "semanticbatchone_full.json"
VECTOR_SIZE = 1024

BATCH_SIZE = 2000
PARALLEL = max(4, min(8, os.cpu_count() or 4))
MAX_RETRIES = 5
INDEXING_OFF = 1_000_000
INDEXING_ON = 20_000


def stable_point_id(video_name, frame_id) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_name}:{frame_id}"))


def l2_normalize(embedding) -> list:
    vec = np.asarray(embedding, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec = vec / n
    return vec.tolist()


def ensure_collection(client: QdrantClient, recreate: bool):
    exists = client.collection_exists(collection_name=COLLECTION_NAME)

    if exists and recreate:
        print(f"[WARN] Xóa collection cũ '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"[INFO] Tạo collection '{COLLECTION_NAME}' (cosine/{VECTOR_SIZE}, HNSW ef_construct=256)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=256,
                on_disk=True,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_OFF,
            ),
            on_disk_payload=True,
        )
    else:
        print(f"[INFO] Dùng lại collection '{COLLECTION_NAME}' (upsert theo id video+frame).")
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_OFF),
        )


def wait_points(client: QdrantClient, expected: int, stable_rounds: int = 8):
    last = -1
    same = 0
    while True:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        count = info.points_count or 0
        print(f"[WAIT] points={count}/{expected}")
        if expected > 0 and count >= expected:
            return info
        if count == last:
            same += 1
            if same >= stable_rounds:
                print(f"[WARN] Số điểm không tăng nữa ({count}/{expected}).")
                return info
        else:
            same = 0
            last = count
        time.sleep(2)


def enable_indexing_after_upload(client: QdrantClient, total_points: int):
    print("[INFO] Bật HNSW và chờ index xong...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_ON),
    )
    while True:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        indexed = info.indexed_vectors_count or 0
        status = str(info.status).lower()
        print(f"[WAIT] status={status} indexed={indexed}/{total_points}")
        if "green" in status and (indexed >= total_points or indexed >= (info.points_count or 0)):
            break
        time.sleep(3)


def ensure_payload_index(client: QdrantClient):
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="video_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def iter_points_from_json(path: str, counters: dict):
    seen_ids = set()
    with open(path, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            embedding = item.get("embedding")
            if embedding is None or len(embedding) != VECTOR_SIZE:
                counters["skipped"] += 1
                continue

            video_name = item.get("video_name")
            frame_id = item.get("frame_id")
            if video_name is None or frame_id is None:
                counters["skipped"] += 1
                continue

            pid = stable_point_id(video_name, frame_id)
            counters["read"] += 1
            if pid in seen_ids:
                counters["dup"] += 1
            else:
                seen_ids.add(pid)
            counters["unique"] = len(seen_ids)

            yield models.PointStruct(
                id=pid,
                vector=l2_normalize(embedding),
                payload={
                    "video_name": video_name,
                    "frame_id": frame_id,
                    "pts_time": item.get("pts_time", 0.0),
                    "image_path": item.get("image_path"),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Xóa collection cũ rồi tạo lại")
    parser.add_argument("--json", default=JSON_FILE_PATH, help="File JSON embedding")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        raise FileNotFoundError(args.json)

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=USE_GRPC,
        timeout=120,
    )

    ensure_collection(client, recreate=args.recreate)

    counters = {"read": 0, "skipped": 0, "dup": 0, "unique": 0}
    points_gen = iter_points_from_json(args.json, counters)

    def progress_wrapped_gen():
        pbar = tqdm(desc="Upload DFN5B", unit=" điểm")
        for p in points_gen:
            pbar.update(1)
            yield p
        pbar.close()

    t0 = time.time()
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=progress_wrapped_gen(),
        batch_size=BATCH_SIZE,
        parallel=PARALLEL,
        max_retries=MAX_RETRIES,
        wait=False,
    )

    uploaded = counters["unique"] or counters["read"]
    print(
        f"[INFO] Đọc {counters['read']} bản ghi, unique={counters['unique']}, "
        f"trùng id={counters['dup']}, bỏ {counters['skipped']}. Chờ Qdrant nhận hết..."
    )
    wait_points(client, uploaded)
    enable_indexing_after_upload(client, uploaded)
    ensure_payload_index(client)

    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(
        f"[DONE] {time.time() - t0:.1f}s | points={info.points_count} "
        f"| indexed={info.indexed_vectors_count} | status={info.status}"
    )
    print("[NOTE] Chỉ search sau khi status=green và indexed == points.")


if __name__ == "__main__":
    main()