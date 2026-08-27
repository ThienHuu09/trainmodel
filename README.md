 pip install -r requirements.txt

trake embedding: https://www.kaggle.com/datasets/bivnvinh/embedding-batch-1-trake

ocr embedding : https://www.kaggle.com/datasets/thinhunguyn/ocrhehe

asr embedding: https://www.kaggle.com/datasets/emb3rw/asr-gemma/data

semantic embedding: https://www.kaggle.com/datasets/aic2026cuathinhuu/fixedbatch1

dataset: https://www.kaggle.com/datasets/thinhunguyn/aic2026

mapkeyframes: https://www.kaggle.com/datasets/thinhunguyn/map-keyframes26

videos: https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?pli=1&gid=0#gid=0

## 📂 Cấu trúc Thư mục Dự án AIC2026

```text
C:\AIC2026\
│
├── dataset_webp/               # Thư mục chứa dữ liệu ảnh keyframe định dạng WebP
├── map-keyframes/              # Thư mục ánh xạ dữ liệu keyframe
├── video/                      # Thư mục lưu trữ các file video gốc (.mp4)
├── qdrant-x86_64-pc-windows/   # Thư mục cơ sở dữ liệu vector Qdrant
│
└── Mfusion-VR-Web/             # Thư mục mã nguồn hệ thống web chính
    ├── frontend/               # Thư mục giao diện (chứa file index.html và style.css)
    ├── server.py               # File cấu hình và chạy FastAPI Backend chính

