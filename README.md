trake embedding: https://www.kaggle.com/datasets/bivnvinh/json-cui-cng

ocr embedding : https://www.kaggle.com/datasets/thinhunguyn/ocrhehe

json embedding: https://www.kaggle.com/datasets/emb3rw/finalasr/data?fbclid=IwY2xjawT04_lwZG9mA2V4dG4DYWVtAjExAHNydGMGYXBwX2lkATAAAR6pVtPXY09zId3dRYkgDRf3KCh7qRwGTddDS4BHTQptzAjU4L3rcw7QBjpyOw_aem_PbV_Cz6Cy9hgUdpamyuszw

semantic embedding: https://www.kaggle.com/datasets/thinhunguyn/jsonaic

dataset: https://www.kaggle.com/datasets/thinhunguyn/aic2026

mapkeyframes: https://www.kaggle.com/datasets/thinhunguyn/map-keyframes26

videos: https://www.kaggle.com/datasets/emb3rw/data-asr-model?select=video

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
