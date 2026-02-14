import os
import sys
import json
import re
import csv
import time
import requests
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import open_clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. ORCHESTRATION CONFIG
# =============================================================================
HF_DATASET_URL = "https://huggingface.co/datasets/yhcao/V3Det_Backup"
BASE_DL_URL = "https://huggingface.co/datasets/yhcao/V3Det_Backup/resolve/main"

# File maps
ANNO_TRAIN = "v3det_2023_v1_train.json"
ZIPS = [f"zips/{i:05d}.zip" for i in range(21)]

# Local Paths
WORKSPACE = Path("v3det_workspace")
RAW_DIR = WORKSPACE / "raw_zips"
IMAGES_DIR = WORKSPACE / "images"  # Only for verification
FEATURES_DIR = WORKSPACE / "features"
WEIGHTS_PATH = WORKSPACE / "v3det_text_encoder.pt"
RESULT_CSV = WORKSPACE / "v3det_dataset_index.csv"

# Hyperparams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 512
HIDDEN_DIM = 512
BATCH_SIZE = 1024
EPOCHS = 20
LR = 5e-4
TEMPERATURE = 0.07

# =============================================================================
# 2. MODEL DEFINITION (Contrastive Text Encoder)
# =============================================================================
# Based on the user's project but upgraded for PyTorch/GPU
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.gru = nn.GRU(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        # hidden: (1, batch, hidden_dim)
        out = self.fc(hidden.squeeze(0))
        return out / out.norm(dim=-1, keepdim=True)

# =============================================================================
# 3. UTILS & DOWNLOADER
# =============================================================================

def download(url, path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=path.name) as bar:
             for chunk in r.iter_content(chunk_size=8192):
                 f.write(chunk)
                 bar.update(len(chunk))

def tokenize(text, word2idx):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = text.split()
    return [word2idx.get(t, word2idx['<unk>']) for t in tokens if t in word2idx]

# =============================================================================
# 4. DATA PIPELINE
# =============================================================================

def run_pipeline():
    print(f"Starting V3Det Integration on {DEVICE}")
    WORKSPACE.mkdir(exist_ok=True)
    FEATURES_DIR.mkdir(exist_ok=True)

    # A. Annotations
    anno_path = WORKSPACE / ANNO_TRAIN
    download(f"{BASE_DL_URL}/{ANNO_TRAIN}", anno_path)
    
    with open(anno_path, "r") as f:
        data = json.load(f)
    
    # Map cats and images
    cats = {c['id']: c['name'] for c in data['categories']}
    # File name -> Metadata (V3Det file_name contains the path like images/n.../...)
    img_map = {img['file_name']: img for img in data['images']}
    
    # img_id -> list of category names
    img_to_cats = {}
    for ann in data['annotations']:
        iid = ann['image_id']
        cid = ann['category_id']
        if iid not in img_to_cats: img_to_cats[iid] = set()
        img_to_cats[iid].add(cats[cid])

    # B. Feature Extraction (CLIP)
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model_clip.to(DEVICE).eval()
    
    dataset_records = [] # (features, text)

    print("\n--- Phase 2: Processing Zips (Download & Feature Extraction) ---")
    for zrel in ZIPS:
        zname = zrel.split("/")[-1]
        zpath = RAW_DIR / zname
        download(f"{BASE_DL_URL}/{zrel}", zpath)
        
        with zipfile.ZipFile(zpath, 'r') as z:
            names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for n in tqdm(names, desc=f"Processing {zname}"):
                # Standard V3Det mapping
                # The file_name in JSON matches zip path
                if n not in img_map: continue
                
                iid = img_map[n]['id']
                labels = list(img_to_cats.get(iid, []))
                if not labels: continue
                
                try:
                    with z.open(n) as f:
                        img_pil = Image.open(BytesIO(f.read())).convert("RGB")
                    
                    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feats = model_clip.encode_image(img_tensor).cpu().numpy().astype(np.float32)
                    
                    # We store the memory-efficient record
                    # Each image can have multiple descriptions if needed, 
                    # but we'll use the category labels as target text.
                    text_query = " ".join(labels)
                    dataset_records.append({"feat": feats[0], "text": text_query, "id": iid})
                except:
                    continue
        
        # Cleanup zip to save 2.5GB per step
        os.remove(zpath)

    # C. Build Vocab
    print("\n--- Phase 3: Building Vocabulary ---")
    word2idx = {"<pad>": 0, "<unk>": 1}
    for rec in dataset_records:
        for w in rec['text'].lower().split():
            if w not in word2idx: word2idx[w] = len(word2idx)
    
    # D. Training
    print("\n--- Phase 4: Training Contrastive Model ---")
    # Prepare DataLoader
    class V3DetDataset(Dataset):
        def __init__(self, records, w2i):
            self.records = records
            self.w2i = w2i
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            rec = self.records[i]
            tokens = tokenize(rec['text'], self.w2i)
            # Pad to 20
            tokens = (tokens + [0]*20)[:20]
            return torch.tensor(tokens), torch.tensor(rec['feat'])

    loader = DataLoader(V3DetDataset(dataset_records, word2idx), batch_size=BATCH_SIZE, shuffle=True)
    
    encoder = TextEncoder(len(word2idx)).to(DEVICE)
    optimizer = optim.Adam(encoder.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for texts, vis in tqdm(loader, desc=f"Epoch {epoch+1}"):
            texts, vis = texts.to(DEVICE), vis.to(DEVICE)
            
            # Forward
            txt_feats = encoder(texts)
            # vis is already normalized from CLIP? Let's ensure.
            vis = vis / vis.norm(dim=-1, keepdim=True)
            
            # Contrastive Loss (InfoNCE)
            logits = (txt_feats @ vis.T) / TEMPERATURE
            labels = torch.arange(logits.size(0)).to(DEVICE)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Loss: {total_loss / len(loader):.4f}")

    # E. Export CSV
    print("\n--- Phase 5: Exporting Result CSV ---")
    with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v3det_id", "categories", "description", "visual_embedding_sample"])
        for rec in dataset_records:
            # We only save a few dimensions of the embedding in CSV for preview
            feat_str = ",".join([f"{v:.4f}" for v in rec['feat'][:5]]) + "..."
            writer.writerow([rec['id'], rec['text'], f"Object detection categories: {rec['text']}", feat_str])
    
    # Save Weights
    torch.save(encoder.state_dict(), WEIGHTS_PATH)
    print(f"Training Complete! Result CSV: {RESULT_CSV}")
    print(f"Weights saved to: {WEIGHTS_PATH}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_pipeline()
    else:
        print("Usage: python train_v3det_standalone.py --run")
        print("This script will download ~52GB of V3Det data and train a model.")
