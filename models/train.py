# models/train.py — ConvAE 학습 스크립트
# UCF-Crime PNG 이미지 폴더 방식 + 전처리 캐싱 (디스크 I/O 병목 제거)
# 파일명 규칙: {영상ID}_{프레임번호}.png (예: NormalVideos001_x264_0.png)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from models.anomaly_ae import build_model
from config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR,
    TRAIN_FRAME_SIZE, TRAIN_SEQ_LEN, TRAIN_DATA_DIR,
    MODEL_PT_PATH,
)


# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────
class NormalVideoDataset(Dataset):
    """
    UCF-Crime PNG 이미지 폴더 데이터셋 + 전처리 캐싱.

    폴더 구조:
        data_dir/
            NormalVideos001_x264_0.png
            NormalVideos001_x264_1.png
            ...

    캐싱:
        최초 1회 전체 PNG를 메모리에 올려둠.
        947,768장 × 64×64 grayscale ≈ 3.8GB RAM
        이후 __getitem__에서 디스크 접근 없음 → GPU 병목 해소.
    """
    def __init__(self, data_dir: str, seq_len: int = 16,
                 frame_size: tuple = (128, 128), stride: int = 8):
        self.seq_len    = seq_len
        self.frame_size = frame_size  # (H, W)

        # ── 1) PNG 수집 + 영상ID별 그룹핑 ──
        all_pngs = sorted(Path(data_dir).glob("*.png"))
        if not all_pngs:
            raise RuntimeError(f"PNG 파일을 찾을 수 없습니다: {data_dir}")

        groups = defaultdict(list)
        for p in all_pngs:
            stem  = p.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                video_id  = parts[0]
                frame_idx = int(parts[1])
            else:
                video_id  = stem
                frame_idx = 0
            groups[video_id].append((frame_idx, str(p)))

        # ── 2) 프레임 정렬 + 클립 생성 ──
        self.clips = []
        for video_id, frame_list in groups.items():
            frame_list.sort(key=lambda x: x[0])
            paths = [p for _, p in frame_list]
            n = len(paths)
            for start in range(0, n - seq_len + 1, stride):
                self.clips.append(paths[start: start + seq_len])

        if not self.clips:
            raise RuntimeError(
                f"클립 구성 실패. seq_len={seq_len}보다 긴 영상이 없습니다."
            )
        print(f"[Dataset] {len(groups):,}개 영상 → {len(self.clips):,}개 클립 준비 완료")

        # ── 3) 전처리 캐싱 (최초 1회) ──
        all_paths = sorted(set(p for clip in self.clips for p in clip))
        est_gb    = len(all_paths) * frame_size[0] * frame_size[1] / 1024**3
        print(f"[Dataset] 이미지 캐싱 시작 — {len(all_paths):,}장 (예상 RAM: {est_gb:.1f}GB)")

        self.cache = {}
        for fp in tqdm(all_paths, desc="캐싱", ncols=80):
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros(frame_size, dtype=np.uint8)
            if img.shape != (frame_size[0], frame_size[1]):
                img = cv2.resize(img, (frame_size[1], frame_size[0]))
            self.cache[fp] = img  # uint8 저장 (RAM 절약)

        print(f"[Dataset] 캐싱 완료 — {len(self.cache):,}장 로드됨")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_paths = self.clips[idx]
        # 캐시에서 바로 읽기 — 디스크 접근 없음
        frames = [self.cache[fp] for fp in frame_paths]
        seq = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(seq)


# ─────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────
def train(data_dir: str = TRAIN_DATA_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 디바이스: {device}")

    dataset = NormalVideoDataset(data_dir, seq_len=TRAIN_SEQ_LEN,
                                  frame_size=TRAIN_FRAME_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,           # 캐싱 후엔 낮아도 충분
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
    )

    model     = build_model(seq_len=TRAIN_SEQ_LEN, use_rgb=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    history   = {"loss": []}

    os.makedirs(os.path.dirname(MODEL_PT_PATH), exist_ok=True)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch:02d}/{TRAIN_EPOCHS}", ncols=80):
            x = batch.to(device)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        history["loss"].append(avg_loss)
        print(f"[Epoch {epoch:02d}] Loss: {avg_loss:.6f}  LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "loss":             best_loss,
                "seq_len":          TRAIN_SEQ_LEN,
                "frame_size":       TRAIN_FRAME_SIZE,
                "use_rgb":          False,
            }, MODEL_PT_PATH)
            print(f"  ✓ Best model saved → {MODEL_PT_PATH}")

    # 학습 곡선 저장
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Train Loss (MSE)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ConvAE Training — UCF-Crime")
    plt.legend(); plt.tight_layout()
    plt.savefig("models/checkpoints/training_curve.png")
    plt.close()
    print("[Train] 완료 → models/checkpoints/training_curve.png")

    with open("models/checkpoints/train_meta.json", "w") as f:
        json.dump({
            "best_loss":  best_loss,
            "epochs":     TRAIN_EPOCHS,
            "seq_len":    TRAIN_SEQ_LEN,
            "frame_size": list(TRAIN_FRAME_SIZE),
            "data_dir":   data_dir,
            "dataset":    "UCF-Crime NormalVideos",
        }, f, indent=2)

    return model


# ─────────────────────────────────────────────
# 임계값 분석
# ─────────────────────────────────────────────
def compute_threshold_candidates(model_path: str = MODEL_PT_PATH,
                                   data_dir: str = TRAIN_DATA_DIR):
    """
    학습 완료 후 정상 데이터 reconstruction error 분포 분석.
    P90 → THRESHOLD_WARNING, P99 → THRESHOLD_CRITICAL 후보.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)
    model  = build_model(ckpt["seq_len"], ckpt["use_rgb"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = NormalVideoDataset(data_dir, seq_len=ckpt["seq_len"],
                                  frame_size=tuple(ckpt["frame_size"]), stride=16)
    indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
    scores  = []
    with torch.no_grad():
        for i in tqdm(indices, desc="Scoring normal clips", ncols=80):
            x     = dataset[i].unsqueeze(0).to(device)
            score = model.anomaly_score(x)
            scores.append(score)

    scores = np.array(scores)
    print("\n===== 정상 데이터 이상 점수 분포 =====")
    print(f"  Mean : {scores.mean():.6f}")
    print(f"  Std  : {scores.std():.6f}")
    print(f"  P90  : {np.percentile(scores, 90):.6f}  ← THRESHOLD_WARNING 후보")
    print(f"  P99  : {np.percentile(scores, 99):.6f}  ← THRESHOLD_CRITICAL 후보")
    print("config.py의 THRESHOLD_WARNING / THRESHOLD_CRITICAL을 위 값으로 업데이트하세요.\n")

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=50, edgecolor="black", alpha=0.8, label="Normal clip scores")
    plt.axvline(np.percentile(scores, 90), color="orange", linestyle="--", label="P90 (Warning)")
    plt.axvline(np.percentile(scores, 99), color="red",    linestyle="--", label="P99 (Critical)")
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count")
    plt.title("Normal Score Distribution — ConvAE (UCF-Crime)"); plt.legend()
    plt.tight_layout()
    plt.savefig("models/checkpoints/score_distribution.png")
    plt.close()
    print("분포 그래프 → models/checkpoints/score_distribution.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConvAE 학습 / 임계값 분석")
    parser.add_argument("--mode",     choices=["train", "threshold"], default="train")
    parser.add_argument("--data_dir", default=TRAIN_DATA_DIR)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir)
    elif args.mode == "threshold":
        compute_threshold_candidates(data_dir=args.data_dir)
