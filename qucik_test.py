# quick_test.py — 각 클래스 여러 씬 score 평균으로 threshold 후보 산출
# 사용법: python quick_test.py

import sys
import glob
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from models.anomaly_ae import build_model

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
MODEL_PATH  = "models/checkpoints/convae_best.pt"
DATA_ROOT   = "D:/CCTV/test"
SEQ_LEN     = 16
FRAME_SIZE  = (128, 128)
N_SCENES    = 3   # 클래스당 테스트할 씬 수

# ─────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────
ckpt  = torch.load(MODEL_PATH, map_location="cpu")
model = build_model(ckpt["seq_len"], ckpt["use_rgb"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"모델 로드 완료 (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})\n")


def get_scenes(class_dir: Path) -> dict:
    pngs = sorted(class_dir.glob("*.png"))
    groups = defaultdict(list)
    for p in pngs:
        stem  = p.stem
        parts = stem.rsplit("_", 1)
        scene_id  = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        frame_num = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        groups[scene_id].append((frame_num, str(p)))
    return {k: [p for _, p in sorted(v)] for k, v in sorted(groups.items())}


def get_score(frame_paths: list) -> float | None:
    if len(frame_paths) < SEQ_LEN:
        return None
    frames = []
    for fp in frame_paths[:SEQ_LEN]:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (FRAME_SIZE[1], FRAME_SIZE[0]))
        frames.append(img)
    seq   = np.stack(frames, axis=0).astype(np.float32) / 255.0
    x     = torch.from_numpy(seq).unsqueeze(0)
    return model.anomaly_score(x)


# ─────────────────────────────────────────────
# 클래스별 score 계산
# ─────────────────────────────────────────────
class_dirs   = sorted([d for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
all_normal   = []
all_anomaly  = []
results      = {}

for d in class_dirs:
    scenes = get_scenes(d)
    scene_list = list(scenes.items())[:N_SCENES]
    scores = []
    for scene_id, paths in scene_list:
        s = get_score(paths)
        if s is not None:
            scores.append(s)

    if not scores:
        continue

    results[d.name] = scores
    if d.name == "NormalVideos":
        all_normal.extend(scores)
    else:
        all_anomaly.extend(scores)

# ─────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────
print(f"{'클래스':<25} {'씬1':>10} {'씬2':>10} {'씬3':>10} {'평균':>10}")
print("─" * 70)

normal_mean = np.mean(all_normal) if all_normal else 0

for name, scores in results.items():
    vals     = [f"{s:.6f}" for s in scores]
    while len(vals) < 3:
        vals.append("      -")
    mean     = np.mean(scores)
    is_normal = name == "NormalVideos"
    ratio    = mean / normal_mean if normal_mean > 0 and not is_normal else None
    judge    = "── 기준" if is_normal else (
               "🔴 이상" if ratio and ratio > 2.0 else
               "🟡 경계" if ratio and ratio > 1.3 else "🟢 정상수준")
    print(f"{name:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {mean:>10.6f}  {judge}")

# ─────────────────────────────────────────────
# Threshold 추천
# ─────────────────────────────────────────────
if all_normal and all_anomaly:
    normal_max  = np.max(all_normal)
    anomaly_min = np.min(all_anomaly)
    mid         = (normal_max + anomaly_min) / 2

    print(f"\n===== Threshold 추천 =====")
    print(f"  Normal  최대값: {normal_max:.6f}")
    print(f"  Anomaly 최소값: {anomaly_min:.6f}")
    print(f"  중간값:         {mid:.6f}")
    print(f"\n  추천 설정 (config.py):")
    print(f"  THRESHOLD_WARNING  = {normal_max * 1.5:.6f}")
    print(f"  THRESHOLD_CRITICAL = {mid:.6f}")