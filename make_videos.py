# make_videos.py — PNG 시퀀스 → 클래스별 mp4 영상 변환
# 사용법: python make_videos.py

import os
import subprocess
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
FFMPEG     = r"C:\Users\8138\Downloads\Compressed\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
TEST_ROOT  = "D:/CCTV/test"
OUTPUT_DIR = "D:/CCTV/demo_videos"
FPS        = 10
SCALE      = 512
MAX_SCENES = 6   # 클래스당 최대 장면 수 (None이면 전부)


# ─────────────────────────────────────────────
# 장면별 PNG 그룹핑 (프레임 번호 숫자 정렬)
# ─────────────────────────────────────────────
def get_scenes(class_dir: Path) -> dict:
    pngs = sorted(class_dir.glob("*.png"))
    scenes = defaultdict(list)
    for p in pngs:
        stem  = p.stem
        parts = stem.rsplit("_", 1)
        scene_id  = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        frame_num = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        scenes[scene_id].append((frame_num, p))
    # 프레임 번호 숫자 기준 정렬
    return {k: [p for _, p in sorted(v)] for k, v in sorted(scenes.items())}


# ─────────────────────────────────────────────
# ffmpeg으로 영상 생성
# ─────────────────────────────────────────────
def make_video(frame_paths: list, output_path: str) -> bool:
    list_path = output_path.replace(".mp4", "_list.txt")
    with open(list_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
            f.write(f"duration {1/FPS:.4f}\n")

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vf", f"scale={SCALE}:{SCALE}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_path)
    return result.returncode == 0


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

class_dirs = sorted([d for d in Path(TEST_ROOT).iterdir() if d.is_dir()])
total = 0

for class_dir in class_dirs:
    class_name = class_dir.name
    scenes     = get_scenes(class_dir)

    if not scenes:
        print(f"[{class_name}] PNG 없음, 스킵")
        continue

    out_class_dir = Path(OUTPUT_DIR) / class_name
    out_class_dir.mkdir(exist_ok=True)

    scene_items = list(scenes.items())
    if MAX_SCENES:
        scene_items = scene_items[:MAX_SCENES]

    print(f"\n[{class_name}] {len(scenes)}개 장면 중 {len(scene_items)}개 변환")

    for scene_id, frame_paths in scene_items:
        out_path = str(out_class_dir / f"{scene_id}.mp4")
        success  = make_video(frame_paths, out_path)
        status   = "✓" if success else "✗"
        print(f"  {status} {scene_id} ({len(frame_paths)}프레임) → {out_path}")
        if success:
            total += 1

print(f"\n완료 — 총 {total}개 영상 생성 → {OUTPUT_DIR}")