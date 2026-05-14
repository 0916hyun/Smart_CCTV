# models/inference_onnx.py — ONNX Export + ONNX Runtime 추론
# 방산 Edge 배포 (Jetson Orin Nano) 검증 동일 사상 적용
# PyTorch 추론 결과와 ONNX 추론 결과의 일치 여부 검증 포함

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import MODEL_PT_PATH, MODEL_ONNX_PATH, TRAIN_SEQ_LEN, TRAIN_FRAME_SIZE
from models.anomaly_ae import build_model


# ─────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────
def export_onnx(pt_path: str = MODEL_PT_PATH,
                onnx_path: str = MODEL_ONNX_PATH,
                opset: int = 17) -> None:
    """
    학습된 ConvAE (.pt) → ONNX 모델로 변환.
    Jetson Orin Nano / TensorRT 배포를 위한 ONNX opset 17 사용.
    """
    print(f"[ONNX Export] {pt_path} → {onnx_path}")
    ckpt = torch.load(pt_path, map_location="cpu")
    seq_len    = ckpt.get("seq_len", TRAIN_SEQ_LEN)
    frame_size = ckpt.get("frame_size", TRAIN_FRAME_SIZE)  # (H, W)
    use_rgb    = ckpt.get("use_rgb", False)

    model = build_model(seq_len=seq_len, use_rgb=use_rgb)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 더미 입력: (1, T, H, W) grayscale
    H, W = frame_size
    dummy = torch.randn(1, seq_len, H, W)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_seq"],
        output_names=["recon", "latent"],
        dynamic_axes={
            "input_seq": {0: "batch_size"},
            "recon":     {0: "batch_size"},
            "latent":    {0: "batch_size"},
        },
    )
    print(f"  ✓ ONNX 저장 완료: {onnx_path}")

    # 저장된 ONNX 그래프 검증
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("  ✓ ONNX 그래프 검증 통과")
    except ImportError:
        print("  ⚠ onnx 패키지 미설치 — 검증 생략 (pip install onnx)")


# ─────────────────────────────────────────────
# ONNX Runtime 추론 클래스
# ─────────────────────────────────────────────
class ONNXInference:
    """
    ONNX Runtime 기반 ConvAE 추론.
    PyTorch 없이 동작 — Jetson / TensorRT 환경에서 사용.

    사용법:
        inf = ONNXInference("models/checkpoints/convae.onnx")
        score = inf.anomaly_score(frame_seq_np)  # (T, H, W) float32 [0,1]
    """
    def __init__(self, onnx_path: str = MODEL_ONNX_PATH):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime  # CPU\n"
                              "pip install onnxruntime-gpu  # GPU")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        ep = self.session.get_providers()[0]
        print(f"[ONNXInference] 로드 완료: {onnx_path}  (EP: {ep})")

    def infer(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            seq: (T, H, W) 또는 (1, T, H, W) float32 [0, 1]
        Returns:
            recon:  (1, T, H, W) float32
            latent: (1, 256, H/16, W/16) float32
        """
        if seq.ndim == 3:
            seq = seq[np.newaxis]  # (1, T, H, W)
        recon, latent = self.session.run(None, {self.input_name: seq})
        return recon, latent

    def anomaly_score(self, seq: np.ndarray) -> float:
        """
        Args:
            seq: (T, H, W) float32 [0, 1]
        Returns:
            float — MSE reconstruction error (이상 점수)
        """
        recon, _ = self.infer(seq)
        if seq.ndim == 3:
            seq_in = seq[np.newaxis]
        else:
            seq_in = seq
        score = float(np.mean((recon - seq_in) ** 2))
        return score


# ─────────────────────────────────────────────
# PyTorch vs ONNX 결과 일치 검증
# ─────────────────────────────────────────────
def validate_onnx_vs_pytorch(pt_path: str = MODEL_PT_PATH,
                               onnx_path: str = MODEL_ONNX_PATH,
                               rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """
    PyTorch 추론 결과와 ONNX Runtime 추론 결과가 일치하는지 검증.
    배포 전 필수 수행 — 방산 SW 검증 추적성 요구사항 대응.
    """
    print("[Validate] PyTorch vs ONNX Runtime 결과 비교 시작")

    # PyTorch 추론
    ckpt = torch.load(pt_path, map_location="cpu")
    model_pt = build_model(ckpt["seq_len"], ckpt["use_rgb"])
    model_pt.load_state_dict(ckpt["model_state_dict"])
    model_pt.eval()
    H, W = ckpt["frame_size"]
    T = ckpt["seq_len"]
    dummy = torch.randn(1, T, H, W)
    with torch.no_grad():
        recon_pt, latent_pt = model_pt(dummy)
    recon_pt_np  = recon_pt.numpy()
    latent_pt_np = latent_pt.numpy()

    # ONNX Runtime 추론
    ort_inf = ONNXInference(onnx_path)
    recon_onnx, latent_onnx = ort_inf.infer(dummy.numpy())

    # 비교
    recon_match  = np.allclose(recon_pt_np,  recon_onnx,  rtol=rtol, atol=atol)
    latent_match = np.allclose(latent_pt_np, latent_onnx, rtol=rtol, atol=atol)

    print(f"  recon 일치:  {'✓' if recon_match  else '✗'}  "
          f"(max diff: {np.max(np.abs(recon_pt_np - recon_onnx)):.2e})")
    print(f"  latent 일치: {'✓' if latent_match else '✗'}  "
          f"(max diff: {np.max(np.abs(latent_pt_np - latent_onnx)):.2e})")

    success = recon_match and latent_match
    print(f"  최종 결과: {'✅ PASS' if success else '❌ FAIL'}")
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ONNX Export / Validate")
    parser.add_argument("--mode", choices=["export", "validate", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("export", "both"):
        export_onnx()
    if args.mode in ("validate", "both"):
        validate_onnx_vs_pytorch()
