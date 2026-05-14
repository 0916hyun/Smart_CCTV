# models/anomaly_ae.py — Convolutional Autoencoder (ConvAE) for Anomaly Detection
# 정상 프레임 시퀀스만으로 학습 → reconstruction error로 이상 점수 산출
# ONNX export 가능하도록 설계 (no dynamic control flow in forward)

import torch
import torch.nn as nn
from config import TRAIN_FRAME_SIZE, TRAIN_SEQ_LEN


class ConvEncoder(nn.Module):
    """
    시간축(T)을 채널로 취급하는 2D ConvEncoder.
    입력: (B, T*C, H, W) — T개 프레임을 채널 방향으로 스택
    출력: (B, 256, H/16, W/16) 잠재 벡터
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: (B, in_ch, 128, 128) → (B, 32, 64, 64)
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: → (B, 64, 32, 32)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: → (B, 128, 16, 16)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: → (B, 256, 8, 8)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """
    ConvEncoder의 역방향 — 잠재 벡터 → 원본 해상도 복원
    입력: (B, 256, H/16, W/16)
    출력: (B, in_channels, H, W)
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: (B, 256, 8, 8) → (B, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: → (B, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: → (B, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 4: → (B, out_channels, 128, 128)
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # 픽셀 값 0~1 정규화
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvAE(nn.Module):
    """
    Convolutional Autoencoder for Video Anomaly Detection.

    학습: 정상 영상만으로 reconstruction 최소화 (MSE loss)
    추론: reconstruction error = 이상 점수
          정상 → error 낮음, 이상 → error 높음

    파라미터 수: ~5M (Edge 배포 가능)

    입력 형식:
        x: (B, T, H, W) grayscale 시퀀스
           or (B, T, C, H, W) RGB → 내부적으로 변환

    방산 적용:
        - ONNX export로 Jetson Orin Nano 배포 가능
        - 외부 API 의존 없는 폐쇄망 운용
        - reconstruction error = 설명 가능한 이상 점수
    """
    def __init__(self, seq_len: int = TRAIN_SEQ_LEN, use_rgb: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.use_rgb = use_rgb
        ch_per_frame = 3 if use_rgb else 1
        in_channels = seq_len * ch_per_frame

        self.encoder = ConvEncoder(in_channels)
        self.decoder = ConvDecoder(in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, H, W) 또는 (B, T*ch, H, W) — 정규화된 [0,1] 텐서

        Returns:
            recon:  (B, T*ch, H, W) — 복원된 시퀀스
            z:      (B, 256, H/16, W/16) — 잠재 벡터
        """
        # 입력 shape 정리: (B, T, H, W) → (B, T, H, W) 유지 후 flatten
        if x.dim() == 4 and not self.use_rgb:
            # (B, T, H, W) → (B, T*1, H, W)
            B, T, H, W = x.shape
            x_flat = x.view(B, T, H, W)
        elif x.dim() == 5:
            # (B, T, C, H, W) → (B, T*C, H, W)
            B, T, C, H, W = x.shape
            x_flat = x.view(B, T * C, H, W)
        else:
            x_flat = x  # 이미 (B, T*ch, H, W)

        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> float:
        """
        단일 시퀀스의 이상 점수 산출.
        Args:
            x: (1, T, H, W) — 배치 크기 1, 정규화된 텐서
        Returns:
            float — MSE reconstruction error (이상 점수)
        """
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            # 입력 shape 맞추기
            if x.dim() == 4:
                x_flat = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            else:
                x_flat = x
            score = torch.mean((recon - x_flat) ** 2).item()
        return score


def build_model(seq_len: int = TRAIN_SEQ_LEN, use_rgb: bool = False) -> ConvAE:
    """모델 인스턴스 생성 팩터리"""
    return ConvAE(seq_len=seq_len, use_rgb=use_rgb)


if __name__ == "__main__":
    # 모델 구조 및 파라미터 수 확인
    model = build_model(seq_len=16, use_rgb=False)
    dummy = torch.randn(2, 16, 128, 128)  # (B, T, H, W) grayscale
    recon, z = model(dummy)
    score = model.anomaly_score(dummy[:1])
    print(f"입력 shape:   {dummy.shape}")
    print(f"잠재 벡터:    {z.shape}")
    print(f"복원 shape:   {recon.shape}")
    print(f"이상 점수:    {score:.6f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터:  {total_params / 1e6:.2f}M")
