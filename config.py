# config.py — Smart CCTV v2.0 전역 설정
# 방산 폐쇄망 환경 가정: 모든 외부 의존성을 config에서 제어

# ─────────────────────────────────────────────
# 탐지 모드
# "hybrid"     : ConvAE 이상 감지 → GPT 상황 설명 (추천)
# "self_model" : ConvAE 단독 (폐쇄망)
# "gpt"        : GPT-4o 단독 (v1.0 방식)
# ─────────────────────────────────────────────
DETECTION_MODE = "hybrid"

# ─────────────────────────────────────────────
# 이상 점수 임계값 (ConvAE reconstruction error)
# 학습 후 threshold 분석 결과로 업데이트 필요
# ─────────────────────────────────────────────
THRESHOLD_WARNING  = 0.000390
THRESHOLD_CRITICAL = 0.001106

# ─────────────────────────────────────────────
# 모델 경로
# ─────────────────────────────────────────────
MODEL_PT_PATH   = "models/checkpoints/convae_best.pt"
MODEL_ONNX_PATH = "models/checkpoints/convae.onnx"

# ─────────────────────────────────────────────
# 입력 소스
# ─────────────────────────────────────────────
INPUT_SOURCE = "screen"          # "screen" | "rtsp"
RTSP_URL_PERIMETER = "rtsp://admin:password@192.168.1.100:554/stream1"
RTSP_URL_FACILITY  = "rtsp://admin:password@192.168.1.101:554/stream2"

# 화면캡처 좌표 (데모용)
SCREEN_REGION_PERIMETER = (224, 129, 478, 359)  # (x, y, w, h)
SCREEN_REGION_FACILITY  = (224, 489, 478, 359)

# ─────────────────────────────────────────────
# 감사 로그
# ─────────────────────────────────────────────
DB_PATH           = "database/db.db"
AUDIT_LOG_ENABLED = True
FRAME_SAVE_DIR    = "database/frames"

# ─────────────────────────────────────────────
# ConvAE 학습 하이퍼파라미터
# ─────────────────────────────────────────────
TRAIN_EPOCHS     = 20
TRAIN_BATCH_SIZE = 256          # RTX 4060 Ti 16GB 기준
TRAIN_LR         = 1e-3
TRAIN_FRAME_SIZE = (128, 128)   # H×W
TRAIN_SEQ_LEN    = 16           # 클립당 프레임 수
TRAIN_DATA_DIR   = "D:/CCTV/train/NormalVideos"

# 모델 버전 태깅 (감사 로그용)
MODEL_VERSION = "ConvAE-v1.0-UCF-Crime"
