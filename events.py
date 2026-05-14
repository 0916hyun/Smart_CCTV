# events.py — 방산 감시정찰 이벤트 Taxonomy
# 한화시스템 DE사업부 운용 시나리오 기반 설계
# 참조: TAS-815K TOD(지상·해안 감시), KF-21 EO TGP 운용 시나리오

from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 심각도 3단계 분류
# ─────────────────────────────────────────────
class Severity(str, Enum):
    NORMAL   = "NORMAL"    # 정상 — 모니터링 지속
    WARNING  = "WARNING"   # 주의 — 운용자 알림
    CRITICAL = "CRITICAL"  # 경보 — 즉각 대응 필요


# ─────────────────────────────────────────────
# 방산 감시정찰 이벤트 타입
# ─────────────────────────────────────────────
class EventType(str, Enum):
    NORMAL             = "NORMAL"             # 이상 없음
    INTRUSION          = "INTRUSION"          # 무단 인원 침입 → 기지 경계·해안 감시
    LOITERING          = "LOITERING"          # 비정상 체류·배회 → 보안 시설 주변 감시
    VEHICLE_INTRUSION  = "VEHICLE_INTRUSION"  # 차량 침입 → 출입통제 구역
    ABNORMAL_BEHAVIOR  = "ABNORMAL_BEHAVIOR"  # 폭력·기물파손·이상행동 → 군중 감시
    NIGHT_MOVEMENT     = "NIGHT_MOVEMENT"     # 야간 이동체 탐지 → 야간 감시정찰
    OBJECT_LEFT        = "OBJECT_LEFT"        # 의심 물체 방치 → 폭발물·위험물 감시


# 이벤트별 기본 심각도 매핑
EVENT_DEFAULT_SEVERITY = {
    EventType.NORMAL:            Severity.NORMAL,
    EventType.INTRUSION:         Severity.CRITICAL,
    EventType.LOITERING:         Severity.WARNING,
    EventType.VEHICLE_INTRUSION: Severity.CRITICAL,
    EventType.ABNORMAL_BEHAVIOR: Severity.WARNING,
    EventType.NIGHT_MOVEMENT:    Severity.WARNING,
    EventType.OBJECT_LEFT:       Severity.CRITICAL,
}

# 이벤트별 한국어 설명 (UI 표시용)
EVENT_LABELS_KO = {
    EventType.NORMAL:            "정상",
    EventType.INTRUSION:         "무단 침입 감지",
    EventType.LOITERING:         "비정상 배회 감지",
    EventType.VEHICLE_INTRUSION: "차량 침입 감지",
    EventType.ABNORMAL_BEHAVIOR: "이상 행동 감지",
    EventType.NIGHT_MOVEMENT:    "야간 이동체 감지",
    EventType.OBJECT_LEFT:       "의심 물체 감지",
}

# 심각도별 UI 색상
SEVERITY_COLORS = {
    Severity.NORMAL:   "#2ecc71",  # 녹색
    Severity.WARNING:  "#f39c12",  # 노랑
    Severity.CRITICAL: "#e74c3c",  # 빨강
}


# ─────────────────────────────────────────────
# 탐지 결과 표준 스키마
# (GPT 엔진 / Self-model 엔진 모두 이 형식으로 반환)
# ─────────────────────────────────────────────
@dataclass
class EventResult:
    event_type:        EventType
    severity:          Severity
    confidence:        float            # 0.0 ~ 1.0
    description:       str             # 운용자용 상황 설명
    raw_anomaly_score: Optional[float] = None   # ConvAE reconstruction error (self_model 모드)
    mode:              str = "unknown"          # "gpt" | "self_model"

    def to_dict(self) -> dict:
        return {
            "event_type":        self.event_type.value,
            "severity":          self.severity.value,
            "confidence":        round(self.confidence, 4),
            "description":       self.description,
            "raw_anomaly_score": self.raw_anomaly_score,
            "mode":              self.mode,
        }

    def label_ko(self) -> str:
        return EVENT_LABELS_KO.get(self.event_type, self.event_type.value)

    def severity_color(self) -> str:
        return SEVERITY_COLORS.get(self.severity, "#ffffff")


def severity_from_score(score: float) -> Severity:
    """ConvAE reconstruction error → 3단계 심각도 변환 (config 임계값 사용)"""
    from config import THRESHOLD_WARNING, THRESHOLD_CRITICAL
    if score >= THRESHOLD_CRITICAL:
        return Severity.CRITICAL
    elif score >= THRESHOLD_WARNING:
        return Severity.WARNING
    else:
        return Severity.NORMAL


def confidence_from_score(score: float) -> float:
    """이상 점수를 0~1 confidence로 정규화 (sigmoid 근사)"""
    import math
    # score가 0에 가까울수록 confidence 낮음 (정상), 높을수록 높음 (이상)
    # THRESHOLD_CRITICAL 기준으로 0.5 중앙 정규화
    from config import THRESHOLD_CRITICAL
    x = score / (THRESHOLD_CRITICAL + 1e-8)
    return round(1 / (1 + math.exp(-5 * (x - 0.5))), 4)
