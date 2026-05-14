# utils.py — 유틸리티 함수
# 기존 v1.0 기능 유지 + 방산 감사 로그(audit_log) 추가
# SQL Injection 방지: 파라미터화 쿼리로 전면 교체

import base64
import os
import sqlite3
import smtplib
import cv2
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image

from config import DB_PATH, AUDIT_LOG_ENABLED, FRAME_SAVE_DIR


# ─────────────────────────────────────────────
# 이미지 유틸리티
# ─────────────────────────────────────────────
def encode_image(image: Image.Image) -> str:
    """PIL Image → base64 JPEG 문자열 (GPT API용)"""
    w, h = image.size
    scale = min(512 / max(w, h), 1.0)
    resized = image.resize((int(w * scale), int(h * scale))).convert("RGB")
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """BGR numpy → PIL RGB Image"""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def save_frame(frame_bgr: np.ndarray, prefix: str = "event") -> str:
    """
    이벤트 발생 시 검증용 프레임을 디스크에 저장.
    감사 로그의 frame_path 필드에 저장 경로 기록.
    """
    os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    path = os.path.join(FRAME_SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path


# ─────────────────────────────────────────────
# 데이터베이스 초기화
# ─────────────────────────────────────────────
def init_db() -> None:
    """DB 초기화: 기존 events 테이블 + 방산 감사 로그 테이블"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 기존 이벤트 테이블 (v1.0 호환)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            place       VARCHAR(10),
            description VARCHAR(255)
        )
    """)

    # 방산 감사 로그 (v2.0 신규)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         DATETIME,
            mode              VARCHAR(20),
            model_version     VARCHAR(50),
            event_type        VARCHAR(30),
            severity          VARCHAR(10),
            confidence        REAL,
            raw_anomaly_score REAL,
            description       VARCHAR(255),
            frame_path        VARCHAR(255)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] 초기화 완료: {DB_PATH}")


# ─────────────────────────────────────────────
# 기존 이벤트 로그 (v1.0 호환)
# ─────────────────────────────────────────────
def save_sqlite3(place: str, description: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO events(place, description) VALUES (?, ?)",
                (place, description))
    conn.commit()
    conn.close()


def get_db_msg(place: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT description FROM events WHERE place=? ORDER BY id DESC LIMIT 1", (place,))
    rows = cur.fetchall()
    conn.close()
    return rows[0][0] if rows else ""


# ─────────────────────────────────────────────
# 방산 감사 로그 (v2.0 신규)
# ─────────────────────────────────────────────
def save_audit_log(result, frame_bgr: np.ndarray | None = None) -> None:
    """
    탐지 결과를 감사 로그에 저장.
    방산 SW 비기능 요구사항: 이벤트 추적성 확보

    Args:
        result: EventResult 인스턴스
        frame_bgr: 저장할 이벤트 프레임 (None이면 저장 생략)
    """
    if not AUDIT_LOG_ENABLED:
        return

    from config import MODEL_VERSION
    frame_path = None
    if frame_bgr is not None and result.severity.value != "NORMAL":
        frame_path = save_frame(frame_bgr, prefix=result.event_type.value)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO audit_log
            (timestamp, mode, model_version, event_type, severity,
             confidence, raw_anomaly_score, description, frame_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        result.mode,
        MODEL_VERSION,
        result.event_type.value,
        result.severity.value,
        result.confidence,
        result.raw_anomaly_score,
        result.description,
        frame_path,
    ))
    conn.commit()
    conn.close()


def get_audit_log(limit: int = 20) -> list[dict]:
    """최신 감사 로그 조회 (Gradio 감사 탭용)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, mode, event_type, severity, confidence,
               raw_anomaly_score, description
        FROM audit_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_audit_log_str(limit: int = 10) -> str:
    """감사 로그 → Gradio Textbox 표시용 문자열"""
    rows = get_audit_log(limit)
    if not rows:
        return "감사 로그 없음"
    lines = []
    for r in rows:
        ts = r["timestamp"][:19]
        lines.append(
            f"[{ts}] [{r['mode'].upper():10}] "
            f"{r['severity']:8} | {r['event_type']:20} | "
            f"conf={r['confidence']:.2f} | {r['description']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 이메일 알람 (v1.0 호환)
# ─────────────────────────────────────────────
def send_email(subject: str, content: str) -> None:
    """경보 이메일 발송. api_key.py의 email_id / email_pwd 필요."""
    try:
        from api_key import email_id, email_pwd
        from email.mime.text import MIMEText
        msg = MIMEText(content)
        msg["Subject"] = subject
        msg["From"] = f"{email_id}@gmail.com"
        msg["To"]   = f"{email_id}@gmail.com"
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls()
        s.login(email_id, email_pwd)
        s.sendmail(f"{email_id}@gmail.com", f"{email_id}@gmail.com", msg.as_string())
        s.quit()
    except Exception as e:
        print(f"[Email] 발송 실패: {e}")
