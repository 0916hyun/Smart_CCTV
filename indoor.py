import cv2, asyncio, base64, time
from collections import deque
from io import BytesIO
from datetime import datetime
from PIL import Image
from events import EventResult, EventType, Severity
from utils import save_sqlite3, save_audit_log, send_email
import config

LLM_EVERY      = 48
HISTORY_EVERY  = 12
LLM_TIMEOUT    = 20.0
DEESC_FRAMES   = 8
CRITICAL_HOLD  = 15.0

_PHASE1_DESC = {
    Severity.WARNING:  "이상 패턴 감지 — LLM 분류 중",
    Severity.CRITICAL: "위협 감지 — 긴급 LLM 분류 중",
}


def _frame_to_html(frame_bgr) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > 640:
        rgb = cv2.resize(rgb, (640, int(h * 640 / w)))
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        "<div style='font-size:.52rem;color:#1a4a6a;letter-spacing:.14em;"
        "padding:3px 8px;border-bottom:1px solid #0a1a2a;background:#010d17'>⬡ LATEST FRAME</div>"
        "<div style='background:#010d17;padding:0;margin:0;line-height:0;'>"
        f"<img src='data:image/jpeg;base64,{b64}' "
        "style='width:100%;height:auto;display:block;margin:0;padding:0;'/>"
        "</div>"
    )


def _fmt(r: EventResult, phase1: bool = False) -> str:
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = {"NORMAL": "●", "WARNING": "◆", "CRITICAL": "▲"}.get(r.severity.value, "○")
    tag  = " [LLM 분류중...]" if phase1 else ""
    desc = _PHASE1_DESC.get(r.severity, r.description) if phase1 else r.description
    return f"[{ts}] {icon} {r.severity.value} | {r.event_type.value}{tag}\n  {desc}"


async def run_facility_watch(engine, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "<div style='color:#f44336;padding:8px'>[ ERR ] 영상을 열 수 없습니다</div>", "", False
        return

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / fps
    frame_idx   = 0

    is_hybrid = hasattr(engine, 'convae') and hasattr(engine, 'gpt') and engine.gpt is not None

    eff_sev        = Severity.NORMAL
    deesc_count    = 0
    critical_until = 0.0

    last_result = None
    last_log    = ""
    lm_task     = None
    lm_pending  = False
    frame_history = deque(maxlen=4)

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        log_changed = False

        if frame_idx % HISTORY_EVERY == 0:
            frame_history.append(frame.copy())

        ae_result = None
        if is_hybrid:
            try:
                ae_result = engine.convae.detect(frame)
            except Exception:
                pass
        elif not hasattr(engine, '_call'):
            try:
                ae_result = engine.detect(frame)
            except Exception:
                pass

        if ae_result and "버퍼" not in ae_result.description:
            raw_sev = ae_result.severity
            now     = time.monotonic()
            if now < critical_until:
                raw_sev = Severity.CRITICAL
            sev_order = {Severity.NORMAL: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
            if sev_order[raw_sev] > sev_order[eff_sev]:
                eff_sev     = raw_sev
                deesc_count = 0
                if raw_sev == Severity.CRITICAL:
                    critical_until = now + CRITICAL_HOLD
            elif sev_order[raw_sev] < sev_order[eff_sev]:
                deesc_count += 1
                if deesc_count >= DEESC_FRAMES:
                    eff_sev     = raw_sev
                    deesc_count = 0
            else:
                deesc_count = 0
            ae_result.severity = eff_sev

            if eff_sev != (last_result.severity if last_result else Severity.NORMAL):
                last_result = ae_result
                last_log    = _fmt(ae_result, phase1=is_hybrid and eff_sev != Severity.NORMAL)
                log_changed = True
                lm_pending  = is_hybrid and eff_sev != Severity.NORMAL
                if eff_sev != Severity.NORMAL:
                    save_audit_log(ae_result, frame_bgr=frame)
                    save_sqlite3("facility", ae_result.description)

        lm_just_completed = None
        if lm_task is not None and lm_task.done():
            try:
                gpt_result = lm_task.result()
                if is_hybrid and hasattr(engine, '_merge') and last_result:
                    merged = engine._merge(last_result, gpt_result)
                else:
                    merged = gpt_result
                lm_task     = None
                lm_pending  = False
                last_result = merged
                last_log    = _fmt(merged, phase1=False)
                log_changed = True
                lm_just_completed = merged
                if merged.severity == Severity.CRITICAL:
                    try:
                        send_email(
                            f"[긴급 경보] {merged.event_type.value} 감지",
                            f"위협: {merged.description}\n신뢰도: {merged.confidence:.0%}\n모드: {merged.mode}"
                        )
                        print("[EMAIL] CRITICAL 경보 이메일 발송 완료")
                    except Exception as e:
                        print(f"[EMAIL] 발송 실패: {e}")
            except asyncio.TimeoutError:
                lm_task    = None
                lm_pending = False
            except Exception as e:
                print(f"[ENGINE ERR] {type(e).__name__}: {e}")
                lm_task    = None
                lm_pending = False

        if is_hybrid and lm_task is None and eff_sev != Severity.NORMAL and frame_idx % LLM_EVERY == 0:
            frames = list(frame_history) if len(frame_history) >= 2 else [frame.copy()]
            lm_task = asyncio.create_task(
                asyncio.wait_for(engine.gpt.detect_async_montage(frames), timeout=LLM_TIMEOUT)
            )

        if hasattr(engine, '_call') and lm_task is None and frame_idx % LLM_EVERY == 0:
            lm_task = asyncio.create_task(
                asyncio.wait_for(engine.detect_async(frame.copy()), timeout=LLM_TIMEOUT)
            )

        ae_score = (ae_result.raw_anomaly_score
                    if ae_result and "버퍼" not in ae_result.description
                    else None)
        yield _frame_to_html(frame), last_log, log_changed, lm_pending, lm_just_completed, ae_score

        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(frame_delay - elapsed, 0))

    if lm_task is not None and not lm_task.done():
        lm_task.cancel()
    cap.release()