import os, asyncio, gradio as gr
from config import DETECTION_MODE as DEFAULT_MODE, DB_PATH, THRESHOLD_WARNING as WARN_TH, THRESHOLD_CRITICAL as CRIT_TH
from utils import init_db, get_audit_log_str
from events import Severity, EventResult, EventType

_FM = "font-family:'Share Tech Mono',monospace"
_FO = "font-family:'Orbitron',monospace"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

*, body, .gradio-container { box-sizing: border-box; }
body, .gradio-container {
    background: #03070a !important;
    color: #4fc3f7 !important;
    font-family: 'Share Tech Mono', monospace !important;
    margin: 0 !important; padding: 0 8px 8px !important;
}
.gradio-container { max-width: 100% !important; }

/* 헤더 */
.sys-hdr {
    background: linear-gradient(90deg,#001a2e 0%,#002a1e 100%);
    border-bottom: 2px solid #0a4a6a;
    padding: 10px 16px; margin-bottom: 6px;
    display: flex; align-items: center; gap: 20px;
}
.sys-hdr .title { font-family:'Orbitron',monospace; font-size:.9rem; color:#29b6f6; letter-spacing:.12em; }
.sys-hdr .sub   { font-size:.56rem; color:#1a5a7a; letter-spacing:.1em; margin-top:3px; }
.sys-hdr .dot   { width:8px; height:8px; border-radius:50%; background:#29b6f6;
                  box-shadow:0 0 8px #29b6f6; animation:blink 1.5s infinite; flex-shrink:0; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* 탭 */
.tab-nav button {
    background:none !important; border:none !important;
    border-bottom:2px solid transparent !important;
    color:#1a4a6a !important; font-family:'Share Tech Mono',monospace !important;
    font-size:.65rem !important; letter-spacing:.12em; padding:6px 14px !important;
}
.tab-nav button.selected { color:#29b6f6 !important; border-bottom-color:#29b6f6 !important; }
.tab-nav { border-bottom:1px solid #0a2a3a !important; margin-bottom:6px !important; }

/* ── 상단 업로드 영역 ── */
.top-upload {
    background:#010d17 !important; border:1px solid #0a4a6a !important;
    border-radius:4px !important; overflow:hidden; padding:0 !important;
    max-width:220px !important;
}
.top-upload label {
    font-size:.6rem !important; color:#29b6f6 !important; font-weight:bold !important;
    letter-spacing:.1em; padding:4px 8px; display:block;
    background:#001525 !important; border-bottom:1px solid #0a3a5a !important;
}
.top-upload video { display:none !important; }
.top-upload [data-testid="video"] > div,
.top-upload .wrap { min-height:44px !important; max-height:50px !important;
                    background:#010d17 !important; }
.top-upload span, .top-upload p { color:#29b6f6 !important; font-size:.55rem !important; }
.top-upload svg  { stroke:#29b6f6 !important; width:14px !important; height:14px !important; }

/* 모드 라디오 */
.mode-sel label       { font-size:.65rem !important; color:#1a5a7a !important; }
.mode-sel .label-wrap { font-size:.62rem !important; color:#0a3a5a !important; }

/* ── LATEST FRAME — elem_id로 고정 크기 ── */
#perimeter-frame, #facility-frame {
    width: 520px !important;
    max-width: 520px !important;
    background: #010d17;
    border: 1px solid #0a2a3a;
    border-radius: 4px;
    overflow: hidden;
}
#perimeter-frame > div, #facility-frame > div {
    width: 520px !important;
    max-width: 520px !important;
}
#perimeter-frame img, #facility-frame img {
    width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 버튼 */
.scan-btn {
    background:#001e35 !important; border:1px solid #0a4a6a !important;
    color:#29b6f6 !important; font-family:'Share Tech Mono',monospace !important;
    font-size:.68rem !important; letter-spacing:.15em;
    border-radius:2px !important; padding:9px !important; width:100%;
    transition:all .15s; margin-top:4px !important;
}
.scan-btn:hover { background:#29b6f6 !important; color:#010d17 !important; }

/* 감사로그 */
.audit-panel textarea {
    background:#010d17 !important; border:1px solid #0a2a3a !important;
    color:#1a7a9a !important; font-family:'Share Tech Mono',monospace !important;
    font-size:.65rem !important;
}

.zone { font-size:.54rem; color:#0a3a5a; letter-spacing:.1em;
        border-bottom:1px solid #0a2a3a; padding-bottom:4px; margin-bottom:6px; }
.main-row { gap:8px !important; align-items:flex-start !important; }
"""


def load_engine(mode):
    gpt = None
    if mode in ("hybrid", "gpt"):
        try:
            from api_key import openai_api_key
            from engines.gpt_engine import GPTEngine
            gpt = GPTEngine(api_key=openai_api_key)
            print("[Engine] GPT OK")
        except Exception as e:
            print(f"[Engine] GPT 초기화 실패: {e} → self_model")
            mode = "self_model"
    if mode == "hybrid":
        from engines.hybrid_engine import HybridEngine
        return HybridEngine(gpt_engine=gpt, use_onnx=True), "hybrid"
    if mode == "gpt":
        return gpt, "gpt"
    from engines.self_model_engine import SelfModelEngine
    return SelfModelEngine(use_onnx=True), "self_model"


_eng,  _mode = load_engine(DEFAULT_MODE)
_eng_f, _    = load_engine(DEFAULT_MODE)

_PLACEHOLDER = (
    "<div style='font-size:.52rem;color:#1a4a6a;letter-spacing:.14em;"
    "padding:3px 8px;border-bottom:1px solid #0a1a2a;background:#010d17'>⬡ LATEST FRAME</div>"
    "<div style='background:#010d17;width:100%;height:200px;"
    "display:flex;align-items:center;justify-content:center;'>"
    "<span style='color:#0a3a5a;font-family:\"Share Tech Mono\",monospace;"
    "font-size:.65rem;letter-spacing:.15em;'>AWAITING FEED</span></div>"
)


def _right_panel(sev: str, log: str) -> str:
    C = {
        "STANDBY": ("#1a5a7a", "#010d17", "STANDBY", "대기중"),
        "NORMAL":  ("#29b6f6", "#001a2e", "CLEAR",   "정상"),
        "WARNING": ("#ffb300", "#1a1200", "CAUTION", "주의"),
        "CRITICAL":("#f44336", "#1a0000", "ALERT",   "경보"),
    }
    col, bg, code, lbl = C.get(sev, C["STANDBY"])
    glow = f"0 0 14px {col}44"

    status_html = (
        f"<div style='background:{bg};border-bottom:1px solid {col}33;padding:16px 18px;'>"
        f"<div style='font-size:.52rem;color:{col};letter-spacing:.22em;opacity:.5'>SYS STATUS</div>"
        f"<div style='font-size:2rem;font-weight:bold;color:{col};letter-spacing:.18em;"
        f"margin:6px 0;{_FO}'>{code}</div>"
        f"<div style='font-size:.75rem;color:{col};opacity:.5'>{lbl}</div>"
        f"</div>"
    )

    log_lines = log.strip().split("\n") if log.strip() else []
    log_items = ""
    for line in log_lines[-30:]:
        c = "#f44336" if "▲ CRITICAL" in line else ("#ffb300" if "◆ WARNING" in line else "#4fc3f7")
        log_items += (
            f"<div style='color:{c};font-size:.82rem;line-height:1.8;"
            f"{_FM};white-space:pre-wrap;margin-bottom:2px'>{line}</div>"
        )
    fallback = log_items or f"<div style='color:#0a3a5a;font-size:.78rem'>— 대기중 —</div>"
    log_html = (
        f"<div style='background:#010d17;padding:12px 18px;flex:1;overflow-y:auto;'>"
        f"<div style='font-size:.5rem;color:#1a5a7a;letter-spacing:.18em;"
        f"margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid #0a2a3a'>"
        f"DETECTION LOG</div>{fallback}</div>"
    )

    return (
        f"<div style='border:1px solid {col};border-radius:4px;overflow:hidden;"
        f"box-shadow:{glow};display:flex;flex-direction:column;height:100%;'>"
        f"{status_html}{log_html}</div>"
    )


def _right_standby():
    return _right_panel("STANDBY", "")


def _ae_chart(scores: list, warn_th: float, crit_th: float) -> str:
    """AE Score 실시간 SVG 차트"""
    W, H = 420, 160
    PAD_L, PAD_R, PAD_T, PAD_B = 48, 12, 20, 28

    if not scores:
        return (
            f"<div style='background:#010d17;border:1px solid #0a2a3a;"
            f"border-radius:4px;width:{W}px;height:{H+10}px;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"<span style='color:#0a3a5a;font-size:.6rem;{_FM}'>AE SCORE — 대기중</span></div>"
        )

    n      = len(scores)
    mx     = max(max(scores) * 1.25, crit_th * 1.5, 0.0001)
    gw     = W - PAD_L - PAD_R
    gh     = H - PAD_T - PAD_B

    def sx(i):  return PAD_L + int(i / max(n - 1, 1) * gw)
    def sy(v):  return PAD_T + gh - int(v / mx * gh)

    # 최신값 기준 색상
    last = scores[-1]
    col  = "#f44336" if last >= crit_th else ("#ffb300" if last >= warn_th else "#29b6f6")

    # 라인 path
    pts  = " ".join(f"{'M' if i == 0 else 'L'}{sx(i)},{sy(v)}" for i, v in enumerate(scores))
    # 채움 영역
    fill = f"M{sx(0)},{sy(scores[0])} " + \
           " ".join(f"L{sx(i)},{sy(v)}" for i, v in enumerate(scores)) + \
           f" L{sx(n-1)},{PAD_T+gh} L{sx(0)},{PAD_T+gh} Z"

    # threshold 선 위치
    wy = sy(warn_th)
    cy = sy(crit_th)

    # Y축 레이블
    y_labels = ""
    for frac in [0, 0.5, 1.0]:
        val = mx * frac
        yp  = sy(val)
        y_labels += (
            f"<text x='{PAD_L-4}' y='{yp+4}' text-anchor='end' "
            f"font-size='7' fill='#1a4a6a'>{val:.4f}</text>"
        )

    svg = (
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='background:#010d17;display:block;'>"

        # 배경 그리드
        f"<line x1='{PAD_L}' y1='{PAD_T}' x2='{PAD_L}' y2='{PAD_T+gh}' stroke='#0a2a3a' stroke-width='1'/>"
        f"<line x1='{PAD_L}' y1='{PAD_T+gh}' x2='{W-PAD_R}' y2='{PAD_T+gh}' stroke='#0a2a3a' stroke-width='1'/>"

        # CRITICAL / WARNING threshold 점선
        f"<line x1='{PAD_L}' y1='{cy}' x2='{W-PAD_R}' y2='{cy}' "
        f"stroke='#f44336' stroke-width='1' stroke-dasharray='4,3' opacity='.5'/>"
        f"<text x='{W-PAD_R-1}' y='{cy-3}' text-anchor='end' font-size='7' fill='#f44336' opacity='.7'>CRITICAL</text>"

        f"<line x1='{PAD_L}' y1='{wy}' x2='{W-PAD_R}' y2='{wy}' "
        f"stroke='#ffb300' stroke-width='1' stroke-dasharray='4,3' opacity='.5'/>"
        f"<text x='{W-PAD_R-1}' y='{wy-3}' text-anchor='end' font-size='7' fill='#ffb300' opacity='.7'>WARNING</text>"

        # 채움 영역
        f"<path d='{fill}' fill='{col}' opacity='.1'/>"

        # 메인 라인
        f"<path d='{pts}' fill='none' stroke='{col}' stroke-width='1.5' stroke-linejoin='round'/>"

        # 최신 점
        f"<circle cx='{sx(n-1)}' cy='{sy(last)}' r='3' fill='{col}'/>"

        # Y축 레이블
        f"{y_labels}"

        # 타이틀
        f"<text x='{PAD_L}' y='11' font-size='7' fill='#1a5a7a' "
        f"font-family='Share Tech Mono,monospace' letter-spacing='1'>AE SCORE MONITOR</text>"

        # 현재값
        f"<text x='{W-PAD_R}' y='11' text-anchor='end' font-size='8' fill='{col}' "
        f"font-family='Share Tech Mono,monospace'>{last:.5f}</text>"

        f"</svg>"
    )
    return (
        f"<div style='background:#010d17;border:1px solid #0a2a3a;"
        f"border-radius:4px;overflow:hidden;'>{svg}</div>"
    )


def _popup(r) -> str:
    from datetime import datetime as _dt
    if r is None or r.get("severity", "NORMAL") != "CRITICAL":
        return ""
    evt  = r.get("event_type", "-").replace("'", "").replace('"', "")
    desc = r.get("description", "").replace("'", "").replace('"', "")
    conf = int(r.get("confidence", 0) * 100)
    ts   = _dt.now().strftime("%H:%M:%S")
    uid  = f"p{ts.replace(':', '')}"
    js = (
        f"var o=document.getElementById('{uid}');if(o)o.remove();"
        f"var e=document.createElement('div');"
        f"e.id='{uid}';"
        f"e.setAttribute('style',"
        f"'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);"
        f"z-index:999999;background:#0a0000;border:3px solid #f44336;"
        f"border-radius:10px;padding:44px 52px;min-width:520px;max-width:640px;"
        f"text-align:center;font-family:monospace;"
        f"box-shadow:0 0 80px #f4433688,0 0 0 9999px rgba(0,0,0,.55);');"
        f"var h='';"
        f"h+='<div style=\\'font-size:.55rem;color:#f44336;letter-spacing:.25em;opacity:.5;margin-bottom:14px\\'>🚨 LLM CRITICAL THREAT // {ts}</div>';"
        f"h+='<div style=\\'font-size:2.2rem;font-weight:bold;color:#f44336;font-family:Orbitron,monospace;letter-spacing:.12em;margin-bottom:16px\\'>긴급 위협 감지</div>';"
        f"h+='<div style=\\'font-size:1.05rem;color:#f44336;margin-bottom:8px\\'><b>{evt}</b> &nbsp;|&nbsp; CONF {conf}%</div>';"
        f"h+='<div style=\\'font-size:.95rem;color:#f44336;opacity:.85;margin-bottom:20px\\'>{desc}</div>';"
        f"h+='<div style=\\'font-size:.65rem;color:#f44336;opacity:.55;border-top:1px solid #f4433633;padding-top:12px;margin-bottom:20px\\'>📧 담당자 이메일 발송 완료</div>';"
        f"h+='<button style=\\'background:#f44336;border:none;color:#000;padding:12px 40px;font-size:.85rem;cursor:pointer;border-radius:5px;font-family:monospace\\'>확인 ✕</button>';"
        f"e.innerHTML=h;"
        f"e.querySelector('button').onclick=function(){{e.remove();}};"
        f"document.body.appendChild(e);"
        f"setTimeout(function(){{if(e.parentNode)e.remove();}},10000);"
    )
    return f'<img src="#" onerror="{js}" style="display:none">'


async def run_p(video, mode):
    global _eng, _eng_f, _mode
    if mode != _mode:
        _mode = mode
        _eng,   _ = load_engine(mode)
        _eng_f, _ = load_engine(mode)
    if video is None:
        yield _PLACEHOLDER, _right_standby(), _ae_chart([], WARN_TH, CRIT_TH), None
        return
    if hasattr(_eng, "reset"):
        _eng.reset()

    from collections import deque
    log_acc   = ""
    ae_scores = deque(maxlen=80)   # 최근 80개 점수 보관

    import outdoor
    async for html, log, changed, pending, lm_done, ae_score in outdoor.run_perimeter_watch(_eng, video):
        if ae_score is not None:
            ae_scores.append(ae_score)
        if changed and log:
            if not pending and "[LLM 분류중...]" in log_acc:
                parts = log_acc.split("\n\n")
                parts = [p for p in parts if "[LLM 분류중...]" not in p]
                parts.insert(0, log)
                log_acc = "\n\n".join(parts)
            else:
                log_acc = log + ("\n\n" + log_acc if log_acc else "")
        recent = "\n\n".join(log_acc.split("\n\n")[:3])
        sev = "CRITICAL" if "▲ CRITICAL" in recent else ("WARNING" if "◆ WARNING" in recent else "NORMAL")
        popup_out = None
        if lm_done and lm_done.severity.value == "CRITICAL":
            desc = lm_done.description
            if "고위험 이상 감지" not in desc and "이상 패턴 감지" not in desc:
                d = {"severity": lm_done.severity.value, "event_type": lm_done.event_type.value,
                     "description": desc, "confidence": lm_done.confidence}
                popup_out = _popup(d)
                gr.Warning(f"🚨 긴급 위협 | {lm_done.event_type.value} | {desc} | "
                           f"CONF {lm_done.confidence:.0%} | 📧 이메일 발송", duration=10)
        chart = _ae_chart(list(ae_scores), WARN_TH, CRIT_TH)
        yield html, _right_panel(sev, log_acc), chart, popup_out


async def run_f(video, mode):
    if video is None:
        yield _PLACEHOLDER, _right_standby(), _ae_chart([], WARN_TH, CRIT_TH), None
        return
    if hasattr(_eng_f, "reset"):
        _eng_f.reset()

    from collections import deque
    log_acc   = ""
    ae_scores = deque(maxlen=80)

    import indoor
    async for html, log, changed, pending, lm_done, ae_score in indoor.run_facility_watch(_eng_f, video):
        if ae_score is not None:
            ae_scores.append(ae_score)
        if changed and log:
            if not pending and "[LLM 분류중...]" in log_acc:
                parts = log_acc.split("\n\n")
                parts = [p for p in parts if "[LLM 분류중...]" not in p]
                parts.insert(0, log)
                log_acc = "\n\n".join(parts)
            else:
                log_acc = log + ("\n\n" + log_acc if log_acc else "")
        recent = "\n\n".join(log_acc.split("\n\n")[:3])
        sev = "CRITICAL" if "▲ CRITICAL" in recent else ("WARNING" if "◆ WARNING" in recent else "NORMAL")
        popup_out = None
        if lm_done and lm_done.severity.value == "CRITICAL":
            desc = lm_done.description
            if "고위험 이상 감지" not in desc and "이상 패턴 감지" not in desc:
                d = {"severity": lm_done.severity.value, "event_type": lm_done.event_type.value,
                     "description": desc, "confidence": lm_done.confidence}
                popup_out = _popup(d)
                gr.Warning(f"🚨 긴급 위협 | {lm_done.event_type.value} | {desc} | "
                           f"CONF {lm_done.confidence:.0%} | 📧 이메일 발송", duration=10)
        chart = _ae_chart(list(ae_scores), WARN_TH, CRIT_TH)
        yield html, _right_panel(sev, log_acc), chart, popup_out


def refresh():
    return get_audit_log_str(limit=25)


init_db()
os.makedirs("database/frames", exist_ok=True)

with gr.Blocks(css=CSS, title="SCCTV v2.0") as demo:
    gr.HTML(
        "<div class='sys-hdr'><div class='dot'></div><div>"
        "<div class='title'>SMART CCTV v2.0 — DEFENSE SURVEILLANCE SYSTEM</div>"
        "<div class='sub'>ConvAE ANOMALY DETECTION + LLM SITUATION ANALYSIS // DUAL-MODE PIPELINE</div>"
        "</div></div>"
    )

    # 상단 컨트롤 바: 모드 선택 + 업로드 (인라인)
    with gr.Row():
        mode_r = gr.Radio(
            ["hybrid", "self_model", "gpt"], value=DEFAULT_MODE,
            label="ENGINE MODE", elem_classes=["mode-sel"],
            info="hybrid: ConvAE→LLM  |  self_model: 폐쇄망  |  gpt: LLM only",
            scale=4
        )
        with gr.Column(scale=1, min_width=220):
            pv = gr.Video(label="📁 PERIMETER 영상 업로드", height=52, elem_classes=["top-upload"])
            fv = gr.Video(label="📁 FACILITY 영상 업로드",  height=52, elem_classes=["top-upload"])

    p_popup = gr.HTML("")
    f_popup = gr.HTML("")

    with gr.Tabs():
        with gr.Tab("◈  PERIMETER"):
            gr.HTML("<div class='zone'>ZONE-A // INTRUSION · LOITERING · VEHICLE_INTRUSION</div>")
            with gr.Row(equal_height=False, elem_classes=["main-row"]):
                # 가운데: LATEST FRAME 고정 크기
                with gr.Column(scale=3, min_width=0):
                    pf = gr.HTML(value=_PLACEHOLDER, elem_id="perimeter-frame")
                    pb = gr.Button("▶  INITIATE PERIMETER SCAN", elem_classes=["scan-btn"])
                # 가운데: AE Score 차트
                with gr.Column(scale=2, min_width=0):
                    pc = gr.HTML(value=_ae_chart([], WARN_TH, CRIT_TH))
                # 오른쪽: STATUS + LOG
                with gr.Column(scale=2, min_width=300):
                    pr = gr.HTML(value=_right_standby())
            pb.click(run_p, inputs=[pv, mode_r], outputs=[pf, pr, pc, p_popup])

        with gr.Tab("◈  FACILITY"):
            gr.HTML("<div class='zone'>ZONE-B // ABNORMAL_BEHAVIOR · OBJECT_LEFT · INTRUSION</div>")
            with gr.Row(equal_height=False, elem_classes=["main-row"]):
                with gr.Column(scale=3, min_width=0):
                    ff = gr.HTML(value=_PLACEHOLDER, elem_id="facility-frame")
                    fb = gr.Button("▶  INITIATE FACILITY SCAN", elem_classes=["scan-btn"])
                with gr.Column(scale=2, min_width=0):
                    fc = gr.HTML(value=_ae_chart([], WARN_TH, CRIT_TH))
                with gr.Column(scale=2, min_width=300):
                    fr_ = gr.HTML(value=_right_standby())
            fb.click(run_f, inputs=[fv, mode_r], outputs=[ff, fr_, fc, f_popup])

        with gr.Tab("◈  AUDIT LOG"):
            gr.HTML("<div class='zone'>AUDIT TRAIL // TIMESTAMP · MODE · EVENT · SEVERITY</div>")
            ab = gr.Textbox(label="", lines=24, interactive=False,
                            value=refresh(), elem_classes=["audit-panel"])
            rb = gr.Button("↻  REFRESH", elem_classes=["scan-btn"])
            rb.click(refresh, outputs=ab)

        with gr.Tab("◈  PIPELINE"):
            gr.Markdown("""
```
INPUT FRAME (매 프레임)
    ↓
[ ConvAE ]  AE Score 계산
    ↓
  NORMAL   →  CLEAR
  WARNING  →  CAUTION  →  [ LLM 4-frame montage ]
  CRITICAL →  ALERT    →  [ LLM ] + 팝업 + 이메일
```
| MODE | ENGINE | 폐쇄망 | EDGE |
|------|--------|--------|------|
| **self_model** | ConvAE only | ✅ | ✅ ONNX |
| hybrid | ConvAE + LLM | ConvAE 단독 | ONNX |
| gpt | LLM only | ❌ | ❌ |
            """)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)