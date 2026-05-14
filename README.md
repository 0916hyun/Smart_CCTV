# 🛡️ 저화질에 강건한 AI 이상탐지 CCTV

[![Demo Video](https://img.youtube.com/vi/qo1KAWjG7HQ/maxresdefault.jpg)](https://youtu.be/qo1KAWjG7HQ)

직접 설계한 경량 CNN 오토인코더(ConvAE)와 VLM(GPT-4o-mini)을 결합한 2단계 하이브리드 이상 탐지 시스템입니다. 저화질·야간 CCTV 환경에서도 강건하게 동작하며, 1차 이상 탐지 → 2차 VLM 세부 분석 → 담당자 긴급 알림까지 자동화된 파이프라인을 구성합니다.

---

## 📌 1. Project Overview

이 프로젝트는 CCTV 영상에서 위험·이상 상황을 실시간으로 탐지하고 대응하는 AI 감시 시스템입니다.

- **1차 감지**: 직접 설계한 경량 ConvAE로 이상 패턴을 즉각 탐지
- **2차 분석**: 이상 감지 시에만 VLM을 호출하여 위협 유형 분류 및 상황 설명
- **저화질 강건성**: 픽셀 재구성 오차 기반 탐지로 화질에 덜 의존적
- **폐쇄망 대응**: ONNX 변환으로 외부 API 없이 단독 운용 가능

---

## ✨ 2. 주요 기능

### 🔍 이상 탐지 (ConvAE)
- 정상 영상만으로 학습된 경량 CNN 오토인코더
- 16프레임 시퀀스 입력 → 재구성 오차(AE Score)로 이상 감지
- 학습 데이터에 없는 미지 위협도 탐지 가능
- ONNX 변환 → 엣지 디바이스 배포 가능

### 🤖 VLM 세부 분석 (GPT-4o-mini)
- 이상 감지 시에만 VLM 호출 (비용 최적화)
- 4프레임 몽타주 전송으로 시간 흐름 기반 위협 분류
- 위협 유형: INTRUSION / ABNORMAL_BEHAVIOR / LOITERING / OBJECT_LEFT 등
- 한국어 위협 설명 자동 생성

### 📊 실시간 모니터링 UI
- AE Score 실시간 차트 (WARNING / CRITICAL 임계선 표시)
- 2단계 탐지 로그 (1차 즉각 감지 → 2차 VLM 분류 결과 업데이트)
- CRITICAL 확정 시 화면 중앙 팝업 알림

### 🗃️ 데이터 관리
- SQLite: 모든 탐지 이벤트 이력 저장 및 조회
- VectorDB (FAISS): LLM 설명 임베딩 저장 → 유사 과거 사례 검색

### 📧 알림
- LLM이 CRITICAL을 확정한 경우에만 이메일 발송 (오경보 최소화)
- Gmail SMTP 기반

---

## ❓ 3. Why This Project?

단순 모션 감지만으로는 실제 상황을 충분히 이해하기 어렵습니다. 예를 들어 다음과 같은 판단은 기존 방식으로 처리하기 어렵습니다.

- "이 움직임이 단순 통행인가, 침입인가?"
- "화면 변화가 조명 변화인가, 화재인가?"
- "점진적으로 진행되는 위협을 초기에 감지할 수 있는가?"

이 프로젝트는 비지도학습 기반 이상 탐지로 1차 필터링을 수행하고, 이상이 감지된 경우에만 VLM을 호출하여 상황을 자연어로 분류하는 구조를 채택했습니다.

---

## 🧠 4. Core Idea

### 1) 🏗️ 직접 설계한 경량 ConvAE

```
인코더: Conv2d + BatchNorm + LeakyReLU × 4 (stride=2 다운샘플링)
         128×128 → 64×64 → 32×32 → 16×16 → 잠재 벡터 (256, 8, 8)
디코더: ConvTranspose2d + BatchNorm + ReLU × 3 + Sigmoid
         잠재 벡터 → 128×128 복원
```

- 파라미터 약 5M → 경량, ONNX 변환 가능
- UCF-Crime 데이터셋의 정상 영상만으로 학습
- 재구성 오차(MSE) = AE Score → 임계값 초과 시 이상 판정

### 2) ⚡ 2단계 하이브리드 파이프라인

```
매 프레임: ConvAE → AE Score 계산 (즉각)
                ↓
    NORMAL → CLEAR (VLM 호출 없음)
    WARNING → CAUTION → VLM 4프레임 몽타주 분석
    CRITICAL → ALERT  → VLM 분석 + 팝업 + 이메일
```

### 3) 🎞️ 4프레임 몽타주 전송

단일 프레임 대신 시간 간격을 둔 4프레임을 가로로 이어붙여 VLM에 전송합니다.

```
[frame N-36] → [frame N-24] → [frame N-12] → [frame N]
   과거                                          현재
```

VLM이 시간 흐름을 보고 판단하므로 단일 프레임 대비 정확도가 향상됩니다.

### 4) 🛡️ 오경보 필터링

- **Hysteresis**: 연속 8프레임 이상 낮은 severity가 유지될 때만 하향
- **CRITICAL Hold**: CRITICAL 감지 후 15초간 유지
- **LLM 확정 시에만 알림**: 1차 감지 단독으로는 알림 미발송

---

## 🔄 5. System Pipeline

```
입력 영상
        ↓
[매 프레임] ConvAE → AE Score
        ↓
  임계값 비교 (Hysteresis + CRITICAL Hold)
        ↓
NORMAL ───────────────────→ CLEAR 표시
        ↓
WARNING / CRITICAL
        ↓
[백그라운드] VLM 4프레임 몽타주 분석
        ↓
Phase 1 결과 즉각 표시
        ↓
Phase 2 VLM 결과 업데이트
        ↓
CRITICAL 확정 시
├── 팝업 알림
├── SMTP 이메일 발송
└── VectorDB 저장 (유사 사례 검색용)
        ↓
SQLite 이벤트 로그 저장
```

---

## 🛠️ 6. How It Was Implemented

### 🤖 6.1 ConvAE 학습
`models/anomaly_ae.py`에 직접 설계한 ConvAE를 UCF-Crime 데이터셋의 NormalVideos로 학습합니다.

- 입력: 16프레임 grayscale 시퀀스 (128×128)
- Loss: MSE (재구성 오차 최소화)
- 출력: ONNX 변환 모델

### ⚙️ 6.2 탐지 엔진

- `SelfModelEngine`: ConvAE ONNX 추론, 매 프레임 AE Score 계산
- `GPTEngine`: VLM 4프레임 몽타주 분석, JSON 위협 분류
- `HybridEngine`: ConvAE 1차 → VLM 2차 파이프라인 조율

### 🔄 6.3 비동기 처리

LLM 호출을 `asyncio.create_task()`로 백그라운드 실행하여 영상 재생이 끊기지 않습니다.

```python
lm_task = asyncio.create_task(engine.gpt.detect_async_montage(frames))
```

---

## 🖥️ 7. UI

Gradio 기반 UI로 구성되어 있습니다.

- **PERIMETER / FACILITY 탭**: 존별 독립 모니터링
- **LATEST FRAME**: 실시간 탐지 프레임 표시
- **AE Score 차트**: 이상 점수 실시간 시각화
- **SYS STATUS**: CLEAR / CAUTION / ALERT 상태 표시
- **DETECTION LOG**: 2단계 탐지 결과 누적 로그
- **AUDIT LOG 탭**: SQLite 저장 이벤트 전체 조회
- **PIPELINE 탭**: 시스템 구조 다이어그램

---

## 🗂️ 8. Project Structure

```
.
├── main.py                  # Gradio UI 및 파이프라인 orchestration
├── outdoor.py               # PERIMETER 존 탐지 파이프라인
├── indoor.py                # FACILITY 존 탐지 파이프라인
├── config.py                # 임계값, 학습 파라미터 설정
├── events.py                # EventResult, EventType, Severity 정의
├── api_key.py               # API 키 (Git 제외)
├── engines/
│   ├── base.py              # DetectionEngine ABC
│   ├── self_model_engine.py # ConvAE ONNX 추론 엔진
│   ├── gpt_engine.py        # GPT-4o-mini VLM 엔진
│   └── hybrid_engine.py     # ConvAE + VLM 하이브리드 엔진
├── models/
│   ├── anomaly_ae.py        # ConvAE 모델 정의
│   └── checkpoints/         # 학습된 가중치 (.pt, .onnx)
├── utils.py                 # DB 저장, 이메일, 유틸리티
├── vector_db.py             # FAISS 임베딩 저장 및 검색
├── quick_test.py            # 클래스별 AE Score 빠른 확인
├── gpt_test.py              # VLM API 연결 테스트
├── database/
│   ├── db.db                # SQLite 이벤트 로그
│   └── frames/              # 탐지 시 저장 프레임
└── demo_videos/             # 테스트용 영상
```

---

## ⚙️ 9. Tech Stack

| 분류 | 기술 |
|------|------|
| AI / ML | PyTorch, ONNX Runtime, OpenAI GPT-4o-mini |
| Interface | Gradio |
| Storage | SQLite3, FAISS |
| Notification | smtplib (Gmail SMTP) |
| Utility | OpenCV, PIL, NumPy, asyncio |

---

## 🚀 10. Getting Started

### 1️⃣ 의존성 설치

```bash
pip install torch torchvision onnxruntime
pip install gradio openai opencv-python pillow numpy faiss-cpu
```

### 2️⃣ API 키 설정

`api_key.py` 파일 생성:

```python
openai_api_key = "sk-..."       # OpenAI API Key
email_id  = "your_gmail_id"     # Gmail 아이디 (@gmail.com 제외)
email_pwd = "your_app_password" # Gmail 앱 비밀번호 (16자리)
```

### 3️⃣ 임계값 설정

```bash
python quick_test.py  # 클래스별 AE Score 확인 후 config.py 조정
```

### 4️⃣ 실행

```bash
python main.py
# http://localhost:7860 접속
```

---

## 🔍 11. VectorDB 검색 기능

CRITICAL 이벤트 발생 시 LLM 설명을 임베딩하여 FAISS에 저장합니다.

```
새 CRITICAL 이벤트 → LLM 설명 임베딩 → FAISS 검색 → 유사 과거 사례 반환
```

단순 키워드 검색이 아닌 의미 기반 유사 사례 검색으로, 반복 패턴 위협 분석에 활용할 수 있습니다.

---

## ⚠️ 12. Limitations

- 영상 파일 입력 기반 프로토타입 (RTSP 스트림 미지원)
- ConvAE는 학습 환경과 크게 다른 카메라에서는 임계값 재조정 필요
- 저화질 이상 장면(방화, 폭발)에서 VLM 오분류 가능성 존재
- 장시간 운용 시 API 비용 모니터링 필요

---

## 🔮 13. Future Work

- 📡 실제 IP Camera / RTSP 스트림 연동
- 🧠 ConvAE → ConvLSTM-AE 교체로 시간적 의존성 명시적 학습
- 🗺️ 다중 카메라 존 관리 및 통합 대시보드
- 📱 모바일 알림 연동
- 🏷️ 자동 임계값 캘리브레이션

---

## ✅ 14. Summary

**직접 설계한 경량 ConvAE 이상 탐지 → VLM 세부 분석 → DB 저장 → 긴급 알림**

까지 이어지는 2단계 하이브리드 파이프라인을 구현했습니다.

- 저화질·야간 CCTV에서도 강건한 이상 탐지
- 미지 위협 대응 (정상 패턴만 학습, 벗어나면 감지)
- ConvAE 매 프레임 실시간 처리로 즉각적인 1차 감지
- ONNX 기반 엣지 배포 가능
