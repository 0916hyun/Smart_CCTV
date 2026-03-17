# 🛡️ Smart CCTV with GPT

> GPT 기반의 **실내·실외 스마트 방범 시스템**입니다.
> CCTV 화면을 분석하여 사람 출현, 택배/배달 물체, 침입자, 낙상, 화재, 침수 등의 상황을 감지하고, 필요 시 **녹화 → 요약 → 로그 저장 → 메일 알림**까지 연결합니다.

---

## 📌 1. Project Overview

기존 CCTV는 영상을 저장하거나 사람이 직접 확인해야 하는 경우가 많습니다.
이 프로젝트는 이러한 한계를 줄이기 위해, **비전-언어 모델(VLM)** 을 활용하여 장면을 이해하고 실제 기능과 연결되는 **스마트 CCTV 백엔드**를 구현한 프로젝트입니다.

### ✨ 주요 기능

#### 🌳 Outdoor Monitoring

* 👤 사람 등장 여부 감지
* 📦 문 앞 택배/배달 박스 감지
* 🎥 사람 등장 시 자동 녹화
* 📝 연속 프레임 기반 행동 요약
* 📧 택배 도착 시 메일 알림

#### 🏠 Indoor Monitoring

* 🧍 재실 모드: 낙상 감지
* 🚨 외출 모드: 침입자 감지
* 🔥 화재/연기 감지
* 💧 침수 감지
* 🧾 침입자 인상착의 및 행동 요약

#### ⚙️ System Features

* 🖥️ Gradio 기반 실내·실외 통합 UI
* 🗂️ SQLite3 기반 이벤트 로그 저장
* 📮 Gmail SMTP 기반 위급 상황 알림 전송
* 🔎 임베딩 기반 이벤트 검색 기능

---

## ❓ 2. Why This Project?

스마트 홈 환경에서는 단순 모션 감지만으로 실제 상황을 충분히 이해하기 어렵습니다.
예를 들어 다음과 같은 질문은 단순 객체 탐지만으로 처리하기 어렵습니다.

* "현관 앞에 사람이 온 것인가?"
* "문 앞에 놓인 물체가 택배인가?"
* "실내에서 누군가 쓰러진 것인가?"
* "외출 중 집 안에 침입자나 화재, 침수가 발생했는가?"

이 프로젝트는 이러한 문제를 해결하기 위해, CCTV 영상을 **GPT-4o에 질의하여 장면을 의미적으로 해석**하고, 그 결과를 **녹화·요약·DB 저장·알림 기능**과 연결하도록 설계했습니다.

즉, 단순한 저장형 CCTV가 아니라
**장면 이해가 가능한 스마트 감시 시스템**을 목표로 했습니다.

---

## 🧠 3. Core Idea

이 프로젝트의 핵심 아이디어는 다음과 같습니다.

### 1) 👁️ Vision-Language 기반 장면 이해

전용 감지 모델을 직접 학습시키는 대신, GPT-4o에 이미지를 전달하여 장면을 해석합니다.
이를 통해 다음과 같은 다양한 판단을 하나의 API 흐름으로 처리할 수 있습니다.

* 사람 유무 판단
* 박스/택배 존재 여부 판단
* 낙상 여부 판단
* 화재/침수 여부 판단
* 행동 요약 및 인상착의 설명

### 2) 💸 Cost-aware Query Strategy

모든 프레임을 분석하면 비용이 커지기 때문에,

* 평상시: 긴 주기로 감시
* 이벤트 발생 시: 촘촘한 프레임 샘플링 후 집중 분석

구조로 설계했습니다.
즉, **평상시 저비용 감시 + 이벤트 발생 시 정밀 분석** 전략입니다.

### 3) 🎬 Event-driven Recording

사람이 등장했을 때만 녹화를 시작하고, 사라지면 종료합니다.
이를 통해 중요 장면만 저장할 수 있도록 구성했습니다.

### 4) 🗃️ Lightweight Local Logging

이벤트 결과는 SQLite3에 저장되어 UI에서 바로 확인할 수 있습니다.
별도 대규모 서버 없이 로컬 환경에서도 쉽게 관리할 수 있도록 구성했습니다.

---

## 🔄 4. System Pipeline

### 🌳 Outdoor Pipeline

```text
Outdoor Screen Capture
        ↓
GPT-4o Image Query
        ↓
JSON Response { person, box }
        ↓
[person == 1] → Start Recording
        ↓
Sample Frames During Recording
        ↓
GPT-4o Multi-image Summarization
        ↓
Save Summary to DB
        ↓
[box detected repeatedly] → Send Mail Alert
```

### 🏠 Indoor Pipeline

#### 🧍 At Home Mode

```text
Indoor Screen Capture
        ↓
GPT-4o Image Query
        ↓
Fall Detection
        ↓
If detected → Send Mail Alert
```

#### 🚪 Outside Mode

```text
Indoor Screen Capture
        ↓
GPT-4o Image Query
        ↓
JSON Response { person, fire, flood }
        ↓
Intruder / Fire / Flood 판단
        ↓
[person == 1] → Start Recording
        ↓
GPT-4o Multi-image Summarization
        ↓
Save Description to DB
        ↓
Send Mail Alert
```

---

## 🛠️ 5. How It Was Implemented

### 📷 5.1 Screen Capture

실제 RTSP 스트림 대신, 실습 및 프로토타입 환경에 맞게 **화면 캡처 기반**으로 영상을 입력받도록 구현했습니다.

* `capture_outdoor()`: 실외 영역 캡처
* `capture_indoor()`: 실내 영역 캡처

### 🤖 5.2 GPT API Query

`gpt_api.py`에서는 OpenAI 비동기 API를 사용해 이미지 질의를 수행합니다.

* `query_with_single_image()`

  * 단일 이미지에 대한 장면 판단
* `query_with_multiple_image()`

  * 여러 장의 프레임을 이용한 상황 요약
* `query_with_text()`

  * 텍스트 임베딩 생성

### 🌳 5.3 Outdoor Logic

`run_main_outdoor()`는 일정 주기로 실외 화면을 캡처하고 GPT에게 다음과 같이 질의합니다.

* `person == 1` → 사람이 등장한 것으로 판단 → 녹화 시작
* `person == 0` → 녹화 중이었다면 종료
* `box == 1` → 최근 기록을 기준으로 일정 횟수 이상 감지되면 택배 도착 판단

### 🎥 5.4 Recording and Summarization

녹화가 시작되면,

* 0.1초 간격으로 프레임 기록
* 1초 간격으로 대표 프레임 샘플링
* 대표 프레임 5장을 GPT에 전달
* 행동/상황 요약 생성
* 결과를 SQLite DB에 저장

구조로 동작합니다.

### 🏠 5.5 Indoor Logic

`run_main_indoor()`는 모드에 따라 다르게 동작합니다.

#### 🧍 At Home

* 낙상 여부 중심으로 점검
* 낙상 감지 시 메일 전송

#### 🚪 Outside

* 사람, 화재, 침수 여부를 동시에 점검
* 침입자 발생 시 녹화 시작
* 연속 프레임을 사용해 인상착의와 행동 요약 생성
* DB 저장 및 알림 전송

### 🖥️ 5.6 UI / Logging / Notification

Gradio UI에서는 다음을 함께 표시합니다.

* 현재 실내/실외 화면
* 현재 상태 로그
* DB에 저장된 최근 설명 결과

또한,

* SQLite3: 이벤트 로그 저장
* Gmail SMTP: 위급 상황 메일 전송

기능이 연동되어 있습니다.

---

## 🗂️ 6. Project Structure

```bash
.
├── main.py
├── outdoor.py
├── indoor.py
├── gpt_api.py
├── utils.py
├── vector_db.py
├── database/
│   ├── db.db
│   └── embeddings.npy
├── recordings/
└── training/
```

### 📄 File Description

* `main.py`
  → 전체 Gradio UI 실행 및 실내/실외 파이프라인 orchestration

* `outdoor.py`
  → 실외 이벤트 녹화, 택배 감지, 행동 요약

* `indoor.py`
  → 실내 이벤트 녹화, 침입자 행동 및 인상착의 요약

* `gpt_api.py`
  → GPT-4o 이미지 질의 및 텍스트 임베딩 생성

* `utils.py`
  → 화면 캡처, 이미지 인코딩, DB 저장/조회, SMTP 메일 전송

* `vector_db.py`
  → 학습 비디오 요약문 생성 및 임베딩 저장

---

## ⚙️ 7. Tech Stack

### 🧩 Backend

* Python
* OpenAI GPT-4o
* text-embedding-3-small
* OpenCV
* PIL
* NumPy

### 🖥️ Interface

* Gradio

### 🗃️ Storage / Search

* SQLite3
* FAISS

### 📬 Notification

* smtplib (Gmail SMTP)

### 🧪 Utility

* PyAutoGUI
* asyncio
* threading

---

## 🚀 8. Getting Started

### 1️⃣ Install Dependencies

```bash
pip install openai gradio opencv-python pillow numpy faiss-cpu pyautogui
```

### 2️⃣ Prepare API Keys

`api_key.py` 파일을 생성한 뒤 아래 내용을 작성합니다.

```python
openai_api_key = "YOUR_OPENAI_API_KEY"
email_id = "YOUR_GMAIL_ID"
email_pwd = "YOUR_GMAIL_APP_PASSWORD"
```

> Gmail SMTP를 사용하려면 Google 계정의 2단계 인증과 앱 비밀번호 설정이 필요합니다.

### 3️⃣ Prepare Directories

```bash
mkdir -p recordings
mkdir -p database
mkdir -p training
```

### 4️⃣ Build Vector DB (Optional)

```bash
python vector_db.py
```

이 스크립트는 `training/` 폴더의 비디오를 읽어 요약문을 만들고, 이를 임베딩으로 변환해 `database/embeddings.npy`에 저장합니다.

### 5️⃣ Run App

```bash
python main.py
```

실행 후 Gradio UI에서 실내/실외 모니터링 화면을 확인할 수 있습니다.

---

## 🧪 9. Example Features

### 🌳 Outdoor

* 👤 현관 앞 사람 출현 감지
* 🎥 사람 등장 시 자동 녹화
* 📝 연속 프레임 기반 행동 요약
* 📦 택배/배달 물체 감지
* 📧 택배 도착 메일 알림

### 🏠 Indoor

* 🧍 재실 시 낙상 감지
* 🚨 외출 시 침입자 감지
* 🔥 화재/연기 감지
* 💧 침수 감지
* 🧾 침입자 인상착의 및 행동 요약 저장

---

## 🔍 10. Search / Retrieval Feature

이 프로젝트에는 간단한 **벡터 검색 기능**도 포함되어 있습니다.

* 학습용 영상들을 요약문으로 변환
* 요약문을 임베딩으로 변환
* 사용자의 질의를 임베딩으로 변환
* FAISS를 이용해 가장 유사한 이벤트 검색

즉, 단순 감시뿐 아니라 **과거 이벤트 검색**까지 확장 가능한 구조를 갖고 있습니다.

---

## ⚠️ 11. Limitations

* 실제 IP Camera / RTSP 스트림이 아닌 **화면 캡처 기반 시뮬레이션**입니다.
* 판단 결과가 GPT 응답 품질에 의존합니다.
* 프롬프트 및 규칙 기반 로직이라 환경에 따라 오탐/미탐 가능성이 있습니다.
* 로컬 프로토타입 중심 구현이므로 상용 서비스 수준의 보안/예외 처리는 추가 보완이 필요합니다.
* 지속적인 API 호출 구조이므로 비용 최적화가 중요합니다.

---

## 🔮 12. Future Work

* 📡 실제 IP Camera / RTSP 스트림 연동
* 📱 모바일 앱 또는 웹 대시보드 연동
* 🧠 객체 탐지 모델 + VLM 하이브리드 구조 적용
* 🗂️ 이벤트 로그와 녹화 파일 자동 매칭
* 🔔 사용자별 알림 정책 설정
* 📊 이벤트 통계 및 관리용 대시보드 고도화
* 🏷️ 더 정교한 위험 상황 분류 체계 확장

---

## ✅ 13. Summary

이 프로젝트는 단순 영상 저장용 CCTV를 넘어,

**장면 이해 → 이벤트 판단 → 자동 녹화 → 자연어 요약 → 로그 저장 → 알림 전송**

까지 이어지는 **스마트 방범 파이프라인**을 Python 기반으로 구현한 프로젝트입니다.

특히 GPT-4o를 활용해,

* 실외 택배 감지
* 사람 출현 감시
* 실내 낙상 감지
* 외출 중 침입자/화재/침수 감지

와 같은 기능을 유연하게 구성할 수 있다는 점이 핵심입니다.

---

## 📎 14. Recommended README Add-ons

레포를 더 보기 좋게 만들고 싶다면 아래 항목도 추가하면 좋습니다.

* 🖼️ 시스템 구조도 이미지
* 📷 UI 실행 화면 GIF 또는 캡처
* 🎬 데모 영상 링크
* 📚 프로젝트 배경/강의 자료 출처
* 🙋 담당 역할 및 기여도
* 🧾 실행 예시 결과 스크린샷

원하는 경우 다음 단계에서
**배지(badge) 추가 버전**, **GIF/스크린샷 배치 버전**, **영문 README 버전**으로 이어서 다듬을 수 있습니다.
