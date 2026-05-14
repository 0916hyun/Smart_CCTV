# gpt_test.py — OpenAI GPT-4o-mini 연결 진단
# 사용법: python gpt_test.py

import asyncio, sys, os, base64
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "gpt-4o-mini"


async def test():
    from api_key import openai_api_key
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)  # OpenAI 기본 엔드포인트

    # 더미 이미지 base64 (재사용)
    arr = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # 1) 텍스트 only
    print("=== 테스트 1: 텍스트 only ===")
    try:
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in Korean. One sentence only."}],
            max_tokens=50,
        )
        print("성공:", r.choices[0].message.content)
    except Exception as e:
        print("실패:", e)

    # 2) 이미지 인식
    print("\n=== 테스트 2: 이미지 인식 ===")
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "이미지에 무엇이 보이나요? 한 문장으로."},
            ]}],
            max_tokens=100,
        )
        print("성공:", resp.choices[0].message.content)
    except Exception as e:
        print("실패:", e)

    # 3) 탐지 JSON 응답
    print("\n=== 테스트 3: 탐지 JSON 응답 ===")
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    'Security camera AI. Reply with JSON only:\n'
                    '{"event_type":"NORMAL","severity":"NORMAL","confidence":0.2,"description":"정상 상황"}\n'
                    'Return ONLY the JSON object, no markdown.'
                )},
            ]}],
            max_tokens=200,
        )
        print("성공:", resp.choices[0].message.content)
    except Exception as e:
        print("실패:", e)


asyncio.run(test())