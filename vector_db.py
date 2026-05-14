# vector_db.py — FAISS 기반 과거 이벤트 검색 (v1.0 호환 유지)
# 기존 기능 그대로 유지. GPT 엔진 사용 시에만 동작.

from openai import OpenAI
import numpy as np
import os
import cv2
import base64


def video_caption(image_list: list, api_key: str) -> str:
    """영상 프레임 리스트 → GPT-4o 캡션 생성"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "These images are from a surveillance video. Describe the situation briefly.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a security event description. No more than 50 words.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_list[0]}"},
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content


def get_text_embedding(text: str, api_key: str,
                        model: str = "text-embedding-3-small") -> list:
    """텍스트 → 임베딩 벡터"""
    client = OpenAI(api_key=api_key)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def build_embeddings(train_path: str, api_key: str,
                      save_path: str = "database/embeddings.npy") -> np.ndarray:
    """
    학습 영상 디렉터리의 영상들로 임베딩 DB 구축.
    GPT 모드에서 과거 유사 이벤트 검색에 사용.
    """
    import os
    files = os.listdir(train_path)
    texts = []

    for file in files:
        if file.endswith(".mp4"):
            video = cv2.VideoCapture(os.path.join(train_path, file))
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(int(length / 6), 1)
            cnt = 0
            base64frames = []

            while video.isOpened() and len(base64frames) < 5:
                success, frame = video.read()
                cnt += 1
                if not success:
                    break
                if cnt == stride:
                    cnt = 0
                    _, buffer = cv2.imencode(".jpg", frame)
                    base64frames.append(base64.b64encode(buffer).decode("utf-8"))
            video.release()

            if base64frames:
                caption = video_caption(base64frames, api_key)
                texts.append(caption)
                print(f"  {file}: {caption[:50]}...")

    embeddings = np.array([get_text_embedding(t, api_key) for t in texts])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    print(f"[VectorDB] 임베딩 저장 완료: {save_path}  ({len(texts)}개)")
    return embeddings


if __name__ == "__main__":
    try:
        from api_key import openai_api_key
        build_embeddings("training", openai_api_key)
    except ImportError:
        print("api_key.py가 없어 vector_db 구축을 건너뜁니다.")
