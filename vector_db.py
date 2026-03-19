from openai import OpenAI
from api_key import openai_api_key
import numpy as np
import os
import cv2
import base64


def video_caption(image_list):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "These images are from a video, answer the question."
            },
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text": "Generate a description for the video. \
                            Your answer should be no more than 50 words."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

# 텍스트 데이터를 임베딩 벡터로 변환하는 함수
def get_text_embedding(text, model="text-embedding-3-small"):
   client = OpenAI(api_key=openai_api_key)
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
 
train_path = 'training'
files = os.listdir(train_path)
texts = []

for file in files:
    if file.endswith('.mp4'):
        video = cv2.VideoCapture(os.path.join(train_path,file))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(file)
        print(length, " frames")
        stride = int(length / 6)
        cnt = 0
        base64frames = []
        while video.isOpened():
            success, frame= video.read()
            cnt=cnt+1
            if not success or len(base64frames)==5:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            if cnt==stride:
                cnt = 0
                base64frames.append(base64.b64encode(buffer).decode('utf-8'))
        video.release()
        caption = video_caption(base64frames)
        print(caption)
        texts.append(caption)

embeddings = np.array([get_text_embedding(text) for text in texts])
np.save('database/embeddings.npy', embeddings)