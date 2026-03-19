from openai import AsyncOpenAI
from api_key import openai_api_key
from utils import encode_image

async def query_with_single_image(image, text, text_system=None):
    client = AsyncOpenAI(api_key=openai_api_key)
    msg = []
    if text_system is not None:
        msg.append(
            {
                "role": "system",
                "content": text_system
            }
        )
    msg.append(
        {
            "role": "user",
            "content":[
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{encode_image(image)}"
                    }
                }
            ]
        }
    )
    msg.append({
        "role": "user",
        "content": text
    })
    response = await client.chat.completions.create(model="gpt-4o", messages=msg)
    return response.choices[0].message.content

async def query_with_multiple_image(images, text, text_system=None):
    client = AsyncOpenAI(api_key=openai_api_key)
    msg = []
    if text_system is not None:
        msg.append(
            {
                "role": "system",
                "content": text_system
            }
        )
    msg.append(
        {
            "role": "user",
            "content":[
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{[encode_image(img) for img in images]}"
                    }
                }
            ]
        }
    )
    msg.append({
        "role": "user",
        "content": text
    })
    response = await client.chat.completions.create(model="gpt-4o", messages=msg)
    return response.choices[0].message.content

async def get_text_embedding(text, model="text-embedding-3-small"):
   client = AsyncOpenAI(api_key=openai_api_key)
   text = text.replace("\n", " ")
   response = await client.embeddings.create(input = [text], model=model)
   return response.data[0].embedding

