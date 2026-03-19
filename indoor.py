import cv2
from datetime import datetime
from gpt_api import *
import asyncio
import numpy as np
from PIL import Image
from utils import * 
import json

async def record_and_summarize_video():
    try:
        fileName = datetime.now().strftime("%H_%M_%S")
        video = cv2.VideoWriter('recordings/indoor_'+fileName+'.mp4',
                                cv2.VideoWriter_fourcc(*"mp4v"), 10, (702-224, 488-129))
        stack_imgs = []
        stack_cnt = 0
        query_in_progress = False

        while True:
            im = get_outdoor_image()
            display_frame = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
            video.write(display_frame)
            stack_cnt = stack_cnt + 1
            if stack_cnt == 10:
                pil_image = Image.fromarray(display_frame)
                stack_imgs.append(pil_image)
                stack_cnt = 0
            if len(stack_imgs) == 5:
                if query_in_progress:
                    response = await task
                    print(response)
                    dict = json.loads(response)
                    save_sqlite3("indoor",dict["appearance"]+" "+dict["actions"])
                    query_in_progress = False
                task = asyncio.create_task(query_with_multiple_image(
                    stack_imgs, "The house is supposed to be empty. \
                        Given the image from the home surveillance camera, \
                        describe the stranger's appearance and actions. \
                        Answer in JSON format with keys 'appearance' and 'actions'. \
                        Answer should be less than 50 words in total. \
                        Do NOT include 'json' and markdown.",
                        "You are a security officer."
                ))
                stack_imgs = []
                query_in_progress = True
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        if query_in_progress:
            response = await task
            print(response)
        video.release()