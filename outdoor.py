import cv2
from datetime import datetime
from gpt_api import *
import asyncio
import numpy as np
from PIL import Image
from utils import *

async def record_and_summarize_video():
    try:
        fileName = datetime.now().strftime("%H_%M_%S")
        video = cv2.VideoWriter('recordings/outdoor_'+fileName+'.mp4',
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
                    save_sqlite3("outdoor",response)
                    query_in_progress = False
                task = asyncio.create_task(query_with_multiple_image(
                    stack_imgs, "What is happening in the images. If there are suspicious actions, please report.\
                        Your answer should be no more than 30 words.",
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

async def classify_package():
    im = get_outdoor_image()
    task = asyncio.create_task(query_with_single_image(im,
        "There are some packages in front of the door.\
        Classify the packaages as either box packages or food deliveries. \
        Return 'box' if they are box packages, 'food' if they are food deliveries,\
        and 'other' if they do not fit into either category. Answer using only single word."))
    response = await task
    print(response)