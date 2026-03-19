import gradio as gr
import cv2
import time
from datetime import datetime
from gpt_api import *
import asyncio
import numpy as np
from PIL import Image
from utils import *
import outdoor
import indoor
import json
import os
import faiss

async def run_main_outdoor_vectordb():
    while True:
        timer = asyncio.create_task(asyncio.sleep(3))
        im = get_outdoor_image()
        task = asyncio.create_task(query_with_single_image(im,
                                                "Generate a description for an image."))
        response = await task
        embed_task = asyncio.create_task(get_text_embedding(response))
        query_embedding = await embed_task
        query_embedding = np.array(query_embedding)[None,:]
        dist, ind = index.search(query_embedding, 3)
        print(dist)
        yield im, dist, ""
        await timer

async def run_main_outdoor():
    recording = False
    box_history = [0,0,0,0]
    logStr = ""
    while True:
        timer = asyncio.create_task(asyncio.sleep(10))
        im = get_outdoor_image()
        task = asyncio.create_task(query_with_single_image(im,
                                        "Make an answer in JSON format. Two keys are 'person' and 'box'. \
                                        If a person appears in the image, save 1, otherwise 0 for 'person'. \
                                        If there are boxes in front of the door, save 1 otherwise save 0 for 'box'. \
                                        Do NOT include 'json' and markdown."))
        print("Query sent")
        response = await task
        print(response)

        dict = json.loads(response)
        box_history.pop(0)
        box_history.append(dict["box"])
        if sum(x !=0 for x in box_history) == 2:
            asyncio.create_task(outdoor.classify_package())
            logStr = "You have got a new package."
            send_email("Alaram from your home camera.", "You have got a package.")

        if dict["person"] == 1 and recording == False:
            recording = True
            record_task = asyncio.create_task(outdoor.record_and_summarize_video())
            logStr = "Now recording..."
        elif dict["person"] == 0 and recording == True:
            recording = False
            record_task.cancel()
            logStr = "Recording finished."
        yield im, logStr, get_db_msg('outdoor')
        await timer

async def run_main_indoor(at_home_mode):
    recording = False
    logStr = ""
    if at_home_mode == "At home":
        while True:
            timer = asyncio.create_task(asyncio.sleep(5))
            im = get_indoor_image()
            task = asyncio.create_task(query_with_single_image(im,
                    "If a people has fallen down on the floor in the image, return 1 otherwise 0."))
            response = await task

            if response=="1":
                logStr = "Fell down occurred."
                send_email("Alaram from your home camera.", "Sombody in the house has fallen down.")
            await timer
            yield im, logStr, get_db_msg('indoor')
    else:
        while True:
            timer = asyncio.create_task(asyncio.sleep(10))
            im = get_indoor_image()
            task = asyncio.create_task(query_with_single_image(im,
                    "Make an answer in JSON format. Three keys are 'person', 'fire', 'flood'. \
                    If a person appears in the image, save 1, otherwise save 0 in 'person' key. \
                    If fire or smoke appears in the image, save 1, otherwise save 0 in 'fire' key. \
                    If flooding appears in the image, save 1, otherwise save 0 in 'flood' key. \
                    Do NOT include 'json' and markdown."))
            response = await task
            print(response)
            dict = json.loads(response)
            if dict["person"] == 1 and recording == False:
                recording = True
                record_task = asyncio.create_task(indoor.record_and_summarize_video())
                logStr = "Now recording..."
            elif dict["person"] == 0 and recording == True:
                recording = False
                record_task.cancel()
                logStr = "Recording finished."

            if dict["fire"] == 1:
                logStr = "Fire detected."
            if dict["flood"] == 1:
                logStr = "Flood detected."

            yield im, logStr, get_db_msg('indoor')
            await timer

with gr.Blocks() as demo:
    with gr.Row():
        input_video_indoor = gr.Video(autoplay=False, label="indoor")
        processed_frames_indoor = gr.Image(label="last frame")
        with gr.Column():
            output_log_indoor = gr.Textbox(label="log")
            db_log_indoor = gr.Textbox(label="description log")
    with gr.Row():
        input_video_outdoor = gr.Video(autoplay=False, label="outdoor")
        processed_frames_outdoor = gr.Image(label="last frame")
        with gr.Column():
            output_log_outdoor = gr.Textbox(label="log")
            db_log_outdoor = gr.Textbox(label="description log")
    with gr.Row():
        at_home_btn = gr.Radio(["At home", "Outside"])
    
    input_video_indoor.play(run_main_indoor, inputs=at_home_btn, outputs=[processed_frames_indoor, output_log_indoor, db_log_indoor])
    input_video_outdoor.play(run_main_outdoor_vectordb, inputs=None, outputs=[processed_frames_outdoor, output_log_outdoor, db_log_outdoor])
if not os.path.isfile('database/db.db'):
    init_db()
embeddings = np.load('database/embeddings.npy')
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
demo.queue()
demo.launch()