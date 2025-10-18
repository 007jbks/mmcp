import logging
import os
import shutil
import tempfile
from typing import Optional

import audio
import pytesseract
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from test import receiver
from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline
from video import process_video

# Configure logging to ensure all messages are displayed in the terminal
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

processor = None
model = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    logging.warning("MPS and CUDA not available")
    device = torch.device("cpu")

try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.to(device)
except Exception as e:
    logging.warning(
        "Unable to import the image captioning model; only text in image will be provided."
    )

classifier = None
try:
    classifier = pipeline(
        "audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14"
    )
except Exception as e:
    logging.warning(
        "Couldn't import audio background identifier; only audio to text will be available."
    )

app = FastAPI()


def process_audio_file(audio_path: str) -> str:
    audio_context = ""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        audtext = audio.convert(audio_path)
        audio_context += f" Audio Transcript:{audtext}"
        if classifier:
            result = classifier(audio_path)
            logging.info(f"Audio classification result: {result[0]['label']}")
            audio_context += f" Background sound:{result[0]['label']}"
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {e}")
    return audio_context


@app.post("/send")
async def send(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    aud: Optional[UploadFile] = File(None),
    vid: Optional[UploadFile] = File(None),
):
    context = ""
    receiver("")
    logging.info("Request received, starting processing.")
    try:
        if text:
            context += f"Text:{text}"

        if aud is not None:
            logging.info("Started processing direct audio upload")
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    shutil.copyfileobj(aud.file, tmp)
                    tmp_path = tmp.name
                context += process_audio_file(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if image is not None:
            logging.info("Started processing image")
            try:
                img = Image.open(image.file)
                imgtext = pytesseract.image_to_string(img)
                context += f" Image Text:{imgtext}"
                if processor and model:
                    inputs = processor(img, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    context += f" Image caption:{caption}"
            except Exception as e:
                logging.error(f"Error processing image: {e}")

        if vid is not None:
            logging.info("Started processing video")
            video_audio_path = None
            try:
                video_captions, video_audio_path = process_video(vid.file)
                context += f" Video Visual Content:{video_captions}"
                if video_audio_path:
                    logging.info("Processing audio extracted from video")
                    context += "Video's audio :" + process_audio_file(video_audio_path)
            except Exception as e:
                logging.error(f"Error processing video: {e}")
            finally:
                if video_audio_path and os.path.exists(video_audio_path):
                    os.remove(video_audio_path)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    response = receiver(context)
    logging.info(f"Final context for AI: {context}")
    return {"AI response": response}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5001, log_level="info")
    server = uvicorn.Server(config)
    server.run()
