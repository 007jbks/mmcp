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
            logging.info("Started processing audio")
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    shutil.copyfileobj(aud.file, tmp)
                    tmp_path = tmp.name

                audtext = audio.convert(tmp_path)
                context += f" Audio:{audtext}"
                if classifier:
                    result = classifier(tmp_path)
                    logging.info(f"Audio classification result: {result[0]['label']}")
                    context += f" Background sound:{result[0]['label']}"
            except Exception as e:
                logging.error(f"Error processing audio: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)  # Clean up the temp file

        if image is not None:
            logging.info("Started processing image")
            try:
                img = Image.open(image.file)
                imgtext = pytesseract.image_to_string(img)
                context += f" Image:{imgtext}"
                if processor and model:
                    inputs = processor(img, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    context += f" Image caption:{caption}"
            except Exception as e:
                logging.error(f"Error processing image: {e}")

        if vid is not video:
            logging.info("Started processing video")
            try:
                video_captions = process_video(vid.file)
                context += f" Video Content:{video_captions}"
            except Exception as e:
                logging.error(f"Error processing video: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    response = receiver(context)
    logging.info(f"Final context for AI: {context}")
    return {"AI response": response}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5001, log_level="info")
    server = uvicorn.Server(config)
    server.run()
