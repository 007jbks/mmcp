import logging
from typing_extensions import Text
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import audio
import pytesseract
from typing import Annotated, Optional
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import uvicorn
import torch
import shutil
import tempfile


processor = None
model = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    logging.warning("MPS is not available")
    device = torch.device("cpu")

try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.to(device)
except Exception as e:
    logging.warning(
        "Unable to import the image captioning model only text in image will be provided"
    )

classifer = None

try:
    classifer = pipeline(
        "audio-classification",
        model="anton-l/xtreme_s_xlsr_300m_minds14",
    )

except Exception as e:
    logging.warning(
        "Couldn't import audio background identifier only audio to text will be available"
    )


app = FastAPI()


@app.post("/send")
async def send(text: str | None, image: UploadFile | None, aud: UploadFile | None):
    context = ""
    logging.warning("started")
    try:
        if text:
            context += f"Text:{text}"
        if aud is not None:
            logging.warning("started processing audio")
            try:
                audtext: str
                audtext = audio.convert(aud.file)
                context += f" Audio:{audtext}"
                if classifer:
                    try:
                        aud.file.seek(0)
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as tmp:
                            shutil.copyfileobj(aud.file, tmp)
                            tmp_path = tmp.name
                        result = classifer(tmp_path)
                        logging.warning(result[0]["label"])
                        context += f" Background sound:{result[0]['label']}"
                    except Exception as e:
                        logging.warning(
                            f"Error is {e} while processing background noise"
                        )
            except Exception as e:
                logging.warning(f"Error:{e}")
        if image is not None:
            logging.warning("started processing image")
            try:
                imgtext: str
                imgtext = pytesseract.image_to_string(Image.open(image.file))
                context += f" Image:{imgtext}"
                if processor and model:
                    image = Image.open(image.file)
                    inputs = processor(image, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    context += f" Image caption:{caption}"
            except Exception as e:
                logging.warning(f"Error:{e}")
    except Exception as e:
        logging.warning(f"Error:{e}")
    return {"Context": context}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
