import cv2
import torch
import logging
import os
import tempfile
import shutil
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "mps"
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


def caption(image):
    """Generates a caption for a given image."""
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption_text = processor.decode(out[0], skip_special_tokens=True)
    return f" {caption_text}"


def process_video(video_file, output_dir="video/video0"):
    temp_file_path = None
    if hasattr(video_file, "read"):  # Check if it's a file-like object
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video_file, tmp)
            temp_file_path = tmp.name
        video_path = temp_file_path
    else:
        video_path = video_file

    if not os.path.exists(video_path):
        if temp_file_path:
            os.remove(temp_file_path)
        return f"Error: Video file not found at {video_path}"

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 10))

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    frame_count = 0
    success = True
    while success:
        success, image = vid.read()
        if frame_count % frame_interval == 0 and success:
            cv2.imwrite(f"{output_dir}/frame{count}.jpg", image)
            count += 1
        frame_count += 1

    print(f"Number of frames extracted: {count}")
    vid.release()

    temporal_info = ""
    for i in range(count):
        image_path = f"{output_dir}/frame{i}.jpg"
        img = cv2.imread(image_path)
        if img is not None:
            temporal_info += f"frame{i} : {caption(img)}\n"

    if temp_file_path:
        os.remove(temp_file_path)  # Clean up the temporary file

    return temporal_info


if __name__ == "__main__":
    video_file = "path for mock"
    all_captions = process_video(video_file)

    if "Error" not in all_captions:
        print("\n--- Generated Captions ---")
        print(all_captions)

        with open("record.txt", "w+") as f:
            f.write(all_captions)
        print("\nCaptions saved to record.txt")
