from transformers import pipeline

image_pipe = pipeline("image-text-to-text", model="google/paligemma-3b-pt-224")
