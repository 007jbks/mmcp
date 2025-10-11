from google import genai

from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.environ["Gem_api1"])


def receiver(cont: str) -> str:
    context = f"Give your response based on the given context through multimodal service the context is:{cont} do not repeat the context just answer the ques asked in the context the answer might not be in the context you can answer the ques based on your own knowledge"
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=context
    )
    return resp.text
