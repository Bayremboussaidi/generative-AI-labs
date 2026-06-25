import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

EMB_MODEL = "text-embedding-3-small"

def get_embedding(text: str):
    return client.embeddings.create(
        model=EMB_MODEL,
        input=text
    ).data[0].embedding