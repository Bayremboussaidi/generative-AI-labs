import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def chat_cycle(messages):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return resp.choices[0].message.content



messages = [
    {
        "role": "user",
        "content": "Explain Agentic AI in one sentence."
    }
]

print(chat_cycle(messages))