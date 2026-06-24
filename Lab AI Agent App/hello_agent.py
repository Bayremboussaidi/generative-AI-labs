import os
from dotenv import load_dotenv


load_dotenv()


from groq import Groq

# Initialize client with API key from environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use a currently supported free-tier model
MODEL = "llama-3.1-8b-instant"

# Send a chat request
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Hello Agent! What can you do?"}]
)

# Print the assistant's reply
print(response.choices[0].message.content)