# Lab 3 – Episodic Memory AI Agent

## Overview

I built this lab to demonstrate how AI systems can “remember” past information using vector databases and embeddings.
It shows how previous user data can be stored and reused later to improve answers.
The goal is to simulate real AI memory, like modern chatbots and assistants.
This helps understand how AI connects old information with new questions.

the main concept "Stored data is converted into embeddings, which are long vectors of numbers (e.g. 1536 values) representing the meaning of the text.
A vector database stores these embeddings and compares them using distance/similarity to find which past data is closest to the new question.
The system then uses these closest matches as context for GPT to generate a predicted answer based on related past information."


## How to Run

### 1. Install dependencies

pip install openai chromadb python-dotenv


and dont forget to enter your api key  in .env file