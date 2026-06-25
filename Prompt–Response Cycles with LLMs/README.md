#  Lab : Prompt–Response Cycles with LLMs

This project is part of an **Agentic AI learning program**.  
It demonstrates how Large Language Models (LLMs) process prompts, maintain context, and generate responses across multiple conversation turns.

---

## 🎯 Objective

Understand and experiment with:

- Single-turn prompting
- Multi-turn conversations (context memory)
- Context reset behavior
- Structured output generation (JSON-style responses)
- How LLMs simulate "memory" via message history

---

## 📁 Project Structure


Lab-2-Prompt-Response-Cycles/
│
├── cycle_lab.py # Main lab implementation
├── .env # OpenAI API key
├── README.md # Project documentation
└── requirements.txt # Python dependencies


---

## ⚙️ Setup

### 1. Create and Activate a Virtual Environment

```bash
python -m venv agentic_env

Windows:

agentic_env\Scripts\activate

Linux/macOS:

source agentic_env/bin/activate
2. Install Dependencies
pip install openai python-dotenv
3. Configure Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
