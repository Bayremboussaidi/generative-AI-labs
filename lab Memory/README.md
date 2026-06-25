# Lab - Building Short-Term and Long-Term Memory Modules

## Goal

This lab demonstrates how AI agents can use:

- Short-Term Memory (conversation context)
- Long-Term Memory (persistent vector database)
- Retrieval-Augmented Generation (RAG)

---

## Technologies

- Python
- OpenAI
- LangChain
- ChromaDB

---

## Installation

Create a virtual environment:

```bash
python -m venv agentic_env
```

Activate it:

Windows

```bash
agentic_env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```text
OPENAI_API_KEY=
```

Run

```bash
python main.py
```

---

## Project Demonstration

### Short-Term Memory

The agent remembers information during the conversation.

Example:

```
User:
My name is Alex.

AI:
Hello Alex.

User:
What is my name?

AI:
Your name is Alex.
```

---

### Long-Term Memory

Knowledge is stored in ChromaDB.

Stored facts include:

- Agentic AI agents use tools and memory.
- LangChain helps build autonomous agents.
- RAG improves accuracy.
- CrewAI enables multi-agent orchestration.

The agent retrieves relevant documents before answering.

---

### Combined Memory

The final demo combines:

- Conversation history
- Retrieval from the vector database

to generate better responses.

---


