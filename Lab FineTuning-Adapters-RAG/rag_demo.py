import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

print("\n===== RAG DEMO =====\n")

# Knowledge base
kb = [
    "Agentic AI agents use memory, tools, and goals to act.",
    "LangChain and CrewAI are popular frameworks for building AI agents.",
    "Retrieval-Augmented Generation improves accuracy by fetching external knowledge."
]

questions = [
    "What are the key components of Agentic AI?",
    "Name one framework for AI agents.",
    "How does RAG improve answers?"
]

# Convert KB into documents
docs = [Document(page_content=x) for x in kb]

# Embeddings
embeddings = OpenAIEmbeddings()

# Vector DB
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

# LLM
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever
)

# Ask questions
for q in questions:
    print("\nQ:", q)
    print("A:", qa.run(q))

print("\n👉 RAG retrieves relevant knowledge before answering.")
print("👉 No training required.")