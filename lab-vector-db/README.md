# Lab 2 – Vector Database Setup with FAISS and Pinecone

## Overview

This project demonstrates how to build and use vector databases for semantic search using embeddings. It covers both:

- FAISS (local vector database for fast prototyping)
- Pinecone (managed cloud vector database for production use)

The goal is to understand how embeddings are created, stored, and queried efficiently for similarity search and Retrieval-Augmented Generation (RAG).

---

## Objectives

By completing this lab, you will learn how to:

- Generate text embeddings using OpenAI models
- Build a local vector index using FAISS
- Perform similarity search on local data
- Create and manage a Pinecone vector database
- Upsert embeddings into Pinecone
- Query Pinecone for semantic search results
- Compare local vs cloud vector storage systems

---

## Technologies Used

- Python
- OpenAI API (text embeddings)
- FAISS (Facebook AI Similarity Search)
- Pinecone (vector database service)
- NumPy
- python-dotenv

---

## Project Structure

```
lab-02-vector-db/
│
├── faiss_demo.py
├── pinecone_demo.py
├── utils.py
├── .env
├── requirements.txt
└── faiss.index
```

---

## Installation

### Create virtual environment

```
python -m venv agentic_env
agentic_env\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install openai tiktoken faiss-cpu pinecone-client python-dotenv numpy
```

---

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=agentic-ai-lab
```

---

## Dataset

The project uses a small sample corpus of documents:

- Agentic AI systems and tools
- Multi-agent orchestration frameworks
- Retrieval-Augmented Generation (RAG)
- Vector database concepts
- Reasoning and planning methods

Each document is converted into embeddings for similarity search.

---

## Part 1: FAISS (Local Vector Database)

### Description

FAISS is used to build a local vector index for fast similarity search. It is suitable for development, testing, and offline use.

### Steps

1. Generate embeddings using OpenAI
2. Normalize vectors for cosine similarity
3. Build FAISS index (IndexFlatIP)
4. Add embeddings to index
5. Perform similarity search
6. Save index locally

### Run FAISS

```
python faiss_demo.py
```

### Output

- Similar documents ranked by score
- Document IDs and similarity values
- Saved FAISS index file

---

## Part 2: Pinecone (Cloud Vector Database)

### Description

Pinecone is a managed vector database used for production-scale semantic search.

### Steps

1. Initialize Pinecone client
2. Create index if not exists
3. Generate embeddings
4. Upsert vectors with metadata
5. Query Pinecone index
6. Retrieve top matches

### Run Pinecone

```
python pinecone_demo.py
```

### Output

- Stored vectors in Pinecone index
- Semantic search results
- Metadata returned with matches

---

## Key Concepts

### Embeddings

Text is converted into numerical vectors that represent semantic meaning.

### Vector Similarity

Similarity is computed using cosine similarity or inner product.

---

## FAISS vs Pinecone

| Feature     | FAISS (Local) | Pinecone (Cloud) |
|------------|---------------|------------------|
| Storage    | Local         | Cloud            |
| Scale      | Limited       | High             |
| Persistence| Manual        | Automatic        |
| Use case   | Dev/Test      | Production       |

---

## Learning Outcomes

After completing this lab, you will understand:

- How embeddings are created
- How vector search works
- How FAISS stores vectors locally
- How Pinecone manages cloud vectors
- How retrieval systems power AI applications

---

## Next Steps

- Build a RAG chatbot using FAISS or Pinecone
- Connect vector search to LangChain agents
- Add FastAPI backend for search API
- Scale dataset with metadata filtering
```
```