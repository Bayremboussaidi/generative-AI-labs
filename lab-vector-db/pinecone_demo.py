import os
import time
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from utils import get_embedding

load_dotenv()

docs = [
    {"id":"d1","text":"Agentic AI agents use tools, memory, and goals to act."},
    {"id":"d2","text":"LangChain and CrewAI help orchestrate multi-agent workflows."},
    {"id":"d3","text":"RAG retrieves external knowledge to improve answer accuracy."},
    {"id":"d4","text":"Vector databases enable fast similarity search over embeddings."},
    {"id":"d5","text":"Planning loops and ReAct improve reasoning in complex tasks."},
]

queries = [
    "How do agents use memory?",
    "Name a framework for multi-agent orchestration.",
    "Why is RAG useful?"
]

# -----------------------
# INIT PINECONE
# -----------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX")

# create index if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)

index = pc.Index(index_name)

# -----------------------
# EMBEDDINGS
# -----------------------
vectors = []
for d in docs:
    emb = get_embedding(d["text"])
    vectors.append({
        "id": d["id"],
        "values": emb,
        "metadata": {"text": d["text"]}
    })

# -----------------------
# UPSERT
# -----------------------
index.upsert(vectors=vectors)
print("Data upserted to Pinecone")

# -----------------------
# QUERY
# -----------------------
def search(vec):
    return index.query(
        vector=vec,
        top_k=2,
        include_metadata=True
    )

for q in queries:
    qv = get_embedding(q)
    res = search(qv)

    print("\nQuery:", q)

    for m in res["matches"]:
        print(m["id"], round(m["score"], 4), m["metadata"]["text"])