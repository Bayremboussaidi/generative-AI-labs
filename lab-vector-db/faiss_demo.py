import faiss
import numpy as np
from utils import get_embedding

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
# EMBEDDINGS
# -----------------------
X = [get_embedding(d["text"]) for d in docs]
Q = [get_embedding(q) for q in queries]

dim = len(X[0])
print("Embedding dim:", dim)

# -----------------------
# BUILD FAISS INDEX
# -----------------------
xb = np.array(X, dtype="float32")
faiss.normalize_L2(xb)

index = faiss.IndexFlatIP(dim)
index.add(xb)

print("Indexed vectors:", index.ntotal)

# -----------------------
# SEARCH FUNCTION
# -----------------------
def search(vec, k=2):
    q = np.array([vec], dtype="float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

# -----------------------
# TEST QUERIES
# -----------------------
for i, q in enumerate(Q):
    D, I = search(q)

    print("\nQuery:", queries[i])
    for score, idx in zip(D, I):
        print(docs[idx]["id"], round(float(score), 4), docs[idx]["text"])

# -----------------------
# SAVE INDEX
# -----------------------
faiss.write_index(index, "faiss.index")
print("\nFAISS index saved.")