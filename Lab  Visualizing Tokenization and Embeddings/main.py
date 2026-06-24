from tokenization import tokenize_text, compare_texts
from embeddings import embed_texts, get_embedding
from visualization import plot_embeddings
from similarity import compute_similarity

# -------------------------
# 1. TOKENIZATION DEMO
# -------------------------
texts = [
    "Agentic AI",
    "Autonomous agents",
    "Bananas are yellow"
]

print("\n=== TOKENIZATION ===")
for t in texts:
    result = tokenize_text(t)
    print(result["text"], "→", result["token_count"], "tokens")

# -------------------------
# 2. EMBEDDINGS
# -------------------------
print("\n=== EMBEDDINGS ===")
embeddings = embed_texts(texts)

# -------------------------
# 3. VISUALIZATION
# -------------------------
print("\n=== VISUALIZATION ===")
plot_embeddings(texts, embeddings)

# -------------------------
# 4. SIMILARITY
# -------------------------
print("\n=== SIMILARITY ===")
sim = compute_similarity(embeddings)

print(sim)