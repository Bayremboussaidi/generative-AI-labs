from config import client


def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def embed_texts(texts):
    embeddings = []

    for t in texts:
        emb = get_embedding(t)
        embeddings.append(emb)
        print(f"{t} → {len(emb)} dimensions")

    return embeddings