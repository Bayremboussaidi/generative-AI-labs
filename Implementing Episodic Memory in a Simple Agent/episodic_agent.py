import time, uuid
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMB_MODEL = "text-embedding-3-small"

chroma = chromadb.PersistentClient(path="./chroma_db")
episodes = chroma.get_or_create_collection(name="episodic_memory")


def embed(text: str):
    return client.embeddings.create(
        model=EMB_MODEL,
        input=text
    ).data[0].embedding


def now_ts():
    return int(time.time())


def ts_to_str(ts: int):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def add_episode(summary: str, who: str = "user", tags=None):
    eid = str(uuid.uuid4())

    meta = {
        "who": who,
        "summary": summary,
        "ts": now_ts(),
        "tags": ",".join(tags) if tags else ""
    }

    episodes.add(
        ids=[eid],
        embeddings=[embed(summary)],
        metadatas=[meta],
        documents=[summary]
    )


def search_episodes(query: str, k: int = 3):
    qemb = embed(query)

    res = episodes.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["metadatas", "documents"]
    )

    if not res["ids"] or len(res["ids"][0]) == 0:
        return []

    results = []

    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]

        results.append({
            "summary": doc,
            "who": meta["who"],
            "when": ts_to_str(meta["ts"])
        })

    return results


def llm(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def agent(user_text: str):
    memories = search_episodes(user_text, k=3)

    memory_text = "\n".join(
        [f"- ({m['when']}) {m['summary']}" for m in memories]
    ) if memories else "None"

    prompt = f"""
You are an AI assistant with memory.

Relevant past episodes:
{memory_text}

User question:
{user_text}
"""

    answer = llm(prompt)

    add_episode(
        f"Q: {user_text} | A: {answer[:200]}",
        who="agent",
        tags=["dialog"]
    )

    return answer, memories


# --- SEED DATA ---
add_episode("User likes Python and builds AI agents.", who="user", tags=["pref"])
add_episode("We used ChromaDB for vector storage.", who="agent", tags=["tech"])
add_episode("User asked about LangChain vs custom agents.", who="user", tags=["topic"])


# --- TEST ---
answer, mems = agent("What tools did I use for memory and what do I like?")

print("\nANSWER:\n", answer)

print("\nMEMORIES USED:")
for m in mems:
    print(m)