import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# -------------------------------------------------
# Load API Key
# -------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------------------------
# SHORT TERM MEMORY
# -------------------------------------------------

def chat_with_memory(messages):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# -------------------------------------------------
# LONG TERM MEMORY
# -------------------------------------------------

embedding = OpenAIEmbeddings()

persist_directory = "./data/chroma_db"

docs = [

    Document(
        page_content="Agentic AI agents use tools and memory."
    ),

    Document(
        page_content="LangChain helps build autonomous agents."
    ),

    Document(
        page_content="RAG improves accuracy by retrieving external knowledge."
    ),

    Document(
        page_content="CrewAI enables multi-agent orchestration."
    ),

    Document(
        page_content="Vector databases store semantic embeddings."
    )

]

if not os.path.exists(persist_directory):

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory
    )

else:

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )


def recall(query):

    retriever = db.as_retriever()

    results = retriever.invoke(query)

    return [doc.page_content for doc in results]


# -------------------------------------------------
# RETRIEVAL QA
# -------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# -------------------------------------------------
# DEMO 1
# SHORT TERM MEMORY
# -------------------------------------------------

print("=" * 60)
print("SHORT TERM MEMORY")
print("=" * 60)

conversation = [

    {
        "role": "system",
        "content": "You are a helpful tutor."
    },

    {
        "role": "user",
        "content": "My name is Alex."
    }

]

reply = chat_with_memory(conversation)

conversation.append(
    {
        "role": "assistant",
        "content": reply
    }
)

conversation.append(
    {
        "role": "user",
        "content": "What is my name?"
    }
)

answer = chat_with_memory(conversation)

print("\nUser: What is my name?")
print("AI:", answer)

# -------------------------------------------------
# DEMO 2
# LONG TERM MEMORY
# -------------------------------------------------

print("\n")
print("=" * 60)
print("LONG TERM MEMORY")
print("=" * 60)

results = recall("How do agents use memory?")

for item in results:
    print("-", item)

# -------------------------------------------------
# DEMO 3
# COMBINED MEMORY
# -------------------------------------------------

print("\n")
print("=" * 60)
print("SHORT + LONG TERM")
print("=" * 60)

conversation = [

    {
        "role": "system",
        "content": "You are a teaching AI assistant."
    },

    {
        "role": "user",
        "content": "Remember my favorite framework is LangChain."
    },

    {
        "role": "assistant",
        "content": "Okay! I will remember that your favorite framework is LangChain."
    }

]

conversation.append(
    {
        "role": "user",
        "content": "What is my favorite framework?"
    }
)

short_answer = chat_with_memory(conversation)

print("\nShort-term Answer:")
print(short_answer)

print("\nLong-term Answer:")

print(
    qa.invoke(
        {
            "query": "How do agents use memory?"
        }
    )["result"]
)

print("\nDone!")