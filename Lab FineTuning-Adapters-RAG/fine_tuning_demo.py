from datasets import Dataset

print("\n===== FINE-TUNING (CONCEPT DEMO) =====\n")

train_data = Dataset.from_dict({
    "prompt": [
        "Q: What are the key components of Agentic AI?\nA:",
        "Q: Name one framework for AI agents.\nA:",
        "Q: How does RAG improve answers?\nA:"
    ],
    "completion": [
        "Agentic AI agents use memory, tools, and goals to act.",
        "LangChain is a framework for building AI agents.",
        "RAG improves accuracy by fetching external knowledge before answering."
    ]
})

print(train_data)

print("\n👉 Fine-tuning means updating model weights using this dataset.")
print("👉 It permanently changes model behavior.")