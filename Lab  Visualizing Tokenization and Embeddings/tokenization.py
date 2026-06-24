import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")


def tokenize_text(text: str):
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]

    return {
        "text": text,
        "token_count": len(tokens),
        "token_ids": tokens,
        "decoded_tokens": decoded
    }


def compare_texts(texts):
    results = []

    for t in texts:
        tokens = enc.encode(t)
        results.append((t, len(tokens), tokens))

    return results