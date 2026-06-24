from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix


def compare_to_reference(reference_embedding, embeddings):
    return cosine_similarity([reference_embedding], embeddings)[0]