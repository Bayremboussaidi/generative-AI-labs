import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embeddings(texts, embeddings):
    pca = PCA(n_components=2)
    points = pca.fit_transform(embeddings)

    plt.figure(figsize=(6, 6))

    for i, txt in enumerate(texts):
        plt.scatter(points[i, 0], points[i, 1])
        plt.text(points[i, 0] + 0.02, points[i, 1] + 0.02, txt)

    plt.title("Embedding Visualization (PCA 2D)")
    plt.show()