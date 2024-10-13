import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def generate_elbow_plot(umap_embeddings, elbow_plot_path, seed):
    print(elbow_plot_path)
    inertias = []
    k_values = list(range(2, 32))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(umap_embeddings)
        inertias.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, "bo-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(elbow_plot_path)
    plt.show()
