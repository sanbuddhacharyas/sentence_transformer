from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def pca_visualization(embeddings, sentences, n_components=2):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Alternative: Use t-SNE instead of PCA
    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    # embeddings_2d = tsne.fit_transform(embeddings)

    # Generate unique colors and markers
    colors = plt.cm.get_cmap("tab10", len(sentences))
    markers = ['o', 's', 'D', '^', 'v', 'x', 'd', 'D']  # Different marker shapes

    # Create the plot
    plt.figure(figsize=(8, 6))

    for i, sentence in enumerate(sentences):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                    color=colors(i), marker=markers[i], label=sentence)
        plt.annotate(sentence, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)


    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Sentence Embeddings Visualization")
    plt.grid()
    plt.savefig('embedding_space.png')
    plt.show()
