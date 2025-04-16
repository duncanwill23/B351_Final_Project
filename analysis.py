import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# K-means function
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Randomly choose k users as initial centroids
    np.random.seed(42)
    initial_idxs = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_idxs]

    for _ in range(max_iters):
        # Assign users to nearest centroid using cosine distance
        distances = cosine_distance(X, centroids)
        cluster_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids

    return cluster_labels, centroids

# Function to calculate cosine distance
def cosine_distance(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = np.dot(a_norm, b_norm.T)  # shape (n_samples, n_clusters)
    return 1 - sim

# Function to calculate Euclidean distance
def eucledian_distance(a, b):
    # Calculate the Euclidean distance between two arrays
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Function to plot clusters
def plot_clusters(X, labels, centroids):
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))

    # Scatter plot for each user
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)

    # Plot centroids
    centroids_2D = pca.transform(centroids)
    plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], c='red', marker='X', s=200, label='Centroids')

    # Add labels and title
    plt.title('User Clusters in 2D')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # Show the plot
    plt.show()

# Function to generate the Elbow Curve (elbow method for finding optimal K)
def plot_elbow_curve(X):
    inertia_values = []

    # Try different values of K
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.plot(K_range, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()
