import numpy as np
import matplotlib.pyplot as plt

# K-means function
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Randomly choose k users as initial centroids
    np.random.seed(42)
    initial_idxs = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_idxs]

    for _ in range(max_iters):
        # Assign users to nearest centroid using Euclidean distance
        distances = eucledian_distance(X, centroids)
        cluster_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids

    return cluster_labels, centroids

# # Function to calculate cosine distance
def cosine_distance(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = np.dot(a_norm, b_norm.T)  # shape (n_samples, n_clusters)
    return 1 - sim


# Function to calculate Euclidean distance
def eucledian_distance(a, b):
    # Broadcasting to calculate distances
    a_expanded = a[:, np.newaxis, :]  # shape (n_samples, 1, n_features)
    b_expanded = b[np.newaxis, :, :]  # shape (1, n_clusters, n_features)
    distances = np.linalg.norm(a_expanded - b_expanded, axis=2)  # shape (n_samples, n_clusters)
    return distances

# Function to manually perform PCA (2D projection) using SVD
def manual_pca(X, n_components=2):
    # Center the data by subtracting the mean of each feature
    X_centered = X - np.mean(X, axis=0)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Select the first n_components principal components
    return U[:, :n_components] @ np.diag(S[:n_components])

# Function to plot clusters
def plot_clusters(X, labels, centroids):
    # Reduce dimensions to 2D using manual PCA
    X_2D = manual_pca(X)

    plt.figure(figsize=(10, 6))

    # Scatter plot for each user
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)

    # Manually project centroids to 2D using PCA
    centroids_2D = manual_pca(centroids)

    # Plot centroids
    plt.scatter(centroids_2D[:, 0], centroids_2D[:, 1], c='red', marker='X', s=200, label='Centroids')

    # Add labels and title
    plt.title('User Clusters in 2D (Without sklearn PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # Show the plot
    plt.show()

# Example to generate Elbow Curve (without using sklearn)
def plot_elbow_curve(X):
    inertia_values = []

    # Try different values of K
    K_range = range(1, 11)  # Test K from 1 to 10
    for k in K_range:
        # Perform K-means for each K and calculate inertia
        labels, centroids = kmeans(X, k)
        distances = eucledian_distance(X, centroids)
        inertia = np.sum(np.min(distances, axis=1))  # Calculate inertia (sum of distances to centroids)
        inertia_values.append(inertia)

    # Plot the Elbow Curve
    plt.plot(K_range, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()