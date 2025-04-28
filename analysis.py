import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

        # Check convergence condition
        # If centroids do not change significantly, break the loop
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


def evaluate_clusters(X, labels):
    """
    Manually evaluate clustering quality: intra-cluster vs inter-cluster distances.
    
    Args:
        X: Feature matrix (user-movie ratings filled)
        labels: K-Means cluster labels

    Returns:
        intra_score: average distance between points in same cluster
        inter_score: average distance between cluster centroids
    """

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Compute centroids manually
    centroids = []
    for cluster_id in unique_labels:
        cluster_points = X[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Intra-cluster distance: avg distance of points to their own centroid
    intra_distances = []
    for cluster_id in unique_labels:
        cluster_points = X[labels == cluster_id]
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_distances.extend(distances)
    intra_score = np.mean(intra_distances)

    # Inter-cluster distance: distance between all pairs of centroids
    inter_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)
    inter_score = np.mean(inter_distances)

    print(f"Average Intra-cluster Distance (lower is better): {intra_score:.4f}")
    print(f"Average Inter-cluster Distance (higher is better): {inter_score:.4f}")

    return intra_score, inter_score

def evaluate_recommendations_manual(ratings_merged, labels, user_movie_matrix, top_n=5):
    """
    Manually evaluate recommendation precision and recall.
    
    Args:
        ratings_merged: ratings + movie info
        labels: K-Means cluster labels
        user_movie_matrix: user-movie ratings pivot
        top_n: number of recommendations

    Returns:
        avg_precision, avg_recall
    """

    users = user_movie_matrix.index
    precision_list = []
    recall_list = []

    for user in users:
        user_ratings = ratings_merged[ratings_merged['userId'] == user]
        liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['title'].unique()

        if len(liked_movies) < 5:
            continue  # Skip users with too few ratings

        # Manually split liked movies: 80% train, 20% test
        liked_movies = np.array(liked_movies)
        np.random.shuffle(liked_movies)
        split_idx = int(0.8 * len(liked_movies))
        train_movies = liked_movies[:split_idx]
        test_movies = liked_movies[split_idx:]

        # Simulate recommendations
        simulated_ratings = ratings_merged[
            (ratings_merged['userId'] == user) &
            (ratings_merged['title'].isin(train_movies))
        ]

        cluster = labels[list(user_movie_matrix.index).index(user)]
        users_in_same_cluster = [u for u, c in zip(user_movie_matrix.index, labels) if c == cluster]

        cluster_ratings = ratings_merged[
            (ratings_merged['userId'].isin(users_in_same_cluster)) &
            (~ratings_merged['title'].isin(train_movies))
        ]

        recommended = cluster_ratings['title'].value_counts().head(top_n).index.tolist()

        # Precision and Recall manually
        hits = len(set(recommended) & set(test_movies))
        precision = hits / top_n
        recall = hits / len(test_movies)

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    return avg_precision, avg_recall


def recommend_movies(movie_titles, ratings_merged, labels, user_movie_matrix, top_n=5):
    # Step 1: Find movies that match any title
    matching_movies = ratings_merged[
        ratings_merged['title'].str.lower().isin([title.lower() for title in movie_titles])
    ]

    if matching_movies.empty:
        print(f"No movies found matching your selections: {movie_titles}")
        return []

    movie_ids = matching_movies['movieId'].unique()

    # Extract common genres and language
    preferred_genres = []
    preferred_languages = []
    preferred_years = []

    for idx, row in matching_movies.iterrows():
        genres = row['genres']
        if isinstance(genres, str):
            try:
                genres = eval(genres)
            except:
                genres = []
        if isinstance(genres, list):
            preferred_genres += [genre['name'] for genre in genres]

        if 'original_language' in row and not pd.isna(row['original_language']):
            preferred_languages.append(row['original_language'])

        if not pd.isna(row['release_date']):
            try:
                preferred_years.append(int(str(row['release_date'])[:4]))
            except:
                continue

    # Mode values: most common genre, language, and year
    from collections import Counter

    preferred_genre_names = [item for item, count in Counter(preferred_genres).most_common(5)]
    preferred_language = Counter(preferred_languages).most_common(1)[0][0] if preferred_languages else 'en'
    preferred_year = int(sum(preferred_years) / len(preferred_years)) if preferred_years else None

    # Step 2: Find users who liked any of these movies highly
    users_who_like = ratings_merged[
        (ratings_merged['movieId'].isin(movie_ids)) & 
        (ratings_merged['rating'] >= 4.0)
    ]['userId'].unique()

    if len(users_who_like) == 0:
        print(f"No users rated your selected movies highly.")
        return []

    # Step 3: Find their clusters
    user_to_cluster = dict(zip(user_movie_matrix.index, labels))
    liked_clusters = [user_to_cluster[u] for u in users_who_like if u in user_to_cluster]

    if not liked_clusters:
        print("No cluster info for users who liked the movies.")
        return []

    target_clusters = set(liked_clusters)
    users_in_same_clusters = [user for user, cluster in user_to_cluster.items() if cluster in target_clusters]

    # Step 4: Find other movies these users liked
    other_movies = ratings_merged[
        (ratings_merged['userId'].isin(users_in_same_clusters)) & 
        (ratings_merged['rating'] >= 4.0) & 
        (~ratings_merged['movieId'].isin(movie_ids))
    ]

    if other_movies.empty:
        print("No other liked movies found in the same cluster.")
        return []

    # Step 5: Filter recommendations
    filtered_recommendations = []

    for idx, row in other_movies.iterrows():
        movie_genres = row['genres']
        if isinstance(movie_genres, str):
            try:
                movie_genres = eval(movie_genres)
            except:
                movie_genres = []
        if isinstance(movie_genres, list):
            movie_genre_names = [genre['name'] for genre in movie_genres]
        else:
            movie_genre_names = []

        language_match = True
        if 'original_language' in row:
            language_match = (row['original_language'] == preferred_language)

        year_match = True
        if preferred_year is not None and not pd.isna(row['release_date']):
            try:
                movie_year = int(str(row['release_date'])[:4])
                year_match = abs(movie_year - preferred_year) <= 5
            except:
                year_match = True

        genre_overlap = any(genre in preferred_genre_names for genre in movie_genre_names)

        if genre_overlap and language_match and year_match:
            filtered_recommendations.append(row['title'])

    # Step 6: Most popular ones
    filtered_recommendations = pd.Series(filtered_recommendations).value_counts().head(top_n).index.tolist()

    return filtered_recommendations
