import pandas as pd
import numpy as np
from analysis import kmeans, cosine_distance, plot_clusters, plot_elbow_curve, recommend_movies, evaluate_clusters, evaluate_recommendations_manual

# Load CSVs
ratings = pd.read_csv('dataset/ratings_small.csv')
movies = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
links = pd.read_csv('dataset/links_small.csv')

# Convert timestamp and drop duplicates
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])

# Clean IDs for movies and links
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')

# Merge ratings with links to get tmdbId
ratings_links = ratings.merge(links, how='left', on='movieId')

# Merge ratings+links with movies_metadata on tmdbId = id
ratings_merged = ratings_links.merge(movies, how='left', left_on='tmdbId', right_on='id')

# Keep necessary columns
ratings_merged = ratings_merged[['userId', 'movieId', 'rating', 'title', 'genres', 'original_language', 'release_date']]

# Filter down to most active users and most rated movies
top_users = ratings['userId'].value_counts().head(500).index
top_movies = ratings['movieId'].value_counts().head(500).index
filtered_ratings = ratings[ratings['userId'].isin(top_users) & ratings['movieId'].isin(top_movies)]

# Create user-movie matrix
user_movie_matrix = filtered_ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# Fill missing values (NaN) with user average
user_movie_matrix_filled = user_movie_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

# K-means clustering
X = user_movie_matrix_filled.values
labels, centroids = kmeans(X, k=3)

# Optional: plot clusters
# plot_clusters(X, labels, centroids)

# Optional: plot elbow curve
# plot_elbow_curve(X)

# After doing K-means:
intra, inter = evaluate_clusters(X, labels)

# To evaluate recommendation system:
precision, recall = evaluate_recommendations_manual(ratings_merged, labels, user_movie_matrix)


print("Enter 3 movies you like:")
movie1 = input("Movie 1: ")
movie2 = input("Movie 2: ")
movie3 = input("Movie 3: ")

movie_list = [movie1, movie2, movie3]

recommendations = recommend_movies(movie_list, ratings_merged, labels, user_movie_matrix)

print("\nBecause you liked those movies, you might also like:")
for rec in recommendations:
    print(f"- {rec}")
