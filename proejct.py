import pandas as pd
import numpy as np
from analysis import kmeans, cosine_distance, plot_clusters, plot_elbow_curve

# Load CSVs
ratings = pd.read_csv('dataset/ratings_small.csv')
movies = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)

# Convert timestamp adn drop duplicates
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])

# Clean movie IDs
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)

# Merge ratings with movies metadata
ratings_merged = ratings.merge(movies, how='left', left_on='movieId', right_on='id')
ratings_merged = ratings_merged[['userId', 'movieId', 'rating', 'title', 'genres']]

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

# Plot clusters in 2D
plot_clusters(X, labels, centroids)

#plot elbow curve
plot_elbow_curve(X)
