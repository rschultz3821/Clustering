import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.metrics import homogeneity_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#https://www.kaggle.com/datasets/insiyeah/musicfeatures
#region Find two most important features
# Load your dataset
df = pd.read_csv("music.csv")
X = df[['tempo', 'beats', 'chroma_stft', 'spectral_centroid']].to_numpy()  # Features
y = df['label'].to_numpy()  # Target

# Fit RandomForest model
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_

# Get the two most important features
important_features = sorted(zip(importances, df.columns), reverse=True)[:2]
print("Most important features:", important_features)
#endregion

# use the important features for clustering
X = df[['beats', 'chroma_stft']].to_numpy()

#region Compute elbow graph for kmeans
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(X)
    inertia[i-1] = kmeans.inertia_

#Create the elbow plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, 15), inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True, linestyle='--', alpha=0.7)

# Mark the elbow point
plt.axvline(x=3, linestyle='--', color='r', label="Elbow at K=3")
plt.legend()
plt.show()
#endregion

#region Apply KMeans clustering
kmeans = KMeans(n_clusters=3)  # You can change the number of clusters as needed
kmeans.fit(X)

# Create the plot
fig, ax = plt.subplots()
vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='white', line_width=1.0, ax=ax)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=10)

# Plot the KMeans cluster centers as red dots
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', s=100)
plt.title('KMeans Clustering with Voronoi Diagram')
plt.show()
#endregion

#region Gridsearch for eps and min_sample
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

eps_values = np.arange(0.1, 50.0, 0.5)  # Try various eps values between 0.1 and 2.0
min_samples_values = np.arange(2, 20, 1)  # Try values for min_samples between 3 and 10

# Initialize variables to store the best score and parameters
best_silhouette_score = -1
best_eps = None
best_min_samples = None

# Grid search over eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        # Apply DBSCAN with current eps and min_samples
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        # We need at least 2 clusters to calculate silhouette score (ignore -1 label)
        if len(set(dbscan_labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, dbscan_labels)

            # Update the best parameters if we have a better score
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples

# Output the best parameters and silhouette score
print(f"Silhouette Score- measures the quality of clustering\n"
      f"Best Silhouette Score: {best_silhouette_score:.4f}")
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}\n")
#endregion

# Select all features (adjust as per your dataset)
X = df[['tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
        'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13',
        'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].to_numpy()

#region Optimal DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # T

dbscan = DBSCAN(eps=1.1, min_samples=15)
dbscan_labels = dbscan.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(8, 6))

# Create a scatter plot for each cluster
for label in set(dbscan_labels):
    # Skip noise points labeled as -1
    if label == -1:
        continue
    # Get the points for the current cluster
    cluster_points = df[dbscan_labels == label]
    plt.scatter(cluster_points['beats'], cluster_points['chroma_stft'], label=f'Cluster {label}', s=10)

# Plot the noise points as a different color
noise_points = df[dbscan_labels == -1]
plt.scatter(noise_points['beats'], noise_points['chroma_stft'], color='red', label='Noise', s=10)

# Add title, labels, and legend
plt.title('DBSCAN Clustering')
plt.xlabel('beats')
plt.ylabel('chroma_stft')
plt.legend(title='Clusters', loc='upper right')
plt.show()

# # Calculate the silhouette score
# silhouette_avg = silhouette_score(X_scaled, dbscan_labels)
# print(f"Optimal DBSCAN- Silhouette Score: {silhouette_avg:.4f}\n")
#endregion

#region DBSCAN equaling number of actual genres
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # T

dbscan = DBSCAN(eps=147, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(8, 6))

# Create a scatter plot for each cluster
for label in set(dbscan_labels):
    # Skip noise points labeled as -1
    if label == -1:
        continue
    # Get the points for the current cluster
    cluster_points = df[dbscan_labels == label]
    plt.scatter(cluster_points['beats'], cluster_points['chroma_stft'], label=f'Cluster {label}', s=10)

# Plot the noise points as a different color
noise_points = df[dbscan_labels == -1]
plt.scatter(noise_points['beats'], noise_points['chroma_stft'], color='red', label='Noise', s=10)

# Add title, labels, and legend
plt.title('DBSCAN Clustering')
plt.xlabel('beats')
plt.ylabel('chroma_stft')
plt.legend(title='Clusters', loc='upper right')
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, dbscan_labels)
print(f"10 cluster DBCAN- Silhouette Score: {silhouette_avg:.4f}\n")
#endregion

#region Actual Genres
X = df[['beats', 'chroma_stft']].to_numpy()
genre_labels = df['label']
plt.figure(figsize=(8, 6))

# Scatter plot for genres
for genre in genre_labels.unique():
    genre_points = X[genre_labels == genre]
    plt.scatter(genre_points[:, 0], genre_points[:, 1], label=genre, s=10, alpha=0.7)

# Add labels and title
plt.title('Actual Data to Genres')
plt.xlabel('Beats')
plt.ylabel('Chroma_stft')

# Add legend for genres
plt.legend(title='Genres', loc='best')
plt.grid(True)
plt.show()
#endregion

#region Compare Actual Genres to DBSCAN Genres
# Encode genre labels as integers
le = LabelEncoder()
true_labels = le.fit_transform(df['label'])  # converts genre strings to numeric labels
predicted_labels = dbscan_labels

# Compute metrics
ari = adjusted_rand_score(true_labels, predicted_labels)
homogeneity = homogeneity_score(true_labels, predicted_labels)

print(f"Adjusted Rand Index: {ari:.4f}\n"
      f"ARI measures how similar two clusters are; clusters don't align well with the actual genre labels\n")
print(f"Homogeneity Score: {homogeneity:.4f}\n"
      f"Homogeneity Score measures how pure each cluster is with the actual; clusters contain mixed genres")
#endregion
