import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import silhouette_score

# https://www.kaggle.com/datasets/kandij/mall-customers/data
#region Read data
df = pd.read_csv("Mall_Customers.csv")
df = df.dropna()

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].to_numpy()
#endregion

#region --- KMeans: Finding optimal k using Elbow method ---
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    inertia[i-1] = kmeans.inertia_

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, 15), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.grid(True)
plt.show()
#endregion

#region --- KMeans Clustering ---
plt.style.use('dark_background')
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Plot with diagram
fig, ax = plt.subplots()
vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='white', line_width=1.0, ax=ax)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('KMeans Clustering')
plt.show()
#endregion

#region --- Grid Search for Best Epsilon and min_samples ---
# Define the range of eps and min_samples values to try
eps_range = np.arange(0.1, 10.1, 0.1)
min_samples_range = [3, 5, 10, 15, 20]  # You can adjust these values based on your data
best_eps = None
best_min_samples = None
best_score = -1
scores = []

# Grid search for both eps and min_samples
for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Evaluate the clustering if more than one cluster is found (not including noise)
        if len(set(labels)) > 1 and len(set(labels) - {-1}) >= 2:
            score = silhouette_score(X, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
        else:
            scores.append(-1)

# Output best DBSCAN parameters
print("-" * 20, " DBSCAN Grid Search Results ", "-" * 20)
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
print(f"    DBSCAN considers any two points within a distance of {best_eps} units as neighbors.")
print(f"Best silhouette score: {best_score:.4f}")
print("     DBSCAN has created clusters with some reasonable separation, but it's not perfect.")
#endregion

#region --- Run DBSCAN with Best Epsilon ---
dbscan = DBSCAN(eps=best_eps, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X)

# Create a figure
plt.figure(figsize=(8, 6))

# Plot points, coloring them according to the cluster
plt.scatter(X[:, 0], X[:, 1], c=clusters_dbscan, cmap='viridis', s=8)

# Highlight noise points (label = -1) in red
plt.scatter(X[clusters_dbscan == -1, 0], X[clusters_dbscan == -1, 1], c='red', s=8, label="Noise", alpha=0.7)

# Set plot labels and title
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')

# Display grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
#endregion

# --- Comparison Summary ---
print("-" * 20, " Comparison Summary ", "-" * 20)
print(f"KMeans: 5 clusters, inertia = {kmeans.inertia_:.2f}")
print("     Clusters in KMeans aren't too tightly packed.")
print(f"DBSCAN: Best eps = {best_eps}, silhouette = {best_score:.2f}")
print("     Moderate clustering quality, DBSCAN detects noise- handles outliers better than KMeans.")

