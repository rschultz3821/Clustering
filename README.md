# Clustering and Classification Projects
## Overview
This repository contains several machine learning projects focused on clustering and classification, utilizing popular algorithms such as KMeans, DBSCAN, and Random Forest. The main goal of these projects is to explore unsupervised learning methods and evaluate their performance on different datasets.

## Projects
 ### 1. Mall Customers Clustering (KMeans and DBSCAN)
This project demonstrates clustering of mall customers based on their annual income and spending score using two popular clustering algorithms: KMeans and DBSCAN. The goal is to find meaningful patterns in customer behavior for potential business insights.

#### Dataset: Mall Customers Dataset

#### Methods Used:

KMeans Clustering with Elbow Method to find optimal number of clusters

DBSCAN Clustering with Grid Search for optimal parameters (eps and min_samples)

#### Key Features: Annual Income (k$), Spending Score (1-100)

#### Results:

KMeans with 5 clusters and visualized with a Voronoi diagram.
![image](https://github.com/user-attachments/assets/0967b3b8-3566-42c8-adc4-e17e5882a0fc)
![image](https://github.com/user-attachments/assets/bba54a85-9839-4af0-bddb-6cc3da042d3e)

DBSCAN with the best eps and min_samples values based on silhouette score.
![image](https://github.com/user-attachments/assets/293e9251-311e-443c-adc3-05c8c487a1a7)
![image](https://github.com/user-attachments/assets/964a3c21-efce-410e-9c9e-393a0cc2a494)


### 2. Music Genre Clustering (KMeans and DBSCAN)
In this project, music genre data is used to perform clustering. The dataset contains various audio features, and the goal is to group the data into clusters using both KMeans and DBSCAN.

#### Dataset: Music Features Dataset

 #### Methods Used:

Random Forest to determine important features
![image](https://github.com/user-attachments/assets/f85db41c-5bd8-44d6-99b1-ae0e899db461)

KMeans Clustering with elbow method to find optimal k

DBSCAN for density-based clustering with optimal parameters found using Grid Search

#### Key Features: beats, chroma_stft, tempo, spectral_centroid, etc.

#### Results:

KMeans with 3 clusters and Voronoi diagram visualization.
<img width="569" alt="image" src="https://github.com/user-attachments/assets/72afaf91-9b89-4339-bd7c-294a0362e880" />
![image](https://github.com/user-attachments/assets/f5581e27-9e51-460a-bbe1-527e38a4c845)

DBSCAN with an optimal eps and min_samples values for good clustering performance.

Comparison of DBSCAN clusters to actual music genres and calculation of metrics such as Adjusted Rand Index and Homogeneity Score.
![image](https://github.com/user-attachments/assets/f17a2959-7d9a-46fd-80c7-c4af00b3d175)
![image](https://github.com/user-attachments/assets/90d728c7-668d-4bfb-b81d-f4dc19757f1e)
![image](https://github.com/user-attachments/assets/b0230087-615b-4271-b98d-ccbbae2a8e7d)


## Key Concepts
KMeans Clustering: A partitioning method where data points are assigned to a predefined number of clusters based on distance to the cluster centroids.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise): A clustering algorithm that groups together points that are close to each other, while marking as noise points that lie alone in low-density regions.

Silhouette Score: A metric used to evaluate the quality of clusters, indicating how similar an object is to its own cluster compared to other clusters.
