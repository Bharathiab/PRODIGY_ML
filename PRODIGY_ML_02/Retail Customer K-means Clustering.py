import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('C:\\Users\\S.Bharathi\\Downloads\\Mall_Customers.csv')

# Drop non-numeric columns
data = data.drop(columns=['CustomerID'])  # Drop CustomerID as it's not useful for clustering

# Convert categorical column 'Gender' to numeric values using one-hot encoding
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)  # Drop first to avoid multicollinearity

# Check for and handle missing values if necessary
data = data.dropna()  # Simple approach to handle missing values

# Feature Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Optionally, determine the optimal k using Silhouette Score
best_k = 5  # Replace with the optimal k identified from the elbow method
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

silhouette_avg = silhouette_score(data_scaled, clusters)
print(f'Silhouette Score for k={best_k}: {silhouette_avg:.2f}')

# Add cluster labels to the original data
data_with_clusters = pd.read_csv('C:\\Users\\S.Bharathi\\Downloads\\Mall_Customers.csv')
data_with_clusters = data_with_clusters.drop(columns=['CustomerID'])
data_with_clusters = pd.get_dummies(data_with_clusters, columns=['Gender'], drop_first=True)
data_with_clusters['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_with_clusters['Annual Income (k$)'], data_with_clusters['Spending Score (1-100)'],
                     c=data_with_clusters['Cluster'], cmap='viridis', s=50)
plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(scatter, label='Cluster')
plt.show()
