
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
n_customers = 200

data = pd.DataFrame({
    'CustomerID': range(1, n_customers+1),
    'Gender': np.random.choice(['Male', 'Female'], size=n_customers),
    'Age': np.random.randint(18, 70, size=n_customers),
    'Annual Income (k$)': np.random.randint(15, 140, size=n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, size=n_customers)
})

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans

cluster_profile = data.groupby('Cluster').mean()
print("Cluster Profiles:\n", cluster_profile)

plt.figure(figsize=(10,6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    cluster = data[data['Cluster'] == i]
    plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'],
                s=50, c=colors[i], label=f'Cluster {i}')

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:,2], centroids[:,3], s=200, c='yellow', marker='X', label='Centroids')

plt.title('Customer Segments (Income vs Spending Score)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
