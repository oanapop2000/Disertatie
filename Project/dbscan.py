import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Read CSV file and select specific columns
df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

# Fill or drop non-finite values
df.fillna(0, inplace=True)  # Fill NaN values with 0, you can choose a different value if needed

# Convert DataFrame values to integers
df = df.astype(int)

# Initialize DBSCAN with the desired parameters
eps = 100  # Epsilon parameter, determines the maximum distance between two samples for one to be considered as in the neighborhood of the other
min_samples = 5  # Minimum number of samples in a neighborhood for a point to be considered as a core point
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit DBSCAN to the data and obtain cluster labels
labels = dbscan.fit_predict(df)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Plot the clusters
plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('DBSCAN Clustering')
plt.show()

# Calculate clustering metrics
silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

# Print the metric scores
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
