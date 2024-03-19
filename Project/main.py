# importing the module

# read specific columns of csv file using Pandas
# print(df)
#
# noDevCommits = df['Number of developer commits'].tolist()
# print(noDevCommits)

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score, \
    mutual_info_score

# Read CSV file and select specific columns
df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits',
                                                               'NoPreviousModifications'])

# Fill or drop non-finite values
df.fillna(0, inplace=True)  # Fill NaN values with 0, you can choose a different value if needed

# Convert DataFrame values to integers
df = df.astype(int)

# Initialize KMeans with the desired number of clusters
kmeans = KMeans(n_clusters=2)

# Fit KMeans to the data
kmeans.fit(df)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Plot the clusters
plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('KMeans Clustering')
plt.show()

# Calculate clustering metrics
silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

# Print the metric scores
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
