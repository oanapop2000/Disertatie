import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)

df = df.astype(int)

n_clusters = 3  # Number of clusters
agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters)

agglomerative_clustering.fit(df)

labels = agglomerative_clustering.labels_

df['Cluster'] = labels

plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('Agglomerative Clustering')
plt.show()

silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
