import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)

df = df.astype(int)

bandwidth = 100
meanshift = MeanShift(bandwidth=bandwidth)

meanshift.fit(df)

labels = meanshift.labels_
cluster_centers = meanshift.cluster_centers_

df['Cluster'] = labels

plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='red')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('Mean Shift Clustering')
plt.show()

silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
