import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)

df = df.astype(int)

n_clusters = 3
n_neighbors = 10
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=42)

spectral_clustering.fit(df)

labels = spectral_clustering.labels_

df['Cluster'] = labels

plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('Spectral Clustering')
plt.show()

# spectral_clustering is not suitable for this dataset
