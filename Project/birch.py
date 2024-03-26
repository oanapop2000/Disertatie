
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)  # Fill NaN values with 0, you can choose a different value if needed

df = df.astype(int)

birch_clusterer = Birch(n_clusters=3)

birch_clusterer.fit(df)

labels = birch_clusterer.predict(df)

plt.scatter(df['McCC'], df['LOC'], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('BIRCH Clustering')
plt.show()

silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
