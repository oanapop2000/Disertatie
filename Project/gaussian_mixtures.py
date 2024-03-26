import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)

df = df.astype(int)

n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)

gmm.fit(df)

labels = gmm.predict(df)

plt.scatter(df['McCC'], df['LOC'], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('Gaussian Mixture Model Clustering')
plt.show()

silhouette = silhouette_score(df, labels)
db_index = davies_bouldin_score(df, labels)
ch_index = calinski_harabasz_score(df, labels)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
