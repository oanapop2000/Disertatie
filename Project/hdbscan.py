import pandas as pd
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)  # Fill NaN values with 0, you can choose a different value if needed

df = df.astype(int)

hdbscan_clusterer = HDBSCAN()

hdbscan_clusterer.fit(df)

plt.scatter(df['McCC'], df['LOC'], c=hdbscan_clusterer.labels_, cmap='viridis', marker='o', alpha=0.6)
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('HDBSCAN Clustering')
plt.show()

# could not be installed
