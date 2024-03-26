import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import OPTICS

df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

df.fillna(0, inplace=True)

df = df.astype(int)

optics_clusterer = OPTICS()

optics_clusterer.fit(df)

labels = optics_clusterer.labels_
core_distances = optics_clusterer.core_distances_

reachability_plot = optics_clusterer.reachability_[optics_clusterer.ordering_]

epsilon = 1e-10
reachability_plot[reachability_plot == 0] = epsilon

ratio = reachability_plot[:-1] / reachability_plot[1:]


plt.scatter(df['McCC'], df['LOC'], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('OPTICS Clustering')
plt.show()

# optics has 0 values and could not be ploted