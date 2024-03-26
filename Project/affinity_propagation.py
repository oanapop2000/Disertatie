import pandas as pd
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

# Read CSV file and select specific columns
df = pd.read_csv("Elasticsearch-0.90.11-Unified.csv", usecols=['McCC', 'LOC', 'NoDevCommits', 'NoPreviousModifications'])

# Fill or drop non-finite values
df.fillna(0, inplace=True)  # Fill NaN values with 0, you can choose a different value if needed

# Convert DataFrame values to integers
df = df.astype(int)

# Initialize Affinity Propagation with the desired parameters
damping = 0.7  # Adjust damping parameter
affinity_propagation = AffinityPropagation(damping=damping)

# Fit Affinity Propagation to the data
affinity_propagation.fit(df)

# Get cluster labels
labels = affinity_propagation.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Plot the clusters
plt.scatter(df['McCC'], df['LOC'], c=df['Cluster'], cmap='viridis')
plt.xlabel('McCabe Cyclomatic Complexity')
plt.ylabel('Lines of Code')
plt.title('Affinity Propagation Clustering')
plt.show()

# affinity_propagation is not suitable for this dataset