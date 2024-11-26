from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()

# Extract the four dimensions
data = iris.data

# Normalize the data to [0, 1]
normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Number of bins


# Plot the percentage distribution for each dimension
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    # Calculate the histogram
    counts, bin_edges = np.histogram(normalized_data[:, i], bins=20, range=(0, 1))
    print(counts)
    print(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Convert counts to percentage
    percentages = (counts / counts.sum()) * 100
    print(counts.sum())

    # Plot the histogram
    ax.bar(bin_centers, percentages, width=1 / 20, alpha=0.6, color='g')
    ax.set_title(iris.feature_names[i])
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Percentage (%)')

plt.tight_layout()
plt.show()