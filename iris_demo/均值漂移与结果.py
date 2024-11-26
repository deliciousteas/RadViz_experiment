
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans



csv_data_path = 'iris-normalization.csv'
data = pd.read_csv(csv_data_path)

dimensions = ['col1', 'col2', 'col3', 'col4']

r=50
bandwidth = 0.2
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
x_bins = np.linspace(0, 1, r)




for idx, col in enumerate(dimensions):

    dimension_data = data[col].values

    # 计算概率密度的 x 和 y 数据
    counts, bins = np.histogram(dimension_data, bins=x_bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 区间中心点
    percentage = counts / np.sum(counts)  # 概率计算
    probability_data = np.column_stack(( bin_centers,percentage))


    # 均值漂移聚类
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(probability_data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    temp_labels = labels
    temp_cluster_centers = cluster_centers
    num_clusters = len(np.unique(labels))

    # K-means聚类
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(probability_data)
    kmeans_labels = kmeans.labels_

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_mask = labels == label
        axs[idx].bar(
            bin_centers[cluster_mask],
            counts[cluster_mask],
            width=((bins[1] - bins[0]) * 0.5),
            label=f"Cluster {label}",
            alpha=0.6
        )
    total_sum = np.sum(counts)
    axs[idx].set_title(f"Clustered Histogram of {col}")
    axs[idx].set_xlabel("Value")
    axs[idx].set_ylabel("Probability Density")
    axs[idx].legend(loc='upper right')

    # 优化x坐标的显示
    axs[idx].set_xticks(np.arange(0, 1.1, 0.1))
    axs[idx].set_xticklabels([f"{x:.1f}" if int(x * 10) % 2 == 0 else "" for x in np.arange(0, 1.1, 0.1)])
    axs[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y / total_sum)))

plt.tight_layout()
plt.show()
