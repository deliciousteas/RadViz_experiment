import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from yellowbrick.features import RadViz
from pandas.plotting import radviz

# 读取数据
csv_data_path = 'iris-normalization.csv'
data = pd.read_csv(csv_data_path)

dimensions = ['col1', 'col2', 'col3', 'col4']

r = 50  # 柱状图分段数量
x_bins = np.linspace(0, 1, r)

# 1. 计算概率分布
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 一次性创建四个dimension的图
axs = axs.flatten()  # 将XD array转为1D array

for idx, col in enumerate(dimensions):
    # 获取当前维度的数据
    dimension_data = data[col].values
    values_sum = np.sum(dimension_data)

    # 计算直方图
    counts, bins = np.histogram(dimension_data, bins=x_bins, density=True)  # 使用 r 个区间

    proportions = counts / values_sum #概率=区间分布数量/总数量

    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 计算每个区间的中心点


    # 绘制直方图
    axs[idx].bar(bin_centers, proportions, width=((bins[1] - bins[0]) * 0.5),
                 color="skyblue", edgecolor="black", linewidth=0.5)
    axs[idx].set_title(f"Probability Distribution of {col}")
    axs[idx].set_xlabel("Value")

    # 优化x坐标的显示
    axs[idx].set_xticks(np.arange(0, 1.1, 0.1))
    axs[idx].set_xticklabels([f"{x:.1f}" if int(x * 10) % 2 == 0 else "" for x in np.arange(0, 1.1, 0.1)])
    axs[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.show()

# 2. 均值漂移


bandwidth = 0.2
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
x_bins = np.linspace(0, 1, r)


dunn_values = []
accuracy_values = []

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


    # 计算Dunn值
    def dunn_index(data, labels):
        unique_labels = np.unique(labels)
        inter_cluster_distances = []
        intra_cluster_distances = []

        for i in unique_labels:
            cluster_i = data[labels == i]
            for j in unique_labels:
                if i != j:
                    cluster_j = data[labels == j]
                    inter_cluster_distances.append(np.min(pairwise_distances(cluster_i, cluster_j)))
            intra_cluster_distances.append(np.max(pairwise_distances(cluster_i, cluster_i)))

        return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


    dunn_value = dunn_index(probability_data, kmeans_labels)
    dunn_values.append(dunn_value)
    # 计算准确率
    accuracy = accuracy_score(temp_labels, kmeans_labels)
    accuracy_values.append(accuracy)
    # 输出Dunn值和准确率
    print(f"Dunn Index for {col}: {dunn_value}")
    print(f"Accuracy for {col}: {accuracy}")



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

    axs[idx].set_title(f"Clustered Histogram of {col}")
    axs[idx].set_xlabel("Value")
    axs[idx].set_ylabel("Probability Density")
    axs[idx].legend(loc='upper right')

    # 优化x坐标的显示
    axs[idx].set_xticks(np.arange(0, 1.1, 0.1))
    axs[idx].set_xticklabels([f"{x:.1f}" if int(x * 10) % 2 == 0 else "" for x in np.arange(0, 1.1, 0.1)])
    axs[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.show()

# 计算Dunn值和准确率的最大值、最小值和平均值
dunn_max = np.max(dunn_values)
dunn_min = np.min(dunn_values)
dunn_mean = np.mean(dunn_values)

accuracy_max = np.max(accuracy_values)
accuracy_min = np.min(accuracy_values)
accuracy_mean = np.mean(accuracy_values)

print(f"Dunn Index - Max: {dunn_max}, Min: {dunn_min}, Mean: {dunn_mean}")
print(f"Accuracy - Max: {accuracy_max}, Min: {accuracy_min}, Mean: {accuracy_mean}")




