import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
# 读取数据
csv_data_path = 'iris-normalization.csv'
data = pd.read_csv(csv_data_path)
data_array = data.values

# 定义维度
dimensions = ['col1', 'col2', 'col3', 'col4']

# 创建四个临时变量，分别包含每个维度的数据，其他数值保持不变
temp_vars = [data_array.copy() for _ in dimensions]

# 只保留对应维度的数据，其他维度的数据置为NaN
for i, col in enumerate(dimensions):
    temp_vars[i][:, [j for j in range(1, len(dimensions) + 1) if j != i + 1]] = np.nan

# 均值漂移参数
bandwidth = 0.2
x_bins = np.linspace(0, 1, 50)

dunn_values = []


# 对每个临时数组进行均值漂移并保存结果
for i, col in enumerate(dimensions):
    temp_var = temp_vars[i]
    dimension_data = temp_var[:, i + 1]  # 跳过第一列id

    # 计算概率密度的 x 和 y 数据
    counts, bins = np.histogram(dimension_data, bins=x_bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # 区间中心点
    proportions = counts / np.sum(counts)  # 概率计算
    probability_data = np.column_stack((bin_centers, proportions))

    # 均值漂移聚类
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(probability_data)
    labels = ms.labels_
    cluster_centers=ms.cluster_centers_

    temp_labels=labels
    temp_cluster_centers=cluster_centers
    num_clusters=len(np.unique(labels))
    kmeans=KMeans(n_clusters=num_clusters)
    kmeans.fit(probability_data)
    kmeans_labels=kmeans.labels_

    cluster_map = {}
    for label in np.unique(labels):
        cluster_map[label] = (
        bin_centers[labels == label].min(), bin_centers[labels == label].max())  # 为临时变量添加cluster标签列

    cluster_labels = np.zeros(len(temp_var), dtype=int)
    for j, value in enumerate(temp_var[:, i + 1]):
        if not np.isnan(value):
            for label, (min_val, max_val) in cluster_map.items():
                if min_val <= value <= max_val:
                    cluster_labels[j] = label
                    break
    temp_var = np.column_stack((temp_var, cluster_labels))
    #保存为本地CSV文件
    result_df = pd.DataFrame(temp_var, columns=list(data.columns) + ['cluster_label'])
    result_df.to_csv(f'{col}_result.csv', index=False)
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
    print(f"Dunn Index for {col}: {dunn_value}")

#1。 计算K-means的Duun值




# 2. 计算准确率

#读取csv文件，读取classID和cluster_label两列，计算准确率。利用匈牙利算法来可视化。
csv_path=["col1_result.csv",
          "col2_result.csv",
          "col3_result.csv",
          "col4_result.csv"]
accuracy_values=[]

for i in range(len(csv_path)):
    df=pd.read_csv(csv_path[i])
    contingency_matrix=pd.crosstab(df['classId'],df['cluster_label'])


    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    mapping={row:col+1 for row,col in zip(row_ind,col_ind)}

    df['aligned_cluster_label']=df['cluster_label'].map(mapping)

    accuracy=accuracy_score(df['classId'],df['aligned_cluster_label'])
    accuracy_values.append(accuracy)
    print(f"Accuracy for {dimensions[i]}: {accuracy}")



#用的混淆局郑

