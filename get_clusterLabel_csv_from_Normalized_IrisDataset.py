import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

# 读取数据
"""
1，读取归一化的鸢尾花数据集
2.获取概率分布直方图，将概率-数值关系作为均值漂移算法的输入。
3. 将结构保存为当前路径的csv文件。

"""
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

    cluster_map = {}
    for label in np.unique(labels):
        cluster_map[label] = (bin_centers[labels == label].min(), bin_centers[labels == label].max())# 为临时变量添加cluster标签列

    cluster_labels = np.zeros(len(temp_var), dtype=int)
    for j, value in enumerate(temp_var[:, i + 1]):
        if not np.isnan(value):
            for label, (min_val, max_val) in cluster_map.items():
                if min_val <= value <= max_val:
                    cluster_labels[j] = label
                    break
    temp_var = np.column_stack((temp_var, cluster_labels))
    # 保存为本地CSV文件
    result_df = pd.DataFrame(temp_var, columns=list(data.columns) + ['cluster_label'])
    result_df.to_csv(f'{col}_result.csv', index=False)

