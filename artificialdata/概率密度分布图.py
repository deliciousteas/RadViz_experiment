import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
csv_data_path = 'D:\\dzg\\040职业发展\\041phd\\042博士测试\\实验数据与文档\\PaperData-软件学报-4DIn12D-350point-withClass3-withXYpointByMDS.csv'
data = pd.read_csv(csv_data_path)

dimensions = ['dim2', 'dim6', 'dim7', 'dim8']

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