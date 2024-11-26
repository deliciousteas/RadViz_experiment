# RadViz实验
Radviz是一种多维数据可视化技术，它通过将数据的多维特征映射到二维平面上，辅助人们理解数据的维度特征。  
然而，Radviz的可视化效果受限于**维度的数量和特征的排序顺序**，这将导致**数据认知信息受损**。该项目尝试对论文[1]技术路线复现。  
首先，利用概率统计的方法捕获鸢尾花数据集不同特征维度的频率-数值关系，然后作为均值漂移算法的输入以得到升维处理；使用Dunn值何准确率来作为升维结果评价指标。
最后，利用皮尔逊系数实获得多个维度的排序序列，基于D3.JS实现可视化结果。
## 技术路线图
<center><img width="610" alt="{2D4D0A00-92A0-4B4B-A328-D4FBB2CEB740}" src="https://github.com/user-attachments/assets/ee0d94d9-7a24-4748-a8da-b3863938cd6c"></center>

图1：技术路线图  

## 技术路径  
- 编程语言与编辑器：  
Pycharm+python 
Vscode+js 
- 环境依赖：  
  Python : Pandas、Numpy、Sklearn、Scipy  
  JS : D3.JS(d3.csv、d3.arc、d3.arc、d3.forceSimulation)

# 效果预览图
<img width="328" alt="{4816AF8D-5A1E-4B17-A9F5-C97D6334CB1B}" src="https://github.com/user-attachments/assets/f5b12f28-f9a5-410d-b06f-85bc136b4420">  

图2：鸢尾花数据原始维度可视化  

---

<img width="332" alt="{0497DC2C-ADFD-4333-B8ED-313A5AD12178}" src="https://github.com/user-attachments/assets/d21ff6d0-8038-4153-a4a5-47b34adeaea1">  

图3：鸢尾花数据升维可视化  


# 参考
[1]周芳芳,李俊材,黄伟,等.基于维度扩展的Radviz可视化聚类分析方法[J].软件学报,2016,27(05):1127-1139.
