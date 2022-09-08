import os
import pandas as pd
import torch


# 小结：
# # pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。
# # 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny_csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) # 用均值填充NA
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)