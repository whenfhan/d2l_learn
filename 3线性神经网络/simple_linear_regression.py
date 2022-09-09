import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 构造数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 小批量随机读取数据
batch_size = 10
data_iter = d2l.load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 正态分布均值为0、标准差为0.01中随机采样
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 清楚上一次的grad，避免累加
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch:{epoch+1}, loss:{l:f}')