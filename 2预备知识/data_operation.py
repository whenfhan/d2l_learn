import torch

# 矩阵元素
x = torch.arange(12)
x.shape     # 张量x的形状
x.numel()   # 张量x中元素的总数
x = x.reshape(3,4)

# 创建矩阵
torch.zeros((2,3,4)) # 全零矩阵
torch.randn(3,4) # 随机矩阵
torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]) # 自定义矩阵

# 矩阵运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y 

# 矩阵拼接
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0)
torch.cat((X, Y), dim=1)

# 广播机制
# 需要满足以下条件才能触发广播机制：
# 从尾部的维度开始，维度尺寸
# 1、或者相等，
# 2、或者其中一个张量的维度尺寸为 1 ，
# 3、或者其中一个张量不存在这个维度。
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
a + b 

# 索引和切片
X[-1], X[1:3]
X[1, 2] = 9
X[0:2, :] = 12

# 节省内存
before = id(Y)
Y = Y + X
id(Y) == before # Y指向新开辟的内存

Z = torch.zeros_like(Y)
before = id(Z)
Z[:] = Y + X    # 使用 [:]= 或者 += 来代替
id(Z) == before # Z指向的内存没变化

