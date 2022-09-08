import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x * y, x / y, x**y

# 向量
torch.tensor([0, 1, 2, 3])

# 矩阵
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)

# 求和降维
A.sum() 
A_sum_axis0 = A.sum(axis=0) # 指定axis轴，将axis轴降维(消失)
A_sum_axis0, A_sum_axis0.shape
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
sum_A
A / sum_A

# 累加统计
A.cumsum(axis=0) # 沿axis轴累加

# 点积
# 两向量点积： 按元素乘积的和
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

# 矩阵-向量积
A.shape, x.shape, torch.mv(A, x)

# 矩阵-矩阵乘法
# [m,n] * [n,k] = [m,k]
B = torch.ones(4, 3)
torch.mm(A, B)

# 范数
# 一般范数指的是L2范数，向量元素平方和的平方根
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# Frobenius范数满足向量范数的所有性质
torch.norm(torch.ones((4,9)))