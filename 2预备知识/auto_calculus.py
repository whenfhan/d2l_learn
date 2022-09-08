import torch


x = torch.arange(4.0)
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x, x) # y = 2 * x.T * x
y.backward()
x.grad
x.grad == 4 * x

x.grad.zero_() # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
y = x.sum()
y.backward()
x.grad

# 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad, u

# python控制流的梯度计算
def f(a):
    """f(a) 是分段线性函数"""
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(2,1), requires_grad=True)
d = f(a)
d.backward()
a.grad, d / a


# practice 练习

# 1 为什么计算二阶导数比一阶导数的开销要更大？

# 计算二阶导数是在一阶导数的基础上进行的，自然开销要大。

# 2 在运行反向传播函数之后，立即再次运行它，看看会发生什么。

import torch

x = torch.arange(40., requires_grad=True)
y = 2 * torch.dot(x**2, torch.ones_like(x))
y.backward()
x.grad
y.backward() # 会报错，要在第一次反向遍历图的时候添加 retain_graph=True， 才能再次执行

# 3 在控制流的例子中，我们计算d关于a的导数，如果我们将变量a更改为随机向量或矩阵，会发生什么？

import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
# d.backward() # 会报错: grad can be implicitly created only for scalar outputs
d.sum().backward() # 成功运行
print(a.grad)

# 4 重新设计一个求控制流梯度的例子，运行并分析结果。

import torch

def f(a):
    while a.norm() < 1000:
        if a % 2 == 0:
            a *= 2
        else:
            a *= 3
    return a

a = torch.randn(size=(), requires_grad=True)
d = f(a)
# d.backward() # 会报错: grad can be implicitly created only for scalar outputs
d.sum().backward() # 成功运行
print(a.grad)

# 5 使f(x) = sin(x)，绘制f(x)和df(x)/dx的图像，其中后者不使用f'(x)=cos(x)。

import torch
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5.,5.,1000)
x1 = torch.tensor(x, requires_grad=True)
y1 = torch.sin(x1)
y1.sum().backward()
plt.plot(x, np.sin(x))
plt.plot(x, x1.grad)