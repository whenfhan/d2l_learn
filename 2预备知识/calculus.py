import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


# 小结：
# # 微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。
# # 导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。
# # 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。
# # 链式法则使我们能够微分复合函数。

def f(x):
    """f(x) = 3x^2 - 4x"""
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x+h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], xlabel='x', ylabel='f(x)', legend=['f(x)', 'Tangent Line (x=1)'])
d2l.plt.show()