import torch
from torch.distributions import multinomial
from d2l import torch as d2l


# 骰子概率
fair_probs = torch.ones([6]) / 6

counts = multinomial.Multinomial(10, fair_probs).sample((500,)) # 500组实验，每组10个样本
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()


# practice 练习

# # 1 进行m=500组实验，每组抽取n=10个样本。改变m和n，观察和分析实验结果。

import torch
from torch.distributions import multinomial
from d2l import torch as d2l

probs = torch.ones(size=(6,)) / 6

counts = multinomial.Multinomial(10, fair_probs).sample((500,)) # 500组实验，每组10个样本
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()

# # 2 

