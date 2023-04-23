# @Time    : 2023/4/6 22:01
# @Author  : ygd
# @FileName: test.py
# @Software: PyCharm

import torch

x = torch.Tensor([2, 7, 3]) #20次，70次，30次
m = torch.distributions.Categorical(x)
re = [0, 0, 0] #三个数抽到的个数
for i in range(100):
    re[m.sample()] += 1 #sample就是抽一次

print(re)
