# 检测batchnorm对数据的变化，作用于每个channels

import torch
from torch import nn


if __name__ == '__main__':
    # batchsize，channels，height，width
    x = torch.rand(size=(1, 2, 3, 3), dtype=torch.float32)
    print(x)
    x = nn.BatchNorm2d(2)(x)
    print(x)
    
    # batchsize，channels，length
    y = torch.rand(size=(2, 3, 3), dtype=torch.float32)
    print(y)
    y = nn.BatchNorm1d(3)(y)
    print(y)

