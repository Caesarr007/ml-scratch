# 实现一个线性回归模型
import torch
from torch import nn, einsum


class linear_regression(nn.Module):
    def __init__(self, xs):
        super(linear_regression, self).__init__()
        self.xs = xs
        self.w = nn.Parameter(torch.zeros(size=(xs, 1)))
        self.w.data.fill_(1)
        self.b = nn.Parameter(torch.zeros(size=(1,)))
        self.b.data.fill_(10)

    def forward(self, x):
        print("w", self.w)
        print("b", self.b)
        return einsum('b i, i j-> b j', x, self.w) + self.b
    

if __name__ == '__main__':
    model = linear_regression(2)
    x = torch.rand(5, 2)
    print("x", x)
    y = model(x)
    print("y", y)