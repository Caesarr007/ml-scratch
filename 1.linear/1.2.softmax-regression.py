# 实现一个softmax回归模型
import torch
from torch import nn, einsum


class softmax_regression(nn.Module):
    def __init__(self, xs, classes):
        super(softmax_regression, self).__init__()
        self.xs = xs
        self.w = nn.Parameter(torch.zeros(size=(xs, classes)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(classes,)))

    def forward(self, x):
        print("w", self.w)
        print("b", self.b)
        out = einsum('b i, i j-> b j', x, self.w) + self.b
        print("out", out)
        return torch.softmax(out, dim=-1)
    

if __name__ == '__main__':
    model = softmax_regression(3, 3)
    x = torch.rand(3, 3)
    print("x", x)
    y = model(x)
    print("y", y)
    print("res", y.argmax(dim=-1))

    y_true = torch.tensor([0, 1, 2])
    loss = nn.CrossEntropyLoss()
    print("loss", loss(y, y_true))