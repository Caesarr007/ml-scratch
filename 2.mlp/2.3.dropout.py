# 随机丢弃一些神经元（失活）

# 实现一个单隐藏层的多层感知机
import torch
from torch import nn, einsum

class mlp(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(mlp, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.zeros(size=(in_dim, hid_dim)))
        self.b1 = nn.Parameter(torch.zeros(size=(hid_dim,)))
        self.w2 = nn.Parameter(torch.zeros(size=(hid_dim, out_dim)))
        self.b2 = nn.Parameter(torch.zeros(size=(out_dim,)))
        nn.init.xavier_uniform_(self.w1.data, gain=1.414)
        nn.init.xavier_uniform_(self.w2.data, gain=1.414)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x):
        out = einsum('b i, i j-> b j', x, self.w1) + self.b1
        print("out before relu", out)
        out = torch.relu(out)
        print("out after relu", out)
        out = self.dropout(out)
        print("out after dropout", out)
        out = einsum('b i, i j-> b j', out, self.w2) + self.b2
        return torch.softmax(out, dim=-1)


if __name__ == '__main__':
    model = mlp(3, 4, 3)
    x = torch.rand(3, 3)
    print("x", x)
    y = model(x)
    print("y", y)
    print("result", y.argmax(dim=-1))
