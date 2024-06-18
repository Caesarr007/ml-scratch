# 实现一个门控循环单元 GRU

import torch
from torch import nn, einsum

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.i2r = nn.Linear(input_size, hidden_size)
        self.h2r = nn.Linear(hidden_size, hidden_size)
        self.i2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 重置门
        r = torch.sigmoid(self.i2r(input) + self.h2r(hidden))
        # 更新门
        z = torch.sigmoid(self.i2z(input) + self.h2z(hidden))
        # 候选隐藏状态
        h_tilde = torch.tanh(self.i2h(input) + self.h2h(r * hidden))
        # 更新隐藏状态
        hidden = z * hidden + (1 - z) * h_tilde
        # 输出
        output = self.softmax(hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    

if __name__ == '__main__':
    # 实例化一个 GRU
    n_hidden = 128
    gru = GRU(100, n_hidden, 10)

    # 输入
    input = torch.randn(1, 100)
    hidden = gru.initHidden()

    # 前向传播
    output, next_hidden = gru(input, hidden)
    print(output)