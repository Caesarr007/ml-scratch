# 实现一个循环神经网络 RNN

import torch
from torch import nn, einsum

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 当前时刻输入 + 之前隐藏状态
        combined = torch.cat((input, hidden), 1)
        # 更新隐藏状态
        hidden = self.i2h(combined)
        # 输出
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    

if __name__ == '__main__':
    # 实例化一个 RNN
    n_hidden = 128
    rnn = RNN(100, n_hidden, 10)

    # 输入
    input = torch.randn(1, 100)
    hidden = rnn.initHidden()

    # 前向传播
    output, next_hidden = rnn(input, hidden)
    print(output)