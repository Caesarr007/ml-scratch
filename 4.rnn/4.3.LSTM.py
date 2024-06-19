# 实现一个长短期记忆网络 LSTM

import torch
from torch import nn, einsum

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # f: 遗忘门
        self.i2f = nn.Linear(input_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size)
        # i: 输入门
        self.i2i = nn.Linear(input_size, hidden_size)
        self.h2i = nn.Linear(hidden_size, hidden_size)
        # o: 输出门
        self.i2o = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, hidden_size)
        # g: 更新门/候选记忆
        self.i2g = nn.Linear(input_size, hidden_size)
        self.h2g = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        # 遗忘门
        f = torch.sigmoid(self.i2f(input) + self.h2f(hidden))
        # 输入门
        i = torch.sigmoid(self.i2i(input) + self.h2i(hidden))
        # 输出门
        o = torch.sigmoid(self.i2o(input) + self.h2o(hidden))
        # 更新门
        g = torch.tanh(self.i2g(input) + self.h2g(hidden))
        # 更新细胞状态
        cell = f * cell + i * g
        # 更新隐藏状态
        hidden = o * torch.tanh(cell)
        # 输出
        output = self.softmax(hidden)
        return output, hidden, cell
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def initCell(self):
        return torch.zeros(1, self.hidden_size)
    

if __name__ == '__main__':
    lstm = LSTM(10, 20, 10)
    input = torch.randn(1, 10)
    hidden = lstm.initHidden()
    cell = lstm.initCell()
    output, hidden, cell = lstm(input, hidden, cell)
    print(output, hidden, cell)