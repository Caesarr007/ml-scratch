# 实现transformer

import torch
from torch import nn, einsum

# 前馈神经网络
class PositionWiseFFN(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )
        
    def forward(self, x):
        return self.net(x)
    
# 残差连接和层规范化
class AddNorm(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


if __name__ == '__main__':
    # ffn test
    # ffn = PositionWiseFFN(4, 4, 8)
    # x = torch.ones((2, 3, 4))
    # print(ffn(x).shape)  # torch.Size([2, 3, 8])

    # addnorm test
    addnorm = AddNorm(4, 0.5)
    x = torch.ones((2, 3, 4))
    y = torch.ones((2, 3, 4))
    print(addnorm(x, y).shape)  # torch.Size([2, 3, 4])