# 实现一个 LeNet

import torch
from torch import nn, einsum

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), 
            nn.Sigmoid(),
            nn.Linear(120, 84), 
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            print(layer.__class__.__name__,'output shape: \t',x.shape)
        return x


if __name__ == '__main__':
    model = LeNet()
    x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    print(model(x).shape)