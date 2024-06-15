# 实现一个 ResNet 的残差连接

import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 如果前后通道数不一致时，使用 1x1 卷积调整通道数
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # ReLU应用在残差连接之后
        return torch.relu(Y + X)
    

if __name__ == "__main__":
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    print(blk(X).shape)  # torch.Size([4, 3, 6, 6])

    blk = Residual(3, 6, use_1x1conv=True, stride=2)
    print(blk(X).shape)  # torch.Size([4, 6, 3, 3])