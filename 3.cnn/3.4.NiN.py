# 实现一个 NiN

import torch
from torch import nn, einsum

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            self.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            nn.Flatten()
        )

    
    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
    

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            print(layer.__class__.__name__,'output shape: \t',x.shape)
        return x
    

if __name__ == '__main__':
    model = NiN()
    x = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
    print(model(x).shape)