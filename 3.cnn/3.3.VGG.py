# 实现一个 VGG

import torch
from torch import nn, einsum

class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()
        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (out_channels, num_convs) in conv_arch:
            conv_blks.append(self.vgg_block(in_channels, out_channels, num_convs))
            in_channels = out_channels
        self.net = nn.Sequential(
            *conv_blks,
            # 全连接层部分
            nn.Flatten(),
            nn.Linear(out_channels*8*8, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    
    def vgg_block(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            print(layer.__class__.__name__,'output shape: \t',x.shape)
        return x
    

if __name__ == '__main__':
    # 第一个参数：输出通道数；第二个参数：卷积层个数
    conv_arch = ((1, 1), (2, 1), (4, 2), (8, 2), (16, 2))
    model = VGG(conv_arch)
    x = torch.rand(size=(1, 1, 256, 256), dtype=torch.float32)
    print(model(x).shape)