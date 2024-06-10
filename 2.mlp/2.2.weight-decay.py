# 权重衰减的正则化技术限制模型权重值大小
import torch


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)