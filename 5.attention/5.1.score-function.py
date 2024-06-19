# 实现 加性注意力 和 缩放点积注意力

import torch
from torch import nn, einsum

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(score, mask):
        score = score.masked_fill(mask, float('-inf'))
        return nn.functional.softmax(score, dim=-1)

    def forward(self, key, query, value, mask):
        query = self.W_q(query)
        key = self.W_k(key)
        # 维度扩展
        # query: [batch_size, query_len, 1, hidden_size]
        # key: [batch_size, 1, key_len, hidden_size]
        features = query.unsqueeze(2) + key.unsqueeze(1)
        features = torch.tanh(features)

        score = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(score, mask)

        return torch.bmm(self.dropout(self.attention_weights), value)
    

if __name__ == '__main__':
    # 测试加性注意力
    batch_size = 2
    key_size = 3
    query_size = 3
    hidden_size = 4
    dropout = 0.1
    key = torch.randn(batch_size, 5, key_size)
    query = torch.randn(batch_size, 5, query_size)
    value = torch.randn(batch_size, 5, hidden_size)
    mask = torch.zeros(batch_size, 5, 5).bool()
    attention = AdditiveAttention(key_size, query_size, hidden_size, dropout)
    output = attention(key, query, value, mask)
    print(output)