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

    def masked_softmax(self, score, mask):
        score = score.masked_fill(mask, float('-inf'))
        return nn.functional.softmax(score, dim=-1)

    def forward(self, key, query, value, mask):
        query = self.W_q(query)
        key = self.W_k(key)
        # 维度扩展
        # query: [batch_size, query_len, 1, hidden_size]
        # key: [batch_size, 1, key_len, hidden_size]
        features = query.unsqueeze(2) + key.unsqueeze(1)
        print("features shape", features.shape)
        features = torch.tanh(features)

        score = self.W_v(features).squeeze(-1)
        print("score shape", score.shape)
        self.attention_weights = self.masked_softmax(score, mask)

        return torch.bmm(self.dropout(self.attention_weights), value)
    

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(self, score, mask):
        score = score.masked_fill(mask, float('-inf'))
        return nn.functional.softmax(score, dim=-1)

    def forward(self, query, key, value, mask):
        # query: [batch_size, query_len, hidden_size]
        # key: [batch_size, kv_len, hidden_size]
        # value: [batch_size, kv_len, hidden_size]
        d = query.shape[-1]
        score = einsum('bqd,bkd->bqk', query, key) / d ** 0.5
        print("score shape", score.shape)
        self.attention_weights = self.masked_softmax(score, mask)
        print("attention_weights", self.attention_weights)
        return einsum('bqk,bkd->bqd', self.dropout(self.attention_weights), value)
    

if __name__ == '__main__':
    # 测试加性注意力
    # batch_size = 2
    # key_size = 3
    # query_size = 3
    # hidden_size = 3
    # dropout = 0.1
    # key = torch.randn(batch_size, 4, key_size)
    # query = torch.randn(batch_size, 5, query_size)
    # value = torch.randn(batch_size, 4, hidden_size)
    # # mask 的形状 [batch_size, query_len, key_len]
    # mask = torch.zeros(batch_size, 5, 4).bool()
    # attention = AdditiveAttention(key_size, query_size, hidden_size, dropout)
    # output = attention(key, query, value, mask)
    # print(output)

    # 测试缩放点积注意力
    queries, keys = torch.normal(0, 1, (2, 1, 2)), torch.ones((2, 10, 2))
    print("queries shape", queries.shape)
    print("keys shape", keys.shape)
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    print("values shape", values.shape)
    mask = torch.zeros(2, 1, 10).bool()
    attention = DotProductAttention(dropout=0.5)
    attention.eval() 
    result = attention(queries, keys, values, mask)
    print("result shape", result.shape)
    print(result)
