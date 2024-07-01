# 实现多头缩放点积注意力

import torch
from torch import nn, einsum

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(self, X, mask):
        if mask is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if mask.dim() == 1:
                mask = torch.repeat_interleave(mask, repeats=shape[1])
            else:
                mask = mask.reshape(-1)
        print("mask", mask)
        X = X.reshape(-1, shape[-1])
        maxlen = X.shape[1]
        value = -1e6
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = value
        X = X.reshape(shape)
        print("X", X)

        return nn.functional.softmax(X, dim=-1)

    def forward(self, query, key, value, mask):
        # query: [batch_size, query_len, hidden_size]
        # key: [batch_size, kv_len, hidden_size]
        # value: [batch_size, kv_len, hidden_size]
        d = query.shape[-1]
        score = einsum('bqd,bkd->bqk', query, key) / d ** 0.5
        self.attention_weights = self.masked_softmax(score, mask)
        return einsum('bqk,bkd->bqd', self.dropout(self.attention_weights), value)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, mask):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=self.num_heads, dim=0)
        print("mask shape", mask.shape)
        print(mask)

        output = self.attention(queries, keys, values, mask)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



    def transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
        # num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
    

    def transpose_output(self, X, num_heads):
        """逆转多头注意力的变换"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    

if __name__ == '__main__':
    batch_size = 2
    num_queries = 4
    num_kv = 6
    num_hiddens = 64
    num_heads = 4
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kv, num_hiddens))
    mask = torch.tensor([3, 2])
    print(mask)

    # attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    # output = attention(X, Y, Y, mask)

    attention = DotProductAttention(0.5)
    score = torch.randn(2, 2, 4)
    valid_lens = torch.tensor([2, 3])
    print("score", score)
    print(attention.masked_softmax(score, valid_lens))
    
