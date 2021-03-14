import torch.nn as nn
import torch.nn.functional as fnn
import torch
import math


class Attention(nn.Module):
    def forward(self, query, key, value, mask):
        model_dim = query.size(-1)

        # Calculating Attention Score
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(model_dim)

        scores = scores.masked_fill(mask == 0, -1e9)

        # Calculating Attention with softmax
        attention = fnn.softmax(scores, dim=-1)

        # Calculating Attention Multiplied actual value
        sum_value = torch.matmul(attention, value)

        return attention, sum_value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.attention = Attention()
        self.d_model = d_model
        self.d_model_h = d_model // h

        # todo: linear should be defined with each H (e.g W_i of H)
        self.linears = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(3)])
        self.h = h

    def forward(self, query, key, value, mask=None):
        batch_size, d_model = query.size(0), query.size(-1)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Making Distributed Tensor by H
        _query, _key, _value = [linear(x).view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2)
                                for linear, x in zip(self.linears, [query, key, value])]

        # Applying Dot-Product Attention
        attention, sum_value = self.attention(_query, _key, _value, mask)

        # Concat H distributed Attention
        # attention = attention.view(batch_size, query.size(1), _key.size(1))
        sum_value = sum_value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Return attention and value
        return sum_value
