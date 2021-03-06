import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def forward(self, query, key, value, mask):
        model_dim, model_len = query.size(-1), query.size(-2)

        query *= float(model_dim) ** -0.5

        # Calculating Attention Score
        scores = torch.matmul(query, key.transpose(-1, -2))

        # Reference: https://discuss.pytorch.org/t/runtimeerror-value-cannot-be-converted-to-type-at-half-without-overflow-1e-30/109768
        _MASKING_VALUE = -1e+30 if scores.dtype == torch.float32 else -5e+4

        # Mask out attention to self
        i = torch.arange(model_len)
        scores[:, :, i, i] = _MASKING_VALUE

        # Fill _MASKING_VALUE when mask == 1
        if mask is not None:
            scores = scores.masked_fill(mask, _MASKING_VALUE)

        # Calculating Attention with softmax
        attention = F.softmax(scores, dim=-1)

        # Calculating Attention Multiplied actual value
        sum_value = torch.matmul(attention, value)

        return attention, sum_value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.attention = Attention()
        self.d_model = d_model
        self.d_model_h = d_model // h

        # todo: linear should be defined with each H (e.g W_i of H)
        self.linears = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(3)])
        self.h = h

        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Making Distributed Tensor by H
        _query, _key, _value = [linear(x).view(batch_size, -1, self.h, self.d_model_h).transpose(1, 2)
                                for linear, x in zip(self.linears, [query, key, value])]

        # Applying Dot-Product Attention
        _, sum_value = self.attention(_query, _key, _value, mask)

        # Concat H distributed Attention
        # attention = attention.view(batch_size, query.size(1), _key.size(1))
        sum_value = sum_value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Return attention and value
        return sum_value

if __name__ == '__main__':
    test = MultiHeadAttention(10, 2)
    x = torch.ones([1, 10, 10], dtype=torch.float)
    y = test(x, x, x)
    print(x, y)