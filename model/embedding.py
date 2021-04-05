import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, seq_len, min_timescale:float=1.0, max_timescale:float=1.0e4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        # Reference: https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L408
        num_timescales = d_model // 2
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = min_timescale * torch.exp(torch.arange(num_timescales) * -(math.log(max_timescale / min_timescale) / max(float(num_timescales) - 1, 1)))
        div_term = div_term.unsqueeze(0)

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("div_term", div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, step, type):
        if type == 'position':
            x = x + self.pe
            pe = torch.zeros(self.seq_len, self.d_model).type_as(x)
            pe[:, 0::2] = torch.sin(step * self.div_term)
            pe[:, 1::2] = torch.cos(step * self.div_term)
            x = x + pe
        elif type == 'step':
            x = x + self.pe[:, step, :].unsqueeze(1)
        else:
            raise ValueError('Unsupported type of positional embedding.')
        return x

if __name__ == '__main__':
    test = PositionalEncoding(10, 10)
    t1 = test(torch.ones([1, 10, 10]), 0, 'position')
    assert t1.shape == (1, 10, 10)
    t2 = test(torch.ones([1, 10, 10]), 0, 'step')
    assert t2.shape == (1, 10, 10)