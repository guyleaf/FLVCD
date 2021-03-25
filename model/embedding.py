import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, seq_len, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
        super().__init__()
        self.register_buffer("seq_len", seq_len)
        self.register_buffer("d_model", d_model)

        # Compute the positional encodings once in log space.
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("div_term", div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, step):
        x = x + self.pe[:, :x.size(1)]
        pe = torch.zeros(self.seq_len, self.d_model).type_as(x)
        pe[:, 0::2] = torch.sin(step * self.div_term)
        pe[:, 1::2] = torch.cos(step * self.div_term)
        x = x + pe
        return x
