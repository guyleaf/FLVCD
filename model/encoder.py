import torch.nn as nn
import torch

from .attention import MultiHeadAttention
from .residential import Residential
from .embedding import PositionalEncoding
from .ACT import ACT


class UTransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_model, h, dropout=0.5):
        super(UTransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.layer_norm = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.residential = Residential()
        self.dropout = nn.Dropout(dropout)
        self.transition = nn.Linear(d_model, d_model)

    def forward(self, source, source_mask):
        x = source

        x = self.residential(x, self.attention(x, x, x, source_mask))
        x = self.dropout(x)
        x = self.layer_norm(x)

        x = self.residential(x, self.transition(x))
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x
