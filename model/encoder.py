import torch.nn as nn
import torch

from .attention import MultiHeadAttention
from .residential import Residential
from .embedding import PositionalEncoding


class UTransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_model, h, dropout=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.layer_norm = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.residential = Residential()
        self.dropout = nn.Dropout(dropout)
        self.transition = nn.Linear(d_model, d_model)
        self.pos_embedding = PositionalEncoding(d_model, seq_len)

    def forward(self, source, t, source_mask):
        x = self.pos_embedding(source, t)

        x = self.residential(x, self.attention(x, x, x, source_mask))
        x = self.dropout(x)
        x = self.layer_norm(x)

        x = self.residential(x, self.transition(x))
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x
