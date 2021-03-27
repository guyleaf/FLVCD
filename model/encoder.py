import torch.nn as nn
import torch

from .attention import MultiHeadAttention
from .residential import Residential
from .position_wise_feedforward import PositionwiseFeedForward


class UTransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, dropout=0.5):
        super(UTransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.layer_norm = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.residential = Residential()
        self.dropout = nn.Dropout(dropout)
        self.transition = PositionwiseFeedForward(d_model, d_inner, d_model, layer_config, padding, transition_dropout)

    def forward(self, source, source_mask):
        x = source

        x = self.residential(x, self.attention(x, x, x, source_mask))
        x = self.dropout(x)
        x = self.layer_norm(x)

        x = self.residential(x, self.transition(x))
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x
