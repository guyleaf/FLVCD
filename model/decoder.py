import torch.nn as nn
import torch

from .attention import MultiHeadAttention
from .residential import Residential
from .position_wise_feedforward import PositionwiseFeedForward


class UTransformerDecoder(nn.Module):
    def __init__(self, seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, dropout=0.5):
        super(UTransformerDecoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, h)
        self.layer_norm1 = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.layer_norm2 = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.layer_norm3 = nn.LayerNorm(torch.Size([seq_len, d_model]))
        self.residential = Residential()
        self.dropout = nn.Dropout(dropout)
        self.transition = PositionwiseFeedForward(d_model, d_inner, d_model, layer_config, padding, transition_dropout)

    def forward(self, encoder_output, target, source_mask, target_mask):
        x, y = encoder_output, target

        y = self.residential(y, self.dropout(self.attention(y, y, y, mask=target_mask)))

        y = self.layer_norm1(y)

        x = self.residential(y, self.dropout(self.attention(y, x, x, mask=source_mask)))

        x = self.layer_norm2(x)

        x = self.residential(x, self.dropout(self.transition(x)))

        x = self.layer_norm3(x)

        return x
