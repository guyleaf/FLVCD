from model.embedding import PositionalEncoding
import torch
import torch.nn as nn
from .encoder import UTransformerEncoder
from .decoder import UTransformerDecoder
from .ACT import ACT


class UTEncoder(nn.Module):
    def __init__(self, enc_seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, t_steps=5, dropout=0.5, enc_act_epilson=0.1):
        super(UTEncoder, self).__init__()

        enc_sinusoid_emb = PositionalEncoding(d_model, enc_seq_len)
        self.enc_1_act = ACT(
            UTransformerEncoder(enc_seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, dropout), 
            d_model, 
            t_steps, 
            enc_sinusoid_emb,
            enc_sinusoid_emb, 
            enc_act_epilson)

    def forward(self, enc_input, src_mask):
        x = enc_input

        x, _ = self.enc_1_act(x, source_mask=src_mask)

        return x

class UTDecoder(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, t_steps=5, dropout=0.5, dec_act_epilson=0.1):
        super(UTDecoder, self).__init__()

        decoder = UTransformerDecoder(dec_seq_len, d_model, d_inner, h, layer_config, padding, transition_dropout, dropout)
        dec_sinusoid_emb = PositionalEncoding(d_model, dec_seq_len)
        self.dec_act = ACT(decoder, d_model, t_steps, dec_sinusoid_emb, dec_sinusoid_emb, dec_act_epilson)

        # TODO: Add parameter for linear output
        self.output_fc = nn.Linear(d_model, enc_seq_len, bias=False)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, enc_output, dec_input, src_mask, tgt_mask):
        y, _ = self.dec_act(dec_input, enc_output, src_mask, tgt_mask)
        y = self.output_fc(y)
        return y