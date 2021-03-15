from model.embedding import PositionalEncoding
import torch.nn as nn
import torch.nn.functional as F
import torch

from .encoder import UTransformerEncoder
from .decoder import UTransformerDecoder
from .ACT import ACT


class UniversalTransformer(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_enc_vocab, n_dec_vocab, h, t_steps=5, dropout=0.5,
                 sos_index=1, enc_act_epilson=0.1, dec_act_epilson=0.1):
        super(UniversalTransformer, self).__init__()
        encoder = UTransformerEncoder(enc_seq_len, d_model, h, dropout)
        enc_sinusoid_emb = PositionalEncoding(d_model, enc_seq_len)
        self.enc_act = ACT(encoder, d_model, t_steps, enc_sinusoid_emb, enc_sinusoid_emb, enc_act_epilson)

        decoder = UTransformerDecoder(dec_seq_len, d_model, h, dropout)
        dec_sinusoid_emb = PositionalEncoding(d_model, dec_seq_len)
        self.dec_act = ACT(decoder, d_model, t_steps, dec_sinusoid_emb, dec_sinusoid_emb, dec_act_epilson)

        self.input_embed = nn.Embedding(n_enc_vocab, d_model, padding_idx=0)
        self.target_embed = nn.Embedding(n_dec_vocab, d_model, padding_idx=0)
        self.generator = nn.Linear(d_model, n_dec_vocab)
        
        self.t_steps = t_steps
        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.sos_index = sos_index

    def forward(self, story, answer, story_mask, answer_mask):
        x = self.input_embed(story)

        # Story Word Embedding Sum
        x = x.sum(dim=-2)

        x = self.enc_act(x, source_mask=story_mask)

        sos_tag = torch.zeros(answer.size(), dtype=torch.long, device=story.device).fill_(3)

        y = self.target_embed(sos_tag)
        y = self.dec_act(y, x, story_mask, answer_mask)

        y = F.log_softmax(self.generator(y), dim=-1)
        return y
