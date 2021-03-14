import torch.nn as nn
import torch.nn.functional as fnn
import torch

from .encoder import UTransformerEncoder
from .decoder import UTransformerDecoder


class UniversalTransformer(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_enc_vocab, n_dec_vocab, h, t_steps=5, dropout=0.5,
                 sos_index=1):
        super().__init__()
        self.encoder = UTransformerEncoder(enc_seq_len, d_model, h, dropout)
        self.decoder = UTransformerDecoder(dec_seq_len, d_model, h, dropout)
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

        for step in range(self.t_steps):
            x = self.encoder(x, step, story_mask)

        sos_tag = torch.zeros(answer.size(), dtype=torch.long, device=story.device).fill_(3)

        y = self.target_embed(sos_tag)
        for step in range(self.t_steps):
            y = self.decoder(x, y, step, story_mask, answer_mask)

        y = fnn.log_softmax(self.generator(y), dim=-1)
        return y
