import torch.nn as nn
import torch.nn.functional as fnn
import torch

from .encoder import UTransformerEncoder
from .decoder import UTransformerDecoder


class UniversalTransformer(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_enc_vocab, n_dec_vocab, h, t_steps=5, dropout=0.5,
                 sos_index=1):
        super(UniversalTransformer, self).__init__()
        self.encoder = UTransformerEncoder(enc_seq_len, d_model, h, dropout)
        self.decoder = UTransformerDecoder(dec_seq_len, d_model, h, dropout)
        self.input_embed = nn.Embedding(n_enc_vocab, d_model, padding_idx=0)
        self.target_embed = nn.Embedding(n_dec_vocab, d_model, padding_idx=0)
        self.generator = nn.Linear(d_model, n_dec_vocab)

        self.t_steps = t_steps
        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.sos_index = sos_index

    def forward(self, source, target=None):
        batch_size, device = source.size(0), source.device

        source_mask, target_mask = source == 0, target == 0 if target is not None else None

        x = self.input_embed(source)

        # Story Word Embedding Sum
        x = x.sum(dim=-2)

        for step in range(self.t_steps):
            x = self.encoder(x, step)

        output_distribution = []
        decoder_input = torch.zeros(source.size(0), 1, device=device).long().fill_(self.sos_index)

        for dec_step in range(self.dec_seq_len):
            if target is not None and dec_step > 0:
                decoder_input = torch.cat([decoder_input, target[:, dec_step].unsqueeze(-1)], dim=-1)

            y = self.input_embed(decoder_input)

            for step in range(self.t_steps):
                y = self.decoder(x, y, step)

            y = fnn.log_softmax(self.generator(y), dim=-1)[:, -1]

            word_idx = y.argmax(dim=-1, keepdim=True)
            output_distribution.append(y)

            if target is None:
                decoder_input = torch.cat([decoder_input, word_idx], dim=-1)

        return torch.cat(output_distribution, dim=-1)
