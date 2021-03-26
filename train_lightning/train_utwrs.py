from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model.wrs_transformer import UTEncoder, UTDecoder
from model.utils import get_pad_mask, get_self_attention_mask

class UTWRS(pl.LightningModule):

    def __init__(self, hparams, src_pad_idx=0, trg_pad_idx=0):
        super().__init__()
        self.hparams = hparams
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = UTEncoder(
            enc_seq_len=self.hparams.enc_seq_len,
            d_model=self.hparams.d_model,
            d_inner=self.hparams.d_inner,
            h=self.hparams.n_heads,
            t_steps=self.hparams.t_steps,
            dropout=self.hparams.dropout,
            enc_act_epilson=self.hparams.enc_act_epilson
        )
        
        self.decoder = UTDecoder(
            enc_seq_len=self.hparams.enc_seq_len,
            dec_seq_len=self.hparams.dec_seq_len,
            d_model=self.hparams.d_model,
            d_inner=self.hparams.d_inner,
            h=self.hparams.n_heads,
            t_steps=self.hparams.t_steps,
            dropout=self.hparams.dropout,
            dec_act_epilson=self.hparams.dec_act_epilson
        )
        self.save_hyperparameters()

    def training_step(self, data, batch_idx):
        enc_input, dec_input, ground_truth, weight = data['encoder'], data['decoder'], data['target'], data['weight']
        # pad == 1
        src_mask = get_pad_mask(enc_input, self.src_pad_idx)
        enc_output = self.encoder(enc_input, src_mask)

        src_mask = get_pad_mask(dec_input, self.trg_pad_idx)
        trg_mask = get_self_attention_mask(dec_input)
        output = self.decoder(enc_output, dec_input, src_mask, trg_mask)

        weight = weight.squeeze()
        loss = F.cross_entropy(output.squeeze(0), ground_truth.squeeze()) * weight
        loss = loss.sum()/weight.sum()

        self.log('train_loss', loss.detach().cpu().numpy(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        enc_input, dec_input, ground_truth, weight = data['encoder'], data['decoder'], data['target'], data['weight']
        src_mask = get_pad_mask(enc_input, self.src_pad_idx)
        enc_output = self.encoder(enc_input, src_mask)

        src_mask = get_pad_mask(dec_input, self.trg_pad_idx)
        trg_mask = get_self_attention_mask(dec_input)
        output = self.decoder(enc_output, dec_input, src_mask, trg_mask)

        weight = weight.squeeze()
        loss = F.cross_entropy(output.squeeze(0), ground_truth.squeeze()) * weight
        loss = loss.sum()/weight.sum()

        self.log('test_loss', loss.detach().cpu().numpy(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--d_model', default=64, type=int)
        parser.add_argument('--d_inner', default=256, type=int)
        parser.add_argument('--dropout', default=0, type=float)
        parser.add_argument('--n_heads', default=8, type=int)
        parser.add_argument('--t_steps', default=4, type=int)
        parser.add_argument('--enc_act_epilson', default=0.1, type=float)
        parser.add_argument('--dec_act_epilson', default=0.1, type=float)
        return parser