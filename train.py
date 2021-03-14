from argparse import ArgumentParser
from typing import List, Any
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from data.dataset.qa_task import BabiQADataset
from data.vocab.word import WordVocab

from model.qa_transformer import UniversalTransformer

class UTWRS(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = UniversalTransformer(
            enc_seq_len=self.hparams.enc_seq_len,
            dec_seq_len=self.hparams.dec_seq_len, 
            d_model=self.hparams.d_model,
            n_enc_vocab=self.hparams.n_enc_vocab,
            n_dec_vocab=self.hparams.n_dec_vocab,
            h=self.hparams.n_heads,
            t_steps=self.hparams.t_steps,
            dropout=self.hparams.dropout,
            sos_index=self.hparams.sos_index
        )

        self.total_nelement = 0
        self.total_correct = 0.0

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def on_train_epoch_start(self) -> None:
        self.total_nelement = 0
        self.total_correct = 0.0
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        story, _, answer = batch["story"], batch["query"], batch["answer"]
        story_mask, answer_mask = batch["story_mask"], batch["answer_mask"]

        output = self.model(story, answer, story_mask, answer_mask)
        loss = F.nll_loss(output.transpose(-1, 1), answer)
        output_word = output.exp().argmax(dim=-1)
        correct = output_word.eq(answer).sum().float()
        acc = correct / answer.nelement() * 100

        self.total_correct += correct.detach()
        self.total_nelement += story.size(0)

        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = self.total_correct / self.total_nelement * 100.0
        tensorboard_logs = {'train_acc_epoch': avg_acc, 'train_loss_epoch': avg_loss}
        self.logger.agg_and_log_metrics(tensorboard_logs, step=self.current_epoch)

    def on_test_epoch_start(self) -> None:
        self.total_nelement = 0
        self.total_correct = 0.0
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        story, _, answer = batch["story"], batch["query"], batch["answer"]
        story_mask, answer_mask = batch["story_mask"], batch["answer_mask"]

        output = self.model(story, answer, story_mask, answer_mask)
        loss = F.nll_loss(output.transpose(-1, 1), answer)
        output_word = output.exp().argmax(dim=-1)
        correct = output_word.eq(answer).sum().float()

        self.total_correct += correct.detach()
        self.total_nelement += story.size(0)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = self.total_correct / self.total_nelement * 100.0

        tensorboard_logs = {'test_loss': test_loss_mean, "test_acc": test_acc_mean, 'step': self.current_epoch}
        self.logger.agg_and_log_metrics(tensorboard_logs, step=self.current_epoch)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--d_model', default=64, type=int)
        parser.add_argument('--dropout', default=0, type=float)
        parser.add_argument('--n_heads', default=8, type=int)
        parser.add_argument('--t_steps', default=4, type=int)
        parser.add_argument('--sos_index', default=1, type=int)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = UTWRS.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    # ------------
    # data
    # ------------
    word_vocab = WordVocab.load_vocab("babi-qa/vocab/task1_vocab.pkl")
    answer_vocab = WordVocab.load_vocab("babi-qa/vocab/task1_answer_vocab.pkl")

    babiqa_train = BabiQADataset("babi-qa/task1_train.txt", word_vocab, answer_vocab, story_len=14, seq_len=6)
    babiqa_test = BabiQADataset("babi-qa/task1_test.txt", word_vocab, answer_vocab, story_len=14, seq_len=6)

    # ------------
    # data args
    # ------------
    parser.add_argument('--enc_seq_len', default=14, type=int)
    parser.add_argument('--dec_seq_len', default=1, type=int)
    parser.add_argument('--n_enc_vocab', default=len(word_vocab), type=int)
    parser.add_argument('--n_dec_vocab', default=len(word_vocab), type=int)
    

    # ------------
    # data loader
    # ------------
    args = parser.parse_args()
    train_loader = DataLoader(babiqa_train, batch_size=args.batch_size)
    test_loader = DataLoader(babiqa_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    
    model = UTWRS(args)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
