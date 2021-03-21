from argparse import ArgumentParser
from io import UnsupportedOperation
from data.dataset.bbc_task import get_bbc_file_paths, BBCDataModule, get_bbc_max_seq_len
import pytorch_lightning as pl

from train_lightning.train_utwrs import UTWRS
from sklearn.model_selection import StratifiedKFold


SRC_PAD_TOKEN = 0
TRG_PAD_TOKEN = 0

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--base_folder', default='data', type=str)
    parser.add_argument('--dataset', default='BBC', type=str)
    parser = UTWRS.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # ------------
    # data path
    # ------------
    file_paths = []
    max_length = 0
    if args.dataset == "BBC":
        file_paths = get_bbc_file_paths(args.base_folder)
        max_length = get_bbc_max_seq_len(args.base_folder)
    elif args.dataset == "OVSD":
        pass
    else:
        raise UnsupportedOperation("--dataset only support BBC or OVSD.")

    # ------------
    # data args
    # ------------
    args.enc_seq_len = max_length
    args.dec_seq_len = max_length
    
    # ------------
    # K-fold
    # ------------
    
    # ------------
    # data loader
    # ------------
    
    train_loader = DataLoader(babiqa_train, batch_size=args.batch_size)
    test_loader = DataLoader(babiqa_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    
    model = UTWRS(args, SRC_PAD_TOKEN, TRG_PAD_TOKEN)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    cli_main()