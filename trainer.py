from argparse import ArgumentParser
from io import UnsupportedOperation

from data.dataset.bbc_task import get_bbc_file_paths, BBCDataModule, get_bbc_max_seq_len
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


from train_lightning.train_utwrs import UTWRS
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tqdm

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
    # Split train/test
    # ------------
    print(f"Total number of videos: {len(file_paths)}")
    print(f"Maximum length of videos: {max_length}\n")

    np.random.shuffle(file_paths)

    train_paths = np.numpy(file_paths[:-2])

    print(f"Training data: f{train_paths}")
    # ------------
    # K-fold
    # ------------

    kfold = StratifiedKFold(n_splits=3, shuffle=False)

    for k, (train, val) in enumerate(tqdm(kfold.split(np.zeros(len(train_paths)), np.zeros(len(train_paths))))):

        # ------------
        # data loader
        # ------------
        
        data_loader = BBCDataModule(args.base_folder, train_paths[train], train_paths[val])

        # ------------
        # model
        # ------------

        model = UTWRS(args, SRC_PAD_TOKEN, TRG_PAD_TOKEN)

        # ------------
        # training
        # ------------
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/', name=f"{k}-fold")
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger)
        trainer.fit(model, data_loader)


if __name__ == '__main__':
    cli_main()