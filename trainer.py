from argparse import ArgumentParser
from io import UnsupportedOperation

from data.dataset.bbc_task import get_bbc_file_paths, BBCDataModule, get_bbc_max_seq_len, get_bbc_max_summary_len
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import PyTorchProfiler


from train_lightning.train_utwrs import UTWRS
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

SRC_PAD_TOKEN = -1
TRG_PAD_TOKEN = -1

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
    max_seq_length = 0
    max_summary_length = 0
    if args.dataset == "BBC":
        file_paths = get_bbc_file_paths(args.base_folder)
        max_seq_length = get_bbc_max_seq_len(args.base_folder)
        max_summary_length = get_bbc_max_summary_len(args.base_folder)
    elif args.dataset == "OVSD":
        pass
    else:
        raise UnsupportedOperation("--dataset only support BBC or OVSD.")

    # ------------
    # data args
    # ------------
    # Add <START> and <END> token
    args.enc_seq_len = max_seq_length + 2
    args.dec_seq_len = max_summary_length + 2
    
    # ------------
    # Split train/test
    # ------------
    print(f"\nTotal number of videos: {len(file_paths)}")
    print(f"Max length of videos: {max_seq_length}")
    print(f"Max length of summary: {max_summary_length}\n")

    np.random.shuffle(file_paths)

    train_paths = np.array(file_paths[:-2])

    # ------------
    # K-fold
    # ------------

    kfold = StratifiedKFold(n_splits=3, shuffle=False)

    for k, (train, val) in enumerate(tqdm(kfold.split(np.zeros(len(train_paths)), np.zeros(len(train_paths))), total=kfold.get_n_splits())):
        print(f"Training data: f{train_paths[train]}")
        print(f"Validation data: f{train_paths[val]}")
        # ------------
        # data loader
        # ------------
        
        data_loader = BBCDataModule(max_seq_length, max_summary_length, args.d_model, args.base_folder, train_paths[train], train_paths[val])

        # ------------
        # model
        # ------------

        model = UTWRS(args, SRC_PAD_TOKEN, TRG_PAD_TOKEN)

        # ------------
        # training
        # ------------
        profiler = PyTorchProfiler(output_filename=f"{k}-fold", use_cuda=True, profile_memory=True, sort_by_key="cuda_memory_usage", row_limit=50)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/', name=f"{k}-fold")
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, profiler=profiler)
        trainer.fit(model, data_loader)


if __name__ == '__main__':
    cli_main()