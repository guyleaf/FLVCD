from argparse import ArgumentParser
from io import UnsupportedOperation

from data.dataset.ovsd_bbc_task import get_file_paths, OVSDBBCDataModule, get_max_seq_len, get_max_summary_len
import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from train_lightning.train_utwrs import UTWRS
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

SRC_PAD_TOKEN = 0
TRG_PAD_TOKEN = 0

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--base_folders', nargs='+', default=[], required=True)
    parser.add_argument('--datasets', nargs='+', default=[], required=True)
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--use_tpu', action="store_true", default=False)
    parser.add_argument('--memory_profile', action="store_true", default=False)
    parser.add_argument('--tags', nargs='*', default=[])
    parser = UTWRS.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data path
    # ------------
    file_paths = []
    max_seq_length = 0
    max_summary_length = 0

    if "BBC" in args.datasets:
        i = args.datasets.index("BBC")
        file_paths.append(get_file_paths(args.base_folders[i]))
        max_seq_length = max(get_max_seq_len(args.base_folders[i]), max_seq_length)
        max_summary_length = max(get_max_summary_len(args.base_folders[i]), max_summary_length)

    if "OVSD" in args.datasets:
        i = args.datasets.index("OVSD")
        file_paths.append(get_file_paths(args.base_folders[i]))
        max_seq_length = max(get_max_seq_len(args.base_folders[i]), max_seq_length)
        max_summary_length = max(get_max_summary_len(args.base_folders[i]), max_summary_length)

    if file_paths == []:
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
    print(f"\nTotal number of videos: {sum([len(i) for i in file_paths])}")
    print(f"Max length of videos: {max_seq_length}")
    print(f"Max length of summary: {max_summary_length}\n")

    train_paths = []
    test_paths = []

    for dataset in file_paths:
        np.random.shuffle(dataset)
        train_paths.extend(dataset[:-2])
        test_paths.extend(dataset[-2:])

    # ------------
    # K-fold
    # ------------

    kfold = StratifiedKFold(n_splits=3, shuffle=False)

    # Generate data index for kfold
    X = [0] * len(train_paths)
    Y = []
    for i, dataset in enumerate(file_paths):
        Y += [i] * (len(dataset) - 2)

    train_paths = np.array(train_paths)
    for k, (train, val) in enumerate(tqdm(kfold.split(X, Y), total=kfold.get_n_splits())):
        print(f"Training data: f{train_paths[train]}")
        print(f"Validation data: f{train_paths[val]}")
        # ------------
        # data loader
        # ------------
        data_loader = OVSDBBCDataModule(
            max_seq_length,
            max_summary_length,
            args.d_model,
            train_paths[train],
            train_paths[val],
            shuffle=args.shuffle,
            use_tpu=args.use_tpu
        )

        # ------------
        # model
        # ------------
        model = UTWRS(args, SRC_PAD_TOKEN, TRG_PAD_TOKEN)

        # ------------
        # neptune logger
        # ------------
        neptune_logger = NeptuneLogger(
            project_name="guyleaf/UTWRS", 
            params=vars(args),
            experiment_name=f"{k+1}-fold_logger",
            tags=args.tags
        )
        neptune_logger.experiment.log_text("training_data", ','.join(train_paths[train]))
        neptune_logger.experiment.log_text("validation_data", ','.join(train_paths[val]))

        # ------------
        # checkpoint
        # ------------
        model_checkpoint = ModelCheckpoint(
            dirpath="checkpoints",
            filename='{epoch:02d}_{test_loss:.2f}',
            save_top_k=3,
            monitor='test_loss',
            mode='min'
        )

        # ------------
        # profiler
        # ------------
        profiler = PyTorchProfiler(
            output_filename=f"profiles/{k}-fold_profiler",
            profile_memory=True,
            sort_by_key="cuda_memory_usage",
            row_limit=50,
            enabled=args.memory_profile
        )

        # ------------
        # training
        # ------------
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=neptune_logger,
            profiler=profiler,
            checkpoint_callback=model_checkpoint,
            track_grad_norm=2,
            log_every_n_steps=100
        )
        trainer.fit(model, data_loader)

        # Log model checkpoint to Neptune
        for k in model_checkpoint.best_k_models.keys():
            model_name = 'checkpoints/' + k.split('/')[-1]
            neptune_logger.experiment.log_artifact(k, model_name)

        # Log score of the best model checkpoint.
        neptune_logger.experiment.set_property('best_model_loss', model_checkpoint.best_model_score.tolist())
        if args.profiler:
            neptune_logger.experiment.log_artifact('profiles')


if __name__ == '__main__':
    cli_main()