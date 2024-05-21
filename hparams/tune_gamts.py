import warnings
import logging
import sys
import os
import argparse
sys.path.append(os.getcwd())

import numpy as np
from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from gamts.data import get_dataset
from gamts.preprocessor import Preprocessor
from gamts.dataloader import get_time_dataloader
from gamts.models import *
from gamts.utils import save_pickle

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--datapath', type=str)  # required only for PhysioNet2012 and PhysioNet2019
parser.add_argument('--device', type=int, required=True)
args = parser.parse_args()

BATCH_SIZE = {
    'PhysioNet2012': 512,
    'PhysioNet2019': 512,
    'AppliancesEnergy': 32,
    'AustraliaRainfall': 32768,
    'BeijingPM10Quality': 4096,
    'Heartbeat': 64,
    'LSST': 1024,
    'NATOPS': 64
}

x, y, task = get_dataset(args.dataset, args.datapath)


def objective(trial):

    n_splits = 5
    losses = []

    for seed in range(1, n_splits + 1):

        pl.seed_everything(seed)

        batch_size = BATCH_SIZE[args.dataset]
        nbm_hidden_dims = [256, 256, 128]
        nbm_n_bases = 100
        nbm_batchnorm = trial.suggest_categorical('nbm_batchnorm', [False, True])
        nbm_dropout = trial.suggest_float('nbm_dropout', 0, 0.9)
        attn_emb_size = trial.suggest_int('attn_emb_size', 8, 128)
        attn_n_heads = trial.suggest_int('attn_n_heads', 1, 8)
        attn_dropout = trial.suggest_float('attn_dropout', 0, 0.9)
        lr = trial.suggest_float('lr', 1e-3, 1e-2)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        
        indices = x.index.unique()
        n_test = len(indices) // 5
        train_indices, _ = train_test_split(indices, test_size=n_test, random_state=seed)
        train_indices , val_indices = train_test_split(train_indices, test_size=n_test, random_state=seed)

        preprocessor = Preprocessor(task)
        x_train, y_train = preprocessor.fit_transform(x.loc[train_indices], y.loc[train_indices])
        x_val, y_val = preprocessor.transform(x.loc[val_indices], y.loc[val_indices])

        train_dataloader = get_time_dataloader(x_train, y_train, task, batch_size, shuffle=True)
        val_dataloader = get_time_dataloader(x_val, y_val, task, batch_size)

        n_features = x.shape[1]
        if task.split(':')[1] == 'cls':
            n_outputs = len(np.unique(y.values))
        else:
            n_outputs = y.shape[1]

        model = GAMTS(
            task,
            n_features,
            n_outputs,
            nbm_hidden_dims,
            nbm_n_bases,
            nbm_batchnorm,
            nbm_dropout,
            attn_emb_size,
            attn_n_heads,
            attn_dropout,
            lr,
            weight_decay
        )

        ckpt_dir = './checkpoints/tune'
        ckpt_filename = f'GAMTS-{args.dataset}'
        ckpt_path = os.path.join(ckpt_dir, f'{ckpt_filename}.ckpt')

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        model_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_filename,
            monitor='val_loss'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=30
        )
        callbacks = [model_checkpoint, early_stopping]

        trainer = pl.Trainer(devices=[args.device], logger=False, max_epochs=-1, enable_progress_bar=False, callbacks=callbacks)
        trainer.fit(model, train_dataloader, val_dataloader)
        test = trainer.test(model, val_dataloader, ckpt_path, verbose=False)
        loss = test[0]['loss']
        losses.append(loss)

    return sum(losses) / n_splits


def main():
    
    n_trials = 100
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials, show_progress_bar=True)
    print(study.best_params)
    save_pickle(f'hparams/GAMTS-{args.dataset}.pkl', study.best_params)


if __name__ == '__main__':
    main()
