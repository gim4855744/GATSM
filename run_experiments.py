import warnings
import logging
import os
import argparse

import yaml
import numpy as np
from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from gamts.data import get_dataset
from gamts.preprocessor import Preprocessor
from gamts.dataloader import get_time_dataloader
from gamts.models import *

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
parser.add_argument('--dataset', type=str, choices=['AppliancesEnergy', 'AustraliaRainfall', 'BeijingPM10Quality',
                                                    'Heartbeat', 'PhysioNet2012', 'PhysioNet2019',
                                                    'LSST', 'NATOPS'], required=True)
parser.add_argument('--datapath', type=str)  # required only for PhysioNet2012 and PhysioNet2019
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()


def main():
    
    pl.seed_everything(args.seed)

    hparams = yaml.load(open(f'hparams/{args.model}.yaml'), yaml.FullLoader)[args.dataset]
    x, y, task = get_dataset(args.dataset, args.datapath)
    
    indices = x.index.unique()
    n_test = len(indices) // 5
    train_indices, test_indices = train_test_split(indices, test_size=n_test, random_state=args.seed)
    train_indices , val_indices = train_test_split(train_indices, test_size=n_test, random_state=args.seed)

    preprocessor = Preprocessor(task)
    x_train, y_train = preprocessor.fit_transform(x.loc[train_indices], y.loc[train_indices])
    x_val, y_val = preprocessor.transform(x.loc[val_indices], y.loc[val_indices])
    x_test, y_test = preprocessor.transform(x.loc[test_indices], y.loc[test_indices])
    n_times = x_train.loc[x_train.index.unique()[0]].shape[0]

    train_dataloader = get_time_dataloader(x_train, y_train, task, hparams['batch_size'], shuffle=True)
    val_dataloader = get_time_dataloader(x_val, y_val, task, hparams['batch_size'])
    test_dataloader = get_time_dataloader(x_test, y_test, task, hparams['batch_size'])
    del hparams['batch_size']

    n_features = x.shape[1]
    if task.split(':')[1] == 'cls':
        n_outputs = len(np.unique(y.values))
    else:
        n_outputs = y.shape[1]

    model = GATSM(
        task,
        n_features,
        n_times,
        n_outputs,
        **hparams
    )

    ckpt_dir = './checkpoints'
    ckpt_filename = f'{args.model}-{args.dataset}-{args.seed}'
    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_filename}.ckpt')

    if args.mode == 'train':

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        model_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_filename,
            monitor='val_loss'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20
        )
        callbacks = [model_checkpoint, early_stopping]

        trainer = pl.Trainer(devices=[2], logger=False, max_epochs=-1, enable_progress_bar=True, callbacks=callbacks)
        trainer.fit(model, train_dataloader, val_dataloader)

    else:

        trainer = pl.Trainer(devices=[2], logger=False, enable_progress_bar=False)

        val_results = trainer.test(model, val_dataloader, ckpt_path, verbose=False)[0]
        val_loss, val_score = val_results['loss'], val_results['score']
        print(f'Val Loss: {val_loss}, Val Score: {val_score}')

        test_results = trainer.test(model, test_dataloader, ckpt_path, verbose=False)[0]
        test_loss, test_score = test_results['loss'], test_results['score']
        print(f'Test Loss: {test_loss}, Test Score: {test_score}')


if __name__ == '__main__':
    main()
