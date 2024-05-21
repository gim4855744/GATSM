from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torch import Tensor
from torch.optim import AdamW

__all__ = ['BaseLitModel']


class BaseLitModel(pl.LightningModule):

    def __init__(
        self,
        task: str,
        n_outputs: int,
        lr: float,
        weight_decay: float
    ):

        super().__init__()

        self._task1, self._task2 = task.split(':')

        self._total_prediction = []
        self._total_target = []

        self._lr = lr
        self._weight_decay = weight_decay
        
        task1, task2 = task.split(':')
        if task2 == 'bincls':
            self._loss_fn = nn.BCEWithLogitsLoss()
            self._score_fn = tm.AUROC(task='binary')
        elif task2 == 'cls':
            self._loss_fn = nn.CrossEntropyLoss()
            self._score_fn = tm.Accuracy(task='multiclass', num_classes=n_outputs)
        else:
            self._loss_fn = nn.MSELoss()
            self._score_fn = tm.R2Score()

    def forward(
        self,
        x
    ):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self._lr, weight_decay=self._weight_decay)
        return {
            'optimizer': optimizer,
            'monitor': 'val_loss',
            'interval': 'epoch'
        }
    
    def training_step(
        self,
        batch,
        batch_idx
    ) -> Tensor:

        self.train()

        x, y, t = batch
        prediction = self(x, t)

        if self._task1 == 'm2m':
            mask = torch.arange(x.size(1), device=x.device).expand(y.shape[:-1]) < t[:, None]
            mask = mask.unsqueeze(dim=-1)
            y = y.masked_select(mask)
            prediction = prediction.masked_select(mask)

        if self._task2 == 'cls':
            y = y.squeeze(dim=-1)

        loss = self._loss_fn(prediction, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        score = self._score_fn(prediction, y)
        self.log('train_score', score, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(
        self,
        batch,
        batch_idx
    ):

        self.eval()

        x, y, t = batch
        prediction = self(x, t)

        if self._task1 == 'm2m':
            mask = torch.arange(x.size(1), device=x.device).expand(y.shape[:-1]) < t[:, None]
            mask = mask.unsqueeze(dim=-1)
            y = y.masked_select(mask)
            prediction = prediction.masked_select(mask)

        if self._task2 == 'cls':
            y = y.squeeze(dim=-1)

        loss = self._loss_fn(prediction, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        score = self._score_fn(prediction, y)
        self.log('val_score', score, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(
        self,
        batch,
        batch_idx
    ):
        
        self.eval()

        x, y, t = batch
        prediction = self(x, t)

        if self._task1 == 'm2m':
            mask = torch.arange(x.size(1), device=x.device).expand(y.shape[:-1]) < t[:, None]
            mask = mask.unsqueeze(dim=-1)
            y = y.masked_select(mask)
            prediction = prediction.masked_select(mask)

        if self._task2 == 'cls':
            y = y.squeeze(dim=-1)

        self._total_prediction.append(prediction)
        self._total_target.append(y)

    def on_test_epoch_end(self):

        self.eval()

        total_prediction = torch.concat(self._total_prediction, dim=0)
        total_target = torch.concat(self._total_target, dim=0)

        loss = self._loss_fn(total_prediction, total_target)
        score = self._score_fn(total_prediction, total_target)

        self.log('loss', loss)
        self.log('score', score)
        
        self._total_prediction.clear()
        self._total_target.clear()
