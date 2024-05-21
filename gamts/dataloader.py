import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

__all__ = ['TimeDataset', 'get_time_dataloader']


class TimeDataset(Dataset):

    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        task: str
    ) -> None:
        
        super().__init__()

        self._x, self._y, self._t = [], [], []
        task1, task2 = task.split(':')

        for idx in x.index.unique():

            xi = x.loc[idx].values
            if len(xi.shape) == 1:
                xi = xi.reshape(1, -1)
            yi = y.loc[idx].values

            xi = torch.tensor(xi, dtype=torch.float32)
            if task2 == 'cls':
                yi = torch.tensor(yi, dtype=torch.int64)
            else:
                yi = torch.tensor(yi, dtype=torch.float32)

            self._x.append(xi)
            self._y.append(yi)
            self._t.append(xi.size(0) - 1)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor, int]:
        return self._x[index], self._y[index], self._t[index]
    
    def __len__(self) -> int:
        return len(self._x)


def _pad_batch(
    batch: tuple[list[Tensor], list[Tensor]]
) -> tuple[Tensor, Tensor, Tensor]:
    x, y, t = zip(*batch)
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    t = torch.tensor(t, dtype=torch.int64)
    return x, y, t


def get_time_dataloader(
    x: pd.DataFrame,
    y: pd.DataFrame,
    task: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    dataset = TimeDataset(x, y, task)
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=_pad_batch)
    return dataloader
