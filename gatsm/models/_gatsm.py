from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseLitModel

__all__ = ['GATSM']


class TimeNBM(nn.Module):

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        n_bases: int,
        batchnorm: bool,
        dropout: float
    ):

        super().__init__()
        
        hidden_dims = [1] + hidden_dims + [n_bases]

        layers = []
        for in_size, out_size in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Conv2d(in_size, out_size, kernel_size=1))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ELU())
        self._layers = nn.Sequential(*layers)

        self._featurizer = nn.Conv1d(
            in_channels=n_features * n_bases,
            out_channels=n_features,
            kernel_size=1,
            groups=n_features
        )

    def forward(self, x: Tensor):
        x = x.unsqueeze(dim=1)  # shape: [batch_size, 1, n_steps, n_features]
        x = self._layers(x)  # shape: [batch_size, n_bases, n_steps, n_features]
        x = x.permute(0, 3, 1, 2).flatten(start_dim=1, end_dim=2)  # shape: [batch_size, n_features * n_bases, n_steps]
        x = self._featurizer(x)  # shape: [batch_size, n_features, n_steps]
        x = x.transpose(2, 1)  # shape: [batch_size, n_steps, n_features]
        return x


class TimeMHA(nn.Module):

    def __init__(self, in_size, n_heads, dropout):
        super().__init__()
        self._q_linears = nn.ModuleList([nn.Conv1d(in_size, out_channels=1, kernel_size=1) for _ in range(n_heads)])
        self._k_linears = nn.ModuleList([nn.Conv1d(in_size, out_channels=1, kernel_size=1) for _ in range(n_heads)])
        self._dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):

        attn_scores, outputs = [], []

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)

        for q_linear, k_linear in zip(self._q_linears, self._k_linears):

            q = q_linear(query).transpose(2, 1)
            k = k_linear(key)

            qk = F.leaky_relu(q + k)
            mask = torch.ones_like(qk, dtype=torch.bool).triu(diagonal=1)
            qk = qk.masked_fill(mask, -1e11)

            a = F.softmax(qk, dim=2)
            a = self._dropout(a)
            o = a @ value

            attn_scores.append(a)
            outputs.append(o)

        attn_scores = torch.concat(attn_scores, dim=2)
        outputs = torch.concat(outputs, dim=2)

        return attn_scores, outputs


class GATSM(BaseLitModel):

    def __init__(
        self,
        task: str,
        n_features: int,
        n_outputs: int,
        nbm_hidden_dims: list[int],
        nbm_n_bases: int,
        nbm_batchnorm: bool,
        nbm_dropout: float,
        attn_emb_size: int,
        attn_n_heads: int,
        attn_dropout: float,
        lr: float,
        weight_decay: float
    ):
        
        self.save_hyperparameters()
        
        super().__init__(task, n_outputs, lr, weight_decay)

        self._task1, self._task2 = task.split(':')

        self._n_features = n_features
        self._attn_emb_size = attn_emb_size
        
        self._featurenet = TimeNBM(n_features, nbm_hidden_dims, nbm_n_bases, nbm_batchnorm, nbm_dropout)  # output shape: [batch_size, n_steps, n_features]
        self._proj_layer = nn.Linear(n_features, attn_emb_size, bias=False)
        self._mha = TimeMHA(attn_emb_size, attn_n_heads, attn_dropout)
        self._output_layer = nn.Linear(attn_n_heads * n_features, n_outputs)

    def forward(
        self,
        x: Tensor,
        t: Tensor
    ):
        
        x = self._featurenet(x)
        v = self._proj_layer(x)

        pe = self._get_positional_encoding(x.size(1))
        v = v + pe
        _, outputs = self._mha(v, v, x)

        outputs = self._output_layer(outputs)
        if self._task1 == 'm2o':
            outputs = outputs[torch.arange(outputs.size(0)), t]

        return outputs
    
    def get_contributions(
        self,
        x: Tensor,
        c: int
    ):
        
        self.eval()
        
        x = self._featurenet(x)
        v = self._proj_layer(x)

        pe = self._get_positional_encoding(x.size(1))
        v = v + pe
        attn_scores, outputs = self._mha(v, v, x)

        n_heads = self._output_layer.weight.size(1) // x.size(2)
        x = x.repeat(1, 1, n_heads)
        
        outputs = outputs * self._output_layer.weight[c]
        x = x * self._output_layer.weight[c]
        
        attn_scores = attn_scores.tensor_split(n_heads, dim=2)
        attn_scores = torch.stack(attn_scores, dim=2)
        
        outputs = outputs.tensor_split(n_heads, dim=2)
        outputs = torch.stack(outputs, dim=2)

        x = x.tensor_split(n_heads, dim=2)
        x = torch.stack(x, dim=2)

        attn_scores = attn_scores.mean(dim=2)
        outputs = outputs.sum(dim=2)  # weighted sum
        x = x.mean(dim=2)  # uniform sum
        
        return {
            'time_importance': attn_scores.detach().cpu().numpy(),
            'dynamic_contributions': outputs.detach().cpu().numpy(),
            'static_contributions': x.detach().cpu().numpy(),
            'base_contribution': self._output_layer.bias[c].detach().cpu().numpy()
        }
    
    def _get_positional_encoding(self, n_steps):

        position = torch.arange(n_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self._attn_emb_size, step=2) * (-log(10000) / self._attn_emb_size))
        s = position * div_term

        pe = torch.zeros(1, n_steps, self._attn_emb_size, device=self.device)
        pe[:, :, 0::2] = torch.sin(s)
        if pe.size(2) % 2 == 0:
            pe[:, :, 1::2] = torch.cos(s)
        else:
            pe[:, :, 1::2] = torch.cos(s[:, :-1])
            
        return pe
