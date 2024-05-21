import warnings
import logging
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

import lightning.pytorch as pl

from gamts.data import get_dataset
from gamts.preprocessor import Preprocessor
from gamts.dataloader import get_time_dataloader
from gamts.models import GATSM

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--datapath', type=str)  # required only for PhysioNet2012 and PhysioNet2019
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

feature_names = {
    'BeijingPM10Quality': ('SO2', 'NO2', 'CO', 'O3', 'temperature', 'pressure', 'dew point', 'rainfall', 'windspeed'),
    'AustraliaRainfall': ('Avg. Temperature', 'Max Temperature', 'Min Temperature')
}

def supylabel2(fig, s, **kwargs):
    defaults = {
        "x": 0.98,
        "y": 0.5,
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "rotation": "vertical",
        "rotation_mode": "anchor",
        "size": plt.rcParams["figure.labelsize"],  # matplotlib >= 3.6
        "weight": plt.rcParams["figure.labelweight"],  # matplotlib >= 3.6
    }
    kwargs["s"] = s
    # kwargs = defaults | kwargs  # python >= 3.9
    kwargs = {**defaults, **kwargs}
    fig.text(**kwargs)


def align_yaxis(axes): 

    y_lims = np.array([ax.get_ylim() for ax in axes])

    # force 0 to appear on all axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize all axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lims = y_new_lims_normalized * y_mags
    for i, ax in enumerate(axes):
        ax.set_ylim(new_lims[i])


def visualize_time_importance(
    model,
    dataloader,
    step=-1,
    c=0
):
    
    """
    step: desired time step to interpret
    c: desired class index to interpret
    """

    device = model.device
    
    t = []  # lengths of time series
    time_importance = []  # attention scores of time steps

    for batch_x, _, batch_t in dataloader:

        interpretations = model.get_contributions(batch_x.to(device), c)
        sample_indices = np.arange(batch_x.size(0))
        
        if step == -1:
            t.append(batch_t)
            time_importance.append(interpretations['time_importance'][sample_indices, batch_t])
        else:
            t.append(np.full((batch_x.size(0),), step))
            time_importance.append(interpretations['time_importance'][sample_indices, step])
            
    t = np.concatenate(t, axis=0)
    max_length = max(t) + 1
    time_importance = [np.pad(time_importance_i, ((0, 0), (0, max_length - time_importance_i.shape[1]))) for time_importance_i in time_importance]
    time_importance = np.concatenate(time_importance, axis=0)
    
    interpolated_time_importance = []
    for time_importance_i, t_i in zip(time_importance, t):
        length_i = t_i + 1
        interp_i = interp1d(np.arange(length_i), time_importance_i[:length_i])
        interp_range = np.linspace(0, t_i, max_length)
        interp_value = interp_i(interp_range)
        if np.isnan(interp_value).sum() == 0:
            interpolated_time_importance.append(interp_value)
    time_importance = np.stack(interpolated_time_importance, axis=0)

    time_importance_mean = time_importance.mean(axis=0)
    time_importance_std = time_importance.std(axis=0)
    
    plt.rc('font', size=20)
    plt.fill_between(np.arange(max_length), time_importance_mean - time_importance_std, time_importance_mean + time_importance_std, color='tab:red', alpha=0.3)
    plt.plot(time_importance_mean, color='tab:red')
    plt.xlabel('Time steps')
    plt.ylabel('Avg. attention score')
    plt.tight_layout()
    plt.savefig(f'figures/time_importance_{args.dataset}.pdf')


def visualize_global_static_contribution(
    model,
    dataloader,
    n_features,
    c=0
):
    
    plt.rc('font', size=15)
    device = model.device

    if n_features < 4:
        width = n_features
        height = 1
    else:
        width = 4
        height = n_features // width + (1 if n_features % width > 0 else 0)

    fig, axes = plt.subplots(height, width, figsize=(5 * width, 5 * height))

    for i in range(n_features):

        h = i // width
        w = i % width

        x = []  # input features
        static_contributions = []  # time independent contributions of features

        for batch_x, _, _ in dataloader:
            interpretations = model.get_contributions(batch_x.to(device), c)
            x.append(batch_x[:, :, i].reshape(-1,))
            static_contributions.append(interpretations['static_contributions'][:, :, i].reshape(-1,))

        x = np.concatenate(x, axis=0)
        static_contributions = np.concatenate(static_contributions, axis=0)

        v = zip(x, static_contributions)
        v = sorted(v, key=lambda x: x[0])
        x, static_contributions = zip(*v)
    
        feature_name = feature_names[args.dataset][i]

        x_min = min(x)
        x_max = max(x)
        linspace = np.linspace(x_min, x_max, num=11)
        for start, end in zip(linspace[:-1], linspace[1:]):
            density = sum((x >= start) & (x < end)) / len(x)
            if n_features < 4:
                axes[w].axvspan(start, end, facecolor='tab:red', alpha=density)
            else:
                axes[h][w].axvspan(start, end, facecolor='tab:red', alpha=density)

        if n_features < 4:
            axes[w].plot(x, static_contributions)
            axes[w].set_title(feature_name, fontsize=25)
            axes[w].tick_params(axis='both', labelsize=25)
        else:
            axes[h][w].plot(x, static_contributions)
            axes[h][w].set_title(feature_name, fontsize=25)
            axes[h][w].tick_params(axis='both', labelsize=25)

    if n_features < 4:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-i])
    else:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-1][-i])
            
    plt.rc('font', size=25)
    fig.supxlabel('Feature value')
    fig.supylabel('Feature contribution')
    fig.tight_layout(rect=(0.015, 0, 1, 1))
    plt.savefig(f'figures/global_static_{args.dataset}.pdf')


def visualize_local_static_contribution(
    model,
    dataloader,
    c=0
):
    
    plt.rc('font', size=15)
    device = model.device
    sample_idx = 1

    x, static_contributions = [], []
    for batch_x, _, _ in dataloader:
        interpretations = model.get_contributions(batch_x.to(device), c)
        x.append(batch_x)
        static_contributions.append(interpretations['static_contributions'])

    x = np.concatenate(x, axis=0)
    static_contributions = np.concatenate(static_contributions, axis=0)

    x = x[sample_idx]
    static_contributions = static_contributions[sample_idx]

    n_times, n_features = x.shape

    if n_features < 4:
        width = n_features
        height = 1
    else:
        width = 4
        height = n_features // width + (1 if n_features % width > 0 else 0)
    fig, axes = plt.subplots(height, width, figsize=(5 * width, 5 * height))

    for i in range(n_features):

        h = i // width
        w = i % width
        feature_name = feature_names[args.dataset][i]

        if n_features < 4:
            axes[w].plot(range(n_times), static_contributions[:, i], c='tab:blue')
            axes[w].bar(range(n_times), static_contributions[:, i], color='tab:blue', alpha=0.3)
            axes[w].axhline(0, color='black')
            axes[w].tick_params(axis='y', labelcolor='tab:blue')
        else:
            axes[h][w].plot(range(n_times), static_contributions[:, i], c='tab:blue')
            axes[h][w].bar(range(n_times), static_contributions[:, i], color='tab:blue', alpha=0.3)
            axes[h][w].axhline(0, color='black')
            axes[h][w].tick_params(axis='y', labelcolor='tab:blue')

        if n_features < 4:
            sub_ax = axes[w].twinx()
        else:
            sub_ax = axes[h][w].twinx()
        sub_ax.plot(range(n_times), x[:, i], c='tab:red')
        sub_ax.bar(range(n_times), x[:, i], color='tab:red', alpha=0.3)
        sub_ax.axhline(0, color='black')
        sub_ax.tick_params(axis='y', labelcolor='tab:red')
        
        if n_features < 4:
            align_yaxis([axes[w], sub_ax])
            axes[w].set_title(feature_name)
        else:
            align_yaxis([axes[h][w], sub_ax])
            axes[h][w].set_title(feature_name)
        
    if n_features < 4:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-i])
    else:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-1][-i])

    fig.supxlabel('Time step')
    fig.supylabel('Feature contribution', color='tab:blue')
    supylabel2(fig, 'Feature value', color='tab:red')
    fig.tight_layout(rect=(0.015, 0, 0.96, 1))
    plt.savefig(f'figures/local_static_{args.dataset}.pdf')


def visualize_local_dynamic_contribution(
    model,
    dataloader,
    c=0
):
    
    plt.rc('font', size=15)
    device = model.device
    sample_idx = 1

    x, contributions = [], []
    for batch_x, _, _ in dataloader:
        interpretations = model.get_contributions(batch_x.to(device), c)
        x.append(batch_x)
        contributions.append(interpretations['dynamic_contributions'])

    x = np.concatenate(x, axis=0)
    contributions = np.concatenate(contributions, axis=0)

    x = x[sample_idx]
    contributions = contributions[sample_idx]

    n_times, n_features = x.shape

    if n_features < 4:
        width = n_features
        height = 1
    else:
        width = 4
        height = n_features // width + (1 if n_features % width > 0 else 0)
    fig, axes = plt.subplots(height, width, figsize=(5 * width, 5 * height))

    for i in range(n_features):

        h = i // width
        w = i % width
        feature_name = feature_names[args.dataset][i]

        if n_features < 4:
            axes[w].plot(range(n_times), contributions[:, i], c='tab:blue')
            axes[w].bar(range(n_times), contributions[:, i], color='tab:blue', alpha=0.3)
            axes[w].axhline(0, color='black')
            axes[w].tick_params(axis='y', labelcolor='tab:blue')
        else:
            axes[h][w].plot(range(n_times), contributions[:, i], c='tab:blue')
            axes[h][w].bar(range(n_times), contributions[:, i], color='tab:blue', alpha=0.3)
            axes[h][w].axhline(0, color='black')
            axes[h][w].tick_params(axis='y', labelcolor='tab:blue')

        if n_features < 4:
            sub_ax = axes[w].twinx()
        else:
            sub_ax = axes[h][w].twinx()
        sub_ax.plot(range(n_times), x[:, i], c='tab:red')
        sub_ax.bar(range(n_times), x[:, i], color='tab:red', alpha=0.3)
        sub_ax.axhline(0, color='black')
        sub_ax.tick_params(axis='y', labelcolor='tab:red')

        if n_features < 4:
            align_yaxis([axes[w], sub_ax])
            axes[w].set_title(feature_name)
        else:
            align_yaxis([axes[h][w], sub_ax])
            axes[h][w].set_title(feature_name)
        
    if n_features < 4:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-i])
    else:
        if n_features % width > 0:
            for i in range(1, width - (n_features % width) + 1):
                fig.delaxes(axes[-1][-i])
            
    fig.supxlabel('Time step')
    fig.supylabel('Feature contribution', color='tab:blue')
    supylabel2(fig, 'Feature value', color='tab:red')
    fig.tight_layout(rect=(0.015, 0, 0.96, 1))
    plt.savefig(f'figures/local_dynamic_{args.dataset}.pdf')


def main():
    
    pl.seed_everything(args.seed)
    
    x, y, task = get_dataset(args.dataset, args.datapath)
    feature_names['PhysioNet2019'] = x.columns
    n_features = x.shape[1]
    
    indices = x.index.unique()
    n_test = len(indices) // 5
    train_indices, test_indices = train_test_split(indices, test_size=n_test, random_state=args.seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=n_test, random_state=args.seed)

    preprocessor = Preprocessor(task)
    x_train, y_train = preprocessor.fit_transform(x.loc[train_indices], y.loc[train_indices])
    x_val, y_val = preprocessor.transform(x.loc[val_indices], y.loc[val_indices])
    x_test, y_test = preprocessor.transform(x.loc[test_indices], y.loc[test_indices])

    train_dataloader = get_time_dataloader(x_train, y_train, task, batch_size=512)
    val_dataloader = get_time_dataloader(x_val, y_val, task, batch_size=512)
    test_dataloader = get_time_dataloader(x_test, y_test, task, batch_size=512)

    ckpt_dir = './checkpoints'
    ckpt_filename = f'GAMTS-{args.dataset}-{args.seed}'
    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_filename}.ckpt')

    model = GATSM.load_from_checkpoint(ckpt_path)

    visualize_time_importance(model, train_dataloader)
    visualize_global_static_contribution(model, train_dataloader, n_features)
    visualize_local_static_contribution(model, train_dataloader)
    visualize_local_dynamic_contribution(model, train_dataloader)
    

if __name__ == '__main__':
    main()
