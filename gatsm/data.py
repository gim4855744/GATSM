import os
from glob import glob

import pandas as pd
import numpy as np

from tsai.all import get_UCR_multivariate_list, get_Monash_regression_list
from tsai.all import get_UCR_data, get_Monash_regression_data

__all__ = ['get_dataset']


def get_dataset(
    dataset_name: str,
    datadir: str = None
):
    
    ucr_list = get_UCR_multivariate_list()
    monash_list = get_Monash_regression_list()

    if dataset_name in ucr_list:

        x, y, _ = get_UCR_data(dataset_name, return_split=False, parent_dir='./data/')

        task1 = 'm2o'
        if len(np.unique(y)) > 2:
            task2 = 'cls'
        else:
            task2 = 'bincls'
        task = task1 + ':' + task2

    elif dataset_name in monash_list:
        x, y, _ = get_Monash_regression_data(dataset_name, split_data=False, path='./data/')
        task = 'm2o:reg'
    elif dataset_name == 'PhysioNet2012':
        x, y, task = get_physionet2012(datadir)
    elif dataset_name == 'PhysioNet2019':
        x, y, task = get_physionet2019(datadir)

    if dataset_name in ucr_list + monash_list:

        n_samples, n_features, n_steps = x.shape
        x_index = np.repeat(range(n_samples), n_steps)
        y_index = range(n_samples)

        x = pd.DataFrame(x.transpose(0, 2, 1).reshape(-1, n_features), index=x_index)
        y = pd.DataFrame(y, index=y_index)

        x.index.name = 'id'
        y.index.name = 'id'

    return x, y, task


def get_physionet2012(
    datadir: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    
    """
    PhysioNet 2012 challenge dataset.
    This dataset can be downloaded from https://physionet.org/content/challenge-2012/1.0.0/.
    """
    
    categorical_feature_names = ['Gender', 'ICUType']
    numerical_feature_names = [
        'Age', 'Height', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
        'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TropI', 'TropT', 'Urine', 'WBC'
    ]
    feature_names = categorical_feature_names + numerical_feature_names
    n_features = len(feature_names)
    task = 'm2o:bincls'
    
    savedir = './data/PhysioNet2012/'
    x_path = os.path.join(savedir, 'x.csv')
    y_path = os.path.join(savedir, 'y.csv')
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = pd.read_csv(x_path, index_col='RecordID')
        y = pd.read_csv(y_path, index_col='RecordID').astype('object')
        x[categorical_feature_names] = x[categorical_feature_names].astype('object')
        x[numerical_feature_names] = x[numerical_feature_names].astype('float')
        return x, y, task
    
    def read_set(
        setdir: str,
        outcome_file: str
    ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        
        sample_pathname = os.path.join(datadir, setdir + '*.txt')
        sample_paths = glob(sample_pathname)
        sample_paths = sorted(sample_paths)
        outcome_path = os.path.join(datadir, outcome_file)

        samples = [read_sample(sample_path) for sample_path in sample_paths]
        outcomes = pd.read_csv(outcome_path, usecols=['RecordID', 'In-hospital_death'])

        return samples, outcomes

    def read_sample(
        path: str
    ) -> pd.DataFrame:

        df = pd.read_csv(path)
        record_id = int(df.iloc[0]['Value'])
        df.drop(0, axis=0, inplace=True)

        time_steps = sorted(set(df['Time']))
        n_time_steps = len(time_steps)
        transformed_sample = np.full([n_time_steps, n_features], fill_value=np.nan)

        timestep2idx = {time_step: idx for idx, time_step in enumerate(time_steps)}
        featurename2idx = {feature_name: idx for idx, feature_name in enumerate(feature_names)}

        for time_step, feature_name, value in df.values:

            if value > -1 and feature_name is not np.nan:  # drop unmeasured values and unknown feature names
                
                if feature_name == 'TroponinI':  # change wrong feature names
                    feature_name = 'TropI'
                elif feature_name == 'TroponinT':
                    feature_name = 'TropT'

                time_idx = timestep2idx[time_step]
                feature_idx = featurename2idx[feature_name]
                transformed_sample[time_idx, feature_idx] = value

        transformed_sample = pd.DataFrame(transformed_sample, columns=feature_names)
        transformed_sample['RecordID'] = record_id

        return transformed_sample
    
    setdirs = ['set-a/', 'set-b/', 'set-c/']
    outcome_files = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']

    total_samples, total_outcomes = [], []
    for setdir, outcome_file in zip(setdirs, outcome_files):
        samples, outcomes = read_set(setdir, outcome_file)
        total_samples.extend(samples)
        total_outcomes.append(outcomes)

    total_samples = pd.concat(total_samples, axis=0).set_index('RecordID')
    total_outcomes = pd.concat(total_outcomes, axis=0).set_index('RecordID').astype('object')

    total_samples[categorical_feature_names] = total_samples[categorical_feature_names].astype('object')
    total_samples[numerical_feature_names] = total_samples[numerical_feature_names].astype('float')

    os.makedirs(savedir, mode=0o755, exist_ok=True)
    total_samples.to_csv(x_path)
    total_outcomes.to_csv(y_path)

    return total_samples, total_outcomes, task


def get_physionet2019(
    datadir: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    
    """
    PhysioNet 2019 challenge dataset.
    This dataset can be downloaded from https://physionet.org/content/challenge-2019/1.0.0/.
    """
    
    categorical_feature_names = ['Gender', 'Unit1', 'Unit2']
    numerical_feature_names = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2',
        'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
        'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'HospAdmTime', 'ICULOS'
    ]
    task = 'm2m:bincls'

    savedir = './data/PhysioNet2019/'
    x_path = os.path.join(savedir, 'x.csv')
    y_path = os.path.join(savedir, 'y.csv')
    if os.path.exists(x_path) and os.path.join(y_path):
        x = pd.read_csv(x_path, index_col='RecordID')
        y = pd.read_csv(y_path, index_col='RecordID').astype('object')
        x[categorical_feature_names] = x[categorical_feature_names].astype('object')
        x[numerical_feature_names] = x[numerical_feature_names].astype('float')
        return x, y, task
    
    def read_set(setdir: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:

        sample_pathname = os.path.join(datadir, setdir + '*.psv')
        sample_paths = glob(sample_pathname)
        sample_paths = sorted(sample_paths)

        x, y = [], []
        for path in sample_paths:
            df = pd.read_csv(path, sep='|')
            yi = df.get(['SepsisLabel'])
            xi = df.drop(['SepsisLabel'], axis=1)
            x.append(xi)
            y.append(yi)

        return x, y
    
    setdirs = ['training/training_setA/', 'training/training_setB/']

    total_samples, total_outcomes = [], []
    for set_dir in setdirs:
        samples, outcomes = read_set(set_dir)
        total_samples.extend(samples)
        total_outcomes.extend(outcomes)

    for i, (sample, outcome) in enumerate(zip(total_samples, total_outcomes)):
        sample['RecordID'] = i
        outcome['RecordID'] = i
    total_samples = pd.concat(total_samples, axis=0).set_index('RecordID')
    total_outcomes = pd.concat(total_outcomes, axis=0).set_index('RecordID').astype('object')
    
    os.makedirs(savedir, mode=0o755, exist_ok=True)
    total_samples.to_csv(x_path)
    total_outcomes.to_csv(y_path)

    return total_samples, total_outcomes, task
