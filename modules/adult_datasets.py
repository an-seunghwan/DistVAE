#%%
import tqdm
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.data_transformer import DataTransformer
#%%
"""
load dataset: Adult
Reference: https://archive.ics.uci.edu/ml/datasets/Adult
"""
class TabularDataset(Dataset): 
    def __init__(self, config, random_state=0):
        # if config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        self.continuous = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.discrete = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'income']
        df = df[self.continuous + self.discrete]
        
        df = df.iloc[:40000, ]
        
        transformer = DataTransformer()
        transformer.fit(df, discrete_columns=self.discrete, random_state=random_state)
        train_data = transformer.transform(df)
        self.transformer = transformer
        
        self.x_data = train_data
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%
class TestTabularDataset(Dataset): 
    def __init__(self, config, random_state=0):
        # if config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        self.continuous = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.discrete = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'income']
        df = df[self.continuous + self.discrete]
        
        df_ = df.iloc[:40000, ]
        df = df.iloc[40000:, ]
        
        transformer = DataTransformer()
        transformer.fit(df_, discrete_columns=self.discrete, random_state=random_state)
        test_data = transformer.transform(df)
        self.transformer = transformer
        
        self.x_data = test_data
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%