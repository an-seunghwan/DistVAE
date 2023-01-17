#%%
"""
Data Source: https://www.kaggle.com/datasets/uciml/adult-census-income
"""
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

from collections import namedtuple

OutputInfo = namedtuple('OutputInfo', ['dim', 'activation_fn'])
#%%
class TabularDataset(Dataset): 
    def __init__(self, train=True):
        base = pd.read_csv('./data/adult.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        base = base[(base == '?').sum(axis=1) == 0]
        
        self.continuous = [
            'age', # target variable
            'educational-num',
            'capital-gain', 
            'capital-loss',
            'hours-per-week',
        ]
        self.discrete = [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native-country',
            'income', # target variable
        ]
        base = base[self.continuous + self.discrete]
        base = base.dropna()
        # [len(base[d].value_counts()) for d in self.discrete]
        
        # one-hot encoding
        df_dummy = []
        for d in self.discrete:
            df_dummy.append(pd.get_dummies(base[d], prefix=d))
        base = pd.concat([base.drop(columns=self.discrete)] + df_dummy, axis=1)
        
        if train:
            df = base.iloc[:40000] # train
            
            df[self.continuous] = (df[self.continuous] - df[self.continuous].mean(axis=0))
            df[self.continuous] /= df[self.continuous].std(axis=0)
            
            self.train = df
            self.x_data = df.to_numpy()
        else:
            df_train = base.iloc[:40000] # train
            df = base.iloc[40000:] # test
            
            df[self.continuous] = (df[self.continuous] - df_train[self.continuous].mean(axis=0))
            df[self.continuous] /= df_train[self.continuous].std(axis=0)
            
            self.test = df
            self.x_data = df.to_numpy()
        
        # Output Information
        self.OutputInfo_list = []
        for c in self.continuous:
            self.OutputInfo_list.append(OutputInfo(1, 'CRPS'))
        for d, dummy in zip(self.discrete, df_dummy):
            self.OutputInfo_list.append(OutputInfo(dummy.shape[1], 'softmax'))
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%