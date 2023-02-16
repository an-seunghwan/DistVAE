#%%
"""
Data Source: https://www.kaggle.com/datasets/teertha/personal-loan-modeling
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
        base = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        
        self.continuous = [
            'Age', # target variable
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        self.discrete = [
            'Family',
            'Personal Loan', # target variable
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        self.integer = [
            'Age',
            'Experience',
            'Income', 
            'Mortgage']
        base = base[self.continuous + self.discrete]
        base = base.dropna()
        
        self.RegTarget = 'Age'
        self.ClfTarget = 'Personal Loan'
        
        # one-hot encoding
        df_dummy = []
        for d in self.discrete:
            df_dummy.append(pd.get_dummies(base[d], prefix=d))
        base_dummy = pd.concat([base.drop(columns=self.discrete)] + df_dummy, axis=1)
        
        split_num = 4000
        
        if train:
            self.train_raw = base.iloc[:split_num]
            
            df = base_dummy.iloc[:split_num] # train
            
            self.mean = df[self.continuous].mean(axis=0)
            self.std = df[self.continuous].std(axis=0)
            
            df[self.continuous] = df[self.continuous] - self.mean
            df[self.continuous] /= self.std
            
            self.train = df
            self.x_data = df.to_numpy()
        else:
            self.train_raw = base.iloc[:split_num]
            self.test_raw = base.iloc[split_num:]
            
            df_train = base_dummy.iloc[:split_num] # train
            df = base_dummy.iloc[split_num:] # test
            
            self.mean = df_train[self.continuous].mean(axis=0)
            self.std = df_train[self.continuous].std(axis=0)
            
            df[self.continuous] = df[self.continuous] - self.mean
            df[self.continuous] /= self.std
            
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