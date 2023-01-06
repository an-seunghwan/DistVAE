#%%
"""
Data Source: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
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
    def __init__(self):
        base = pd.read_csv('./data/covtype.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        base = base.iloc[:50000]
        
        self.continuous = [
            'Elevation', 
            'Aspect', 
            'Slope',
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
        ]
        self.discrete = [
            'Cover_Type', # target variable
        ]
        df = base[self.continuous + self.discrete]
        df = df.dropna(axis=0)
        
        df = df.iloc[:45000] # train
        
        df[self.continuous] = (df[self.continuous] - df[self.continuous].mean(axis=0))
        df[self.continuous] /= df[self.continuous].std(axis=0)
        
        # one-hot encoding
        df_dummy = []
        for d in self.discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        
        # Output Information
        self.OutputInfo_list = []
        for c in self.continuous:
            self.OutputInfo_list.append(OutputInfo(1, 'CRPS'))
        for d, dummy in zip(self.discrete, df_dummy):
            self.OutputInfo_list.append(OutputInfo(dummy.shape[1], 'softmax'))
        
        df = pd.concat([df.drop(columns='Cover_Type')] + df_dummy, axis=1)
        
        self.train = df
        self.x_data = df.to_numpy()
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%
class TestTabularDataset(Dataset): 
    def __init__(self):
        base = pd.read_csv('./data/covtype.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        base = base.iloc[:50000]
        
        self.continuous = [
            'Elevation', 
            'Aspect', 
            'Slope',
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
        ]
        self.discrete = [
            'Cover_Type', # target variable
        ]
        df = base[self.continuous + self.discrete]
        df = df.dropna(axis=0)
        
        df_train = df_train.iloc[:45000] # train
        df = df.iloc[45000:] # test
        
        df[self.continuous] = (df[self.continuous] - df_train[self.continuous].mean(axis=0))
        df[self.continuous] /= df_train[self.continuous].std(axis=0)
        
        # one-hot encoding
        df_dummy = []
        for d in self.discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        
        # Output Information
        self.OutputInfo_list = []
        for c in self.continuous:
            self.OutputInfo_list.append(OutputInfo(1, 'CRPS'))
        for d, dummy in zip(self.discrete, df_dummy):
            self.OutputInfo_list.append(OutputInfo(dummy.shape[1], 'softmax'))
        
        df = pd.concat([df.drop(columns='Cover_Type')] + df_dummy, axis=1)
        
        self.train = df
        self.x_data = df.to_numpy()
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%