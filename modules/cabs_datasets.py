#%%
"""
Data Source: https://www.kaggle.com/datasets/arashnic/taxi-pricing-with-mobility-analytics?select=test.csv
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
        base = pd.read_csv('./data/sigma_cabs.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        base = base.dropna().reset_index().drop(columns='index')
        
        self.continuous = [
            'Trip_Distance', # target variable
            'Life_Style_Index', 
            'Customer_Rating', 
            'Var1',
            'Var2',
            'Var3',
        ]
        self.discrete = [
            'Type_of_Cab',
            'Customer_Since_Months',
            'Confidence_Life_Style_Index',
            'Destination_Type',
            'Cancellation_Last_1Month',
            'Gender',
            'Surge_Pricing_Type', # target variable
        ]
        base = base[self.continuous + self.discrete]
        # [len(base[d].value_counts()) for d in discrete]
        
        # one-hot encoding
        df_dummy = []
        for d in self.discrete:
            df_dummy.append(pd.get_dummies(base[d], prefix=d))
        base = pd.concat([base.drop(columns=self.discrete)] + df_dummy, axis=1)
        
        if train:
            df = base.iloc[:40000] # train
            
            self.mean = df[self.continuous].mean(axis=0)
            self.std = df[self.continuous].std(axis=0)
            
            df[self.continuous] = df[self.continuous] - self.mean
            df[self.continuous] /= self.std
            
            self.train = df
            self.x_data = df.to_numpy()
        else:
            df_train = base.iloc[:40000] # train
            df = base.iloc[40000:] # test
            
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