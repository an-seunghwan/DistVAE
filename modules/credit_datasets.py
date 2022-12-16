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
# https://chocoffee20.tistory.com/6
class TabularDataset(Dataset): 
    def __init__(self, config, random_state=0):
        base = pd.read_csv('./data/application_train.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        
        self.continuous = [
            'AMT_INCOME_TOTAL', 
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE',
            'REGION_POPULATION_RELATIVE', 
            'DAYS_BIRTH', 
            'DAYS_EMPLOYED', 
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        df = df.iloc[:300000]
        
        if config["vgmm"]:
            transformer = DataTransformer()
            transformer.fit(df, random_state=random_state)
            # transformer.fit(df, discrete_columns=self.discrete, random_state=random_state)
            train_data = transformer.transform(df)
            self.transformer = transformer
            self.x_data = train_data
        else:
            df[self.continuous] = (df[self.continuous] - df[self.continuous].mean(axis=0))
            df[self.continuous] /= df[self.continuous].std(axis=0)
            self.train = df
            self.x_data = df.to_numpy()
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%
class TestTabularDataset(Dataset): 
    def __init__(self, config, random_state=0):
        base = pd.read_csv('./data/application_train.csv')
        base = base.sample(frac=1, random_state=0).reset_index(drop=True)
        
        self.continuous = [
            'AMT_INCOME_TOTAL', 
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE',
            'REGION_POPULATION_RELATIVE', 
            'DAYS_BIRTH', 
            'DAYS_EMPLOYED', 
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        df_ = df.iloc[:300000]
        df = df.iloc[300000:]
        
        if config["vgmm"]:
            transformer = DataTransformer()
            transformer.fit(df_, random_state=random_state)
            # transformer.fit(df, discrete_columns=self.discrete, random_state=random_state)
            train_data = transformer.transform(df)
            self.transformer = transformer
            self.x_data = train_data
        else:
            df[self.continuous] = (df[self.continuous] - df_[self.continuous].mean(axis=0))
            df[self.continuous] /= df_[self.continuous].std(axis=0)
            self.test = df
            self.x_data = df.to_numpy()
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%