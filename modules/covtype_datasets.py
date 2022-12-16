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
class TabularDataset(Dataset): 
    def __init__(self, config, random_state=0):
        base = pd.read_csv('./data/covtype.csv')
        base = base.sample(frac=1, random_state=5).reset_index(drop=True)
        
        self.continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            'Slope', 
            'Cover_Type',
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        df = df.iloc[2000:]
        
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
        base = pd.read_csv('./data/covtype.csv')
        base = base.sample(frac=1, random_state=5).reset_index(drop=True)
        
        self.continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            'Slope', 
            'Cover_Type',
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        df_ = df.iloc[2000:]
        df = df.iloc[:2000]
        
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