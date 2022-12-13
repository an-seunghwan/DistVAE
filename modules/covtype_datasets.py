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
def interleave_float(a: float, b: float):
    a_rest = a
    b_rest = b
    result = 0
    dst_pos = 1.0  # position of written digit
    while a_rest != 0 or b_rest != 0:
        dst_pos /= 10  # move decimal point of write
        a_rest *= 10  # move decimal point of read
        result += dst_pos * (a_rest // 1)
        a_rest %= 1  # remove current digit

        dst_pos /= 10
        b_rest *= 10
        result += dst_pos * (b_rest // 1)
        b_rest %= 1
    return result
#%%
class TabularDataset(Dataset): 
    def __init__(self, config):
        # if config["dataset"] == 'covtype':
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
        self.topology = [
            ['Horizontal_Distance_To_Hydrology'], 
            ['Vertical_Distance_To_Hydrology'],
            ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'],
            ['Elevation'], 
            ['Aspect'], 
            ['Slope', 'Cover_Type'],
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        scaling = [x for x in self.continuous if x != 'Cover_Type']
        df_ = df.copy()
        df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
        train = df_.iloc[2000:, ]
        
        min_ = df_.min(axis=0)
        max_ = df_.max(axis=0)
        df = (df_ - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 2:
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[2000:, ]
        
        self.train = train
        self.x_data = train.to_numpy()
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%
class TestTabularDataset(Dataset): 
    def __init__(self, config):
        # if config["dataset"] == 'covtype':
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
        self.topology = [
            ['Horizontal_Distance_To_Hydrology'], 
            ['Vertical_Distance_To_Hydrology'],
            ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'],
            ['Elevation'], 
            ['Aspect'], 
            ['Slope', 'Cover_Type'],
        ]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        scaling = [x for x in self.continuous if x != 'Cover_Type']
        df_ = df.copy()
        df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
        test = df_.iloc[:2000, ]
        
        min_ = df_.min(axis=0)
        max_ = df_.max(axis=0)
        df = (df_ - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 2:
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[:2000, ]
        
        self.test = test
        self.x_data = test.to_numpy()
        
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%
class TabularDataset2(Dataset): 
    def __init__(self, config, random_state=0):
        # if config["dataset"] == 'covtype':
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
        self.topology = [
            ['Horizontal_Distance_To_Hydrology'], 
            ['Vertical_Distance_To_Hydrology'],
            ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'],
            ['Elevation'], 
            ['Aspect'], 
            ['Slope', 'Cover_Type'],
        ]
        self.flatten_topology = [item for sublist in self.topology for item in sublist]
        df = base[self.continuous]
        df = df.dropna(axis=0)
        
        min_ = df.min(axis=0)
        max_ = df.max(axis=0)
        df_normalized = (df - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df_normalized[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 2:
                df_tmp = df_normalized[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[2000:, ]
        
        df = df[self.flatten_topology].iloc[2000:]
        
        transformer = DataTransformer()
        transformer.fit(df, discrete_columns=['Cover_Type'], random_state=random_state)
        train_data = transformer.transform(df)
        self.transformer = transformer
            
        self.train = train_data
        self.x_data = torch.from_numpy(train_data.astype('float32'))
            
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y
#%%