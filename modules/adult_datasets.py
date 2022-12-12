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
"""
load dataset: Adult
Reference: https://archive.ics.uci.edu/ml/datasets/Adult
"""
class TabularDataset(Dataset): 
    def __init__(self, config):
        # if config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
        df = df[self.continuous]
        
        scaling = [x for x in self.continuous if x != 'income']
        df_ = df.copy()
        df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
        # df_ = (df - df.mean(axis=0)) / df.std(axis=0)
        train = df_.iloc[:40000, ]
        
        min_ = df_.min(axis=0)
        max_ = df_.max(axis=0)
        df = (df_ - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 3:
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp[:, :2]:
                    bijection_tmp.append(interleave_float(x, y))
                tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                bijection_tmp = []
                for x, y in tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[:40000, ]
        
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
        # if config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
        self.flatten_topology =  [self.continuous.index(item) for sublist in self.topology for item in sublist]
        df = df[self.continuous]
        
        scaling = [x for x in self.continuous if x != 'income']
        df_ = df.copy()
        df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
        # df_ = (df - df.mean(axis=0)) / df.std(axis=0)
        test = df_.iloc[40000:, ]
        
        min_ = df_.min(axis=0)
        max_ = df_.max(axis=0)
        df = (df_ - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 3:
                df_tmp = df[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp[:, :2]:
                    bijection_tmp.append(interleave_float(x, y))
                tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                bijection_tmp = []
                for x, y in tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[40000:, ]
        
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
        # if config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        self.continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.topology = [['capital-gain'], ['capital-loss'], ['income', 'educational-num', 'hours-per-week']]
        self.flatten_topology = [item for sublist in self.topology for item in sublist]
        # self.flatten_topology = [self.continuous.index(item) for sublist in self.topology for item in sublist]
        df = df[self.continuous]
        
        min_ = df.min(axis=0)
        max_ = df.max(axis=0)
        df_normalized = (df - min_) / (max_ - min_) 
        
        bijection = []
        for i in range(len(self.topology)):
            if len(self.topology[i]) == 1:
                bijection.append(df_normalized[self.topology[i]].to_numpy())
                continue
            if len(self.topology[i]) == 3:
                df_tmp = df_normalized[self.topology[i]].to_numpy()
                bijection_tmp = []
                for x, y in df_tmp[:, :2]:
                    bijection_tmp.append(interleave_float(x, y))
                tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
                bijection_tmp = []
                for x, y in tmp:
                    bijection_tmp.append(interleave_float(x, y))
                bijection.append(np.array([bijection_tmp]).T)
        bijection = np.concatenate(bijection, axis=1)
        self.label = bijection[:40000, ]
        
        df = df[self.flatten_topology].iloc[:4000]
        
        transformer = DataTransformer()
        transformer.fit(df, discrete_columns=['income'], random_state=random_state)
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