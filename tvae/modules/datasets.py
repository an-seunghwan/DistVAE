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

from .data_transformer import DataTransformer
#%%
def generate_dataset(config, device, random_state=0):
    
    if config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        
        continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            # 'Slope', 
            # 'Cover_Type'
            ]
        df = df[continuous]
        df = df.dropna(axis=0)
        df = df.iloc[2000:]
        
        transformer = DataTransformer()
        transformer.fit(df, random_state=random_state)
        # transformer.fit(df, discrete_columns=['Cover_Type'], random_state=random_state)
        train_data = transformer.transform(df)
    
    elif config["dataset"] == 'credit':
        df = pd.read_csv('./data/application_train.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        continuous = [
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
        df = df[continuous]
        df = df.dropna(axis=0)
        df = df.iloc[:300000]
        
        transformer = DataTransformer()
        transformer.fit(df.iloc[:30000], random_state=random_state)
        train_data = transformer.transform(df)
        
    else:
        raise ValueError('Not supported dataset!')    

    dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(device))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    
    return dataset, dataloader, transformer
#%%