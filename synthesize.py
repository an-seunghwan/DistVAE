#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from sklearn.ensemble import RandomForestRegressor

from modules.simulation import (
    set_random_seed
)

from modules.model import (
    VAE
)
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="VAE(CRPS)", 
    entity="anseunghwan",
    tags=["Credit", "Synthesize"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=3, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/VAE(CRPS)/model_credit:v{}'.format(config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    class TabularDataset(Dataset): 
        def __init__(self, config):
            """
            load dataset: Credit
            Reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
            """
            df = pd.read_csv('./data/creditcard.csv')
            df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
            continuous = [x for x in df.columns if x != 'Class']
            self.y_data = df["Class"].to_numpy()[:int(len(df) * 0.8), None]
            df = df[continuous]
            self.continuous = continuous
            
            train = df.iloc[:int(len(df) * 0.8)]
            
            # scaling
            mean = train.mean(axis=0)
            std = train.std(axis=0)
            self.mean = mean
            self.std = std
            train = (train - mean) / std
            self.train = train
            self.x_data = train.to_numpy()

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    class TestTabularDataset(Dataset): 
        def __init__(self, config):
            """
            load dataset: Credit
            Reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
            """
            df = pd.read_csv('./data/creditcard.csv')
            df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
            continuous = [x for x in df.columns if x != 'Class']
            self.y_data = df["Class"].to_numpy()[int(len(df) * 0.8):, None]
            df = df[continuous]
            self.continuous = continuous
            
            train = df.iloc[:int(len(df) * 0.8)]
            test = df.iloc[int(len(df) * 0.8):]
            
            # scaling
            mean = train.mean(axis=0)
            std = train.std(axis=0)
            self.mean = mean
            self.std = std
            test = (test - mean) / std
            self.test = test
            self.x_data = test.to_numpy()

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y
    
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = TestTabularDataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    config["input_dim"] = len(dataset.continuous)
    #%%
    model = VAE(config, device).to(device)
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model_credit.pth'))
    else:
        model.load_state_dict(torch.load(model_dir + '/model_credit.pth', 
                                         map_location=torch.device('cpu')))
    
    model.eval()
    #%%    
    """baseline"""
    n = 1000
    baseline_rsquare = []
    for s in tqdm.tqdm(range(10), desc="Original the dataset and its performance"):
        baseline_rf = RandomForestRegressor(
            max_features='sqrt',
            random_state=0
        )
        np.random.seed(s)
        idx = list(np.random.choice(range(len(dataset.train)), n, replace=False))
        baseline_rf.fit(
            dataset.x_data[idx, :len(dataset.continuous)-1], 
            dataset.train[dataset.continuous[-1]].iloc[idx])
        baseline_rsquare.append(
            baseline_rf.score(
                test_dataset.x_data[:, :len(dataset.continuous)-1], 
                test_dataset.test[dataset.continuous[-1]]))
    print("[Baseline Rsquare], mean: {:.2f}, std: {:.2f}".format(np.mean(baseline_rsquare), np.std(baseline_rsquare)))
    #%%
    """synthesize the dataset"""
    n = 1000
    rsquare = []
    for s in tqdm.tqdm(range(10), desc="Synthesize the dataset and its performance"):
        torch.manual_seed(s)
        randn = torch.randn(n, 2) # prior
        alpha = torch.rand(n, 1)
        quantiles = []
        for k, v in enumerate(dataset.continuous):
            with torch.no_grad():
                gamma, beta = model.quantile_parameter(randn)
                quantiles.append(model.quantile_function(alpha, gamma, beta, k))
        x_train_syn = torch.cat(quantiles[:len(dataset.continuous)-1], axis=1).numpy()
        y_train_syn = quantiles[len(dataset.continuous)-1].numpy()[:, 0]
        
        rf = RandomForestRegressor(
            max_features='sqrt',
            random_state=0
        )
        rf.fit(x_train_syn, y_train_syn)
        rsquare.append(
            rf.score(
                test_dataset.x_data[:, :len(dataset.continuous)-1], 
                test_dataset.test[dataset.continuous[-1]]))
    print("[Synthesized Rsquare], mean: {:.2f}, std: {:.2f}".format(np.mean(rsquare), np.std(rsquare)))
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%