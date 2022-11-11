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
    tags=["Credit", "Inference"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=1, 
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
    # for (x_batch, _) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
    #     if config["cuda"]:
    #         x_batch = x_batch.cuda()
    #         # y_batch = y_batch.cuda()
        
    #     with torch.no_grad():
    #         z, mean, logvar, gamma, beta = model(x_batch, deterministic=True)
    #%%    
    xs = torch.linspace(-2, 2, steps=10)
    ys = torch.linspace(-2, 2, steps=10)
    x, y = torch.meshgrid(xs, ys)
    grid_z = torch.cat([x.flatten()[:, None], y.flatten()[:, None]], axis=1)
    
    j = 1
    alpha = 0.5
    quantiles = []
    for alpha in np.linspace(0.1, 0.9, 9):
        with torch.no_grad():
            gamma, beta = model.quantile_parameter(grid_z)
            quantiles.append(model.quantile_function(alpha, gamma, beta, j))
    #%%
    ax = plt.axes(projection='3d')
    for i in range(len(quantiles)):
        ax.plot_surface(x.numpy(), y.numpy(), quantiles[i].reshape(x.shape).numpy())
    plt.savefig('./assets/latent_quantile.png')
    plt.show()
    plt.close()
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%