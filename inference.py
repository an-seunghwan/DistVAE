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
    tags=["Credit", "Inference", "v2"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=10, 
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
            df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True).iloc[:62500]
            continuous = [x for x in df.columns if x != 'Class']
            df = df[continuous]
            self.continuous = continuous
            
            train = df.iloc[:int(len(df) * 0.8)]
            
            # normalization
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
            return x
    
    class TestTabularDataset(Dataset): 
        def __init__(self, config):
            """
            load dataset: Credit
            Reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
            """
            df = pd.read_csv('./data/creditcard.csv')
            df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True).iloc[:62500]
            continuous = [x for x in df.columns if x != 'Class']
            df = df[continuous]
            self.continuous = continuous
            
            train = df.iloc[:int(len(df) * 0.8)]
            test = df.iloc[int(len(df) * 0.8):]
            
            # normalization
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
            return x
    
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
    """3D visualization of quantile function"""
    if not os.path.exists("./assets/latent_quantile"): 
        os.makedirs("./assets/latent_quantile")
            
    xs = torch.linspace(-2, 2, steps=30)
    ys = torch.linspace(-2, 2, steps=30)
    x, y = torch.meshgrid(xs, ys)
    grid_z = torch.cat([x.flatten()[:, None], y.flatten()[:, None]], axis=1)
    
    j = 1
    alpha = 0.5
    for j in range(config["input_dim"]):
        quantiles = []
        for alpha in np.linspace(0.1, 0.9, 9):
            with torch.no_grad():
                gamma, beta = model.quantile_parameter(grid_z)
                quantiles.append(model.quantile_function(alpha, gamma, beta, j))
        
        fig = plt.figure(figsize=(6, 4))
        ax = fig.gca(projection='3d')
        for i in range(len(quantiles)):
            ax.plot_surface(x.numpy(), y.numpy(), quantiles[i].reshape(x.shape).numpy())
            ax.set_xlabel('$z_1$', fontsize=14)
            ax.set_ylabel('$z_2$', fontsize=14)
            ax.set_zlabel('{}'.format(dataset.continuous[j]), fontsize=14)
        ax.view_init(30, 60)
        plt.tight_layout()
        plt.savefig('./assets/latent_quantile/latent_quantile_{}.png'.format(j))
        # plt.show()
        plt.close()
        wandb.log({'latent space ~ quantile': wandb.Image(fig)})
    #%%
    """latent space"""
    latents = []
    for (x_batch) in tqdm.tqdm(iter(dataloader)):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        with torch.no_grad():
            mean, logvar = model.get_posterior(x_batch.tanh())
        latents.append(mean)
    latents = torch.cat(latents, axis=0)
    
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(latents[:, 0], latents[:, 1], 
                alpha=0.7, s=1)
    plt.xlabel('$z_1$', fontsize=14)
    plt.ylabel('$z_2$', fontsize=14)
    plt.tight_layout()
    plt.savefig('./assets/latent.png')
    # plt.show()
    plt.close()
    wandb.log({'latent space': wandb.Image(fig)})
    #%%
    """Empirical quantile plot"""
    q = np.arange(0.01, 0.99, 0.01)
    fig, ax = plt.subplots(5, 6, figsize=(12, 10))
    for k, v in enumerate(dataset.continuous):
        ax.flatten()[k].plot(q, np.quantile(np.tanh(dataset.x_data[:, k]), q=q))
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.tight_layout()
    plt.savefig('./assets/empirical_quantile.png')
    # plt.show()
    plt.close()
    #%%
    """estimated quantile plot"""
    n = 100
    q = np.arange(0.01, 0.99, 0.01)
    j = 0
    randn = torch.randn(n, 2) # prior
    fig, ax = plt.subplots(6, 5, figsize=(10, 12))
    for k, v in enumerate(dataset.continuous):
        quantiles = []
        for alpha in q:
            with torch.no_grad():
                gamma, beta = model.quantile_parameter(randn)
                quantiles.append(model.quantile_function(alpha, gamma, beta, k))
        ax.flatten()[k].plot(q, np.quantile(np.tanh(dataset.x_data[:, k]), q=q), label="empirical")
        ax.flatten()[k].plot(q, [x.mean().item() for x in quantiles], label="prior")
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/prior_estimated_quantile.png')
    # plt.show()
    plt.close()
    wandb.log({'Estimated quantile (prior)': wandb.Image(fig)})
    #%%
    n = 100
    q = np.arange(0.01, 0.99, 0.01)
    j = 0
    idx = np.random.choice(range(len(latents)), n, replace=False)
    fig, ax = plt.subplots(6, 5, figsize=(10, 12))
    for k, v in enumerate(dataset.continuous):
        quantiles = []
        for alpha in q:
            with torch.no_grad():
                gamma, beta = model.quantile_parameter(latents[idx, :]) # aggregated
                quantiles.append(model.quantile_function(alpha, gamma, beta, k))
        ax.flatten()[k].plot(q, np.quantile(np.tanh(dataset.x_data[:, k]), q=q), label="empirical")
        ax.flatten()[k].plot(q, [x.mean().item() for x in quantiles], label="aggregated")
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/aggregated_estimated_quantile.png')
    # plt.show()
    plt.close()
    wandb.log({'Estimated quantile (aggregated)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%