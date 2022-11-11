#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

# from modules.model import (
    
# )
#%%
# import sys
# import subprocess
# try:
#     import wandb
# except:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
#     with open("./wandb_api.txt", "r") as f:
#         key = f.readlines()
#     subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
#     import wandb

# run = wandb.init(
#     project="VAE(CRPS)", 
#     entity="anseunghwan",
#     # tags=[""],
# )
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument("--latent_dim", default=2, type=int,
                        help="the number of latent codes")
    
    # optimization options
    parser.add_argument('--epochs', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    # loss coefficients
    parser.add_argument('--beta', default=0.1, type=float,
                        help='observation noise')
  
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """
    load dataset: Credit
    Reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
    """
    df = pd.read_csv('./data/creditcard.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    continuous = [x for x in df.columns if x != 'Class']
    df = df[continuous]
    
    config["input_dim"] = len(continuous)
    
    train = df.iloc[:int(len(df) * 0.8)]
    test = df.iloc[int(len(df) * 0.8):]
    # scaling
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    train = (train - mean) / std
    test = (test - mean) / std
    #%%
    """Empirical quantile plot"""
    q = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots(5, 6, figsize=(12, 10))
    for k, v in enumerate(continuous):
        ax.flatten()[k].plot(q, np.quantile(train[v], q=q))
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.tight_layout()
    plt.savefig('./assets/empirical_quantile.png')
    plt.show()
    plt.close()
    #%%
    """model"""
    """encoder"""
    encoder = nn.Sequential(
        nn.Linear(config["input_dim"], 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, config["latent_dim"] * 2),
    ).to(device)
    
    M = 10
    spline = nn.Sequential(
        nn.Linear(config["latent_dim"], 4),
        nn.ReLU(),
        nn.Linear(4, config["input_dim"] * (1 + (M + 1))),
    ).to(device)
    #%%
    batch = torch.randn(10, config["input_dim"])
    h = encoder(batch)
    mean, logvar = torch.split(h, config["latent_dim"], dim=1)
    
    deterministic = False
    if deterministic:
        z = mean
    else:
        noise = torch.randn(batch.size(0), config["latent_dim"]).to(device) 
        z = mean + torch.exp(logvar / 2) * noise
    
    delta = torch.arange(0, 1.1, step=0.1)
    h = spline(z)
    h = torch.split(h, 1 + (M + 1), dim=1)
    
    gamma = [h_[:, [0]] for h_ in h]
    beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
    
    alpha = 0.5
    quant_reg = [g + (b * torch.where(alpha - delta > 0,
                                    alpha - delta,
                                    torch.zeros(()))).sum(axis=1, keepdims=True) for g, b in zip(gamma, beta)]
    
    #%%
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    model.train()
    
    for epoch in range(config["epochs"]):
        if config["model"] == 'VAE':
            logs, xhat = train_VAE(dataloader, model, config, optimizer, device)
        elif config["model"] == 'InfoMax':
            logs, xhat = train_InfoMax(dataloader, model, discriminator, config, optimizer, optimizer_D, device)
        elif config["model"] == 'GAM':
            logs, xhat = train_GAM(dataloader, model, config, optimizer, device)
        else:
            raise ValueError('Not supported model!')
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
            
        if epoch % 10 == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
                plt.axis('off')
            plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
            plt.close()
    
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/recon.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/model_{}_{}.pth'.format(config["model"], config["scm"]))
    artifact = wandb.Artifact('model_{}_{}'.format(config["model"], config["scm"]), 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file('./assets/model_{}_{}.pth'.format(config["model"], config["scm"]))
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%