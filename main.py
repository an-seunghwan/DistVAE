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

from modules.model import (
    VAE
)
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
def train_VAE(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'quantile': [],
        'KL': [],
    }
    # for debugging
    for i in range(config["latent_dim"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, _) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            # y_batch = y_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        z, mean, logvar, gamma, beta = model(x_batch)
        
        loss_ = []
        
        """alpha_tilde"""
        j = 0
        alpha_tilde_list = []
        for j in range(config["input_dim"]):
            mask = [model.quantile_function(d, gamma, beta, j) for d in model.delta[0]]
            mask = torch.cat(mask, axis=1)
            mask = torch.where(mask <= x_batch[:, [j]], 
                            mask, 
                            torch.zeros(())).type(torch.bool).type(torch.float)
            alpha_tilde = x_batch[:, [j]] - gamma[j]
            alpha_tilde += (mask * beta[j] * model.delta).sum(axis=1, keepdims=True)
            alpha_tilde /= (mask * beta[j]).sum(axis=1, keepdims=True) + 1e-6
            alpha_tilde = torch.clip(alpha_tilde, 1e-4, 1) # numerical stability
            alpha_tilde_list.append(alpha_tilde)
        
        """loss"""
        j = 0
        total_loss = 0
        for j in range(config["input_dim"]):
            term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
            term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
            
            loss = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
            loss += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
            loss += (beta[j] * term).sum(axis=1, keepdims=True)
            loss *= 0.5
            total_loss += loss.mean()
        # print(loss.mean())
        loss_.append(('quantile', total_loss))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["latent_dim"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
        loss = total_loss + config["beta"] * KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
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
    class TabularDataset(Dataset): 
        def __init__(self, config):
            df = pd.read_csv('./data/creditcard.csv')
            df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
            continuous = [x for x in df.columns if x != 'Class']
            self.y_data = df["Class"].to_numpy()[:, None]
            df = df[continuous]
            self.continuous = continuous
            
            train = df.iloc[:int(len(df) * 0.8)]
            # test = df.iloc[int(len(df) * 0.8):]
            
            # scaling
            mean = train.mean(axis=0)
            std = train.std(axis=0)
            self.mean = mean
            self.std = std
            train = (train - mean) / std
            # test = (test - mean) / std
            self.train = train
            self.x_data = train.to_numpy()

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
    """Empirical quantile plot"""
    q = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots(5, 6, figsize=(12, 10))
    for k, v in enumerate(dataset.continuous):
        ax.flatten()[k].plot(q, np.quantile(dataset.train[v], q=q))
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.tight_layout()
    plt.savefig('./assets/empirical_quantile.png')
    plt.show()
    plt.close()
    #%%
    model = VAE(config, device).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    #%%
    model.train()
    
    for epoch in range(config["epochs"]):
        logs = train_VAE(dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        # """update log"""
        # wandb.log({x : np.mean(y) for x, y in logs.items()})
            
        # if epoch % 10 == 0:
        #     plt.figure(figsize=(4, 4))
        #     for i in range(9):
        #         plt.subplot(3, 3, i+1)
        #         plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        #         plt.axis('off')
        #     plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
        #     plt.close()
    
    # """reconstruction result"""
    # fig = plt.figure(figsize=(4, 4))
    # for i in range(9):
    #     plt.subplot(3, 3, i+1)
    #     plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
    #     plt.axis('off')
    # plt.savefig('./assets/recon.png')
    # plt.close()
    # wandb.log({'reconstruction': wandb.Image(fig)})
    #%%
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