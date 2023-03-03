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
# import matplotlib as mpl
# mpl.style.use('seaborn')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.simulation import set_random_seed
from modules.model import VAE

from statsmodels.distributions.empirical_distribution import ECDF
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
    project="DistVAE", 
    entity="anseunghwan",
    tags=['DistVAE', 'Complexity'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: supports only covtype!')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/beta0.5_DistVAE_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
    #     config["dataset"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    if not os.path.exists('./assets/{}'.format(config["dataset"])):
        os.makedirs('./assets/{}'.format(config["dataset"]))
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    import importlib
    dataset_module = importlib.import_module('modules.{}_datasets'.format(config["dataset"]))
    TabularDataset = dataset_module.TabularDataset
    
    dataset = TabularDataset()
    test_dataset = TabularDataset(train=False)
    
    OutputInfo_list = dataset.OutputInfo_list
    CRPS_dim = sum([x.dim for x in OutputInfo_list if x.activation_fn == 'CRPS'])
    softmax_dim = sum([x.dim for x in OutputInfo_list if x.activation_fn == 'softmax'])
    config["CRPS_dim"] = CRPS_dim
    config["softmax_dim"] = softmax_dim
    #%%
    model = VAE(config, device).to(device)
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """Complexity of CDF"""
    MC = 100
    j = 4 # Vertical_Distance_To_Hydrology
    
    n = 100
    x_linspace_est = np.linspace(
        np.min(dataset.x_data[:, j]),
        np.max(dataset.x_data[:, j]),
        n)
    x_linspace_est = torch.from_numpy(x_linspace_est[:, None]).clone()
    
    alpha_MC = torch.zeros(n, 1)
    alpha_conditional = []
    for _ in range(MC):
        idx = np.random.choice(
            range(len(dataset.x_data)), 1, replace=False)
        x_batch = torch.FloatTensor(dataset.x_data[idx, :]).to(device)
        with torch.no_grad():
            z, mean, logvar, gamma, beta, _ = model(x_batch) # posterior
            a = model._quantile_inverse(x_linspace_est, gamma, beta, j)
        alpha_conditional.append(a)
        alpha_MC += a
    alpha_MC /= MC
    #%%
    x_linspace_est = np.linspace(
        np.min(dataset.x_data[:, j]),
        np.max(dataset.x_data[:, j]),
        n)
    x_linspace_est = x_linspace_est * dataset.std[j] + dataset.mean[j]
    
    m = 10
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for a in alpha_conditional[-m:]:
        ax[0].plot(
            x_linspace_est, a, 
            color='black', linestyle='--', linewidth=1)
    ax[0].plot(
        x_linspace_est, alpha_MC,
        color='blue', linewidth=3)
    ax[0].set_xlabel(dataset.continuous[j])
    for a in alpha_conditional[-m:]:
        ax[1].plot(
            x_linspace_est, a, 
            color='black', linestyle='--', linewidth=1)
    ax[1].plot(
        x_linspace_est, alpha_MC,
        color='blue', linewidth=3)
    ax[1].set_xlim(0, 150)
    ax[1].set_xlabel(dataset.continuous[j])
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_CDF_complexity.png'.format(config["dataset"], config["dataset"]))
    plt.show()
    plt.close()
    wandb.log({'CDF complexity': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%