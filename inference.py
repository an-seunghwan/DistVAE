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

from modules.simulation import set_random_seed
from modules.model import VAE

from scipy import stats
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
    tags=['DistVAE', 'Inference'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(config["dataset"], config["num"]), type='model')
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
    
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
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
    """Quantile Estimation with sampling mechanism"""
    n = 100
    MC = 5000
    x_linspace = np.linspace(
        [np.quantile(dataset.x_data[:, k], q=0.01) for k in range(len(dataset.continuous))],
        [np.quantile(dataset.x_data[:, k], q=0.99) for k in range(len(dataset.continuous))],
        n)
    x_linspace = torch.from_numpy(x_linspace)
    
    alpha_hat = torch.zeros((n, len(dataset.continuous)))
    for _ in tqdm.tqdm(range(MC), desc="Estimate CDF..."):
        randn = torch.randn(n, config["latent_dim"]) # prior
        with torch.no_grad():
            gamma, beta, _ = model.quantile_parameter(randn)
            alpha_tilde_list = model.quantile_inverse(x_linspace, gamma, beta)
            alpha_hat += torch.cat(alpha_tilde_list, dim=1)
    alpha_hat /= MC
    #%%
    if config["dataset"] == "covtype":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2, 
                               figsize=(3 * config["CRPS_dim"] // 2, 3 * 2))
        integer = dataset.continuous
        
    elif config["dataset"] == "credit":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2, 
                               figsize=(3 * config["CRPS_dim"] // 2, 3 * 2))
        integer = [
            'DAYS_BIRTH', 
            'DAYS_EMPLOYED', 
            'DAYS_ID_PUBLISH']
        
    elif config["dataset"] == "loan":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
        integer = [
            'Age',
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage']
        
    elif config["dataset"] == "adult":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
        integer = dataset.continuous
        
    elif config["dataset"] == "cabs":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
        integer = [
            'Var2',
            'Var3']
        
    elif config["dataset"] == "kings":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2 + 1, 
                               figsize=(3 * config["CRPS_dim"] // 2 + 1, 3 * 2))
        integer = [
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'sqft_living15',
            'sqft_lot15',]
        
    else:
        raise ValueError('Not supported dataset!')
    
    orig = dataset.x_data[:, :len(dataset.continuous)] * np.array(dataset.std)
    orig += np.array(dataset.mean)
    orig = pd.DataFrame(orig, columns=dataset.continuous).astype(int)
    
    for k, v in enumerate(dataset.continuous):
        ax.flatten()[k].plot(x_linspace[:, k], alpha_hat[:, k], label="sampled")
        
        x_linspace_orig = [np.arange(x, y, 1) for x, y in zip(
            [np.quantile(orig.to_numpy()[:, k], q=0.01)],
            [np.quantile(orig.to_numpy()[:, k], q=0.99)])][0]
        if v in integer:
            emp = [stats.percentileofscore(
                orig.to_numpy()[:, dataset.continuous.index(v)],
                x.item()) * 0.01 for x in x_linspace_orig]
            ax.flatten()[k].step((x_linspace_orig - dataset.mean[k]) / dataset.std[k], 
                                 emp, 
                                 label="empirical")
        else:
            q = np.arange(0.01, 1, 0.01)
            ax.flatten()[k].step(np.quantile(dataset.x_data[:, k], q=q), q, label="empirical")
        ax.flatten()[k].set_xlabel(v)
        ax.flatten()[k].set_ylabel('alpha')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_estimated_quantile.png'.format(config["dataset"], config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'Estimated quantile (sampling mechanism)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%