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

from statsmodels.distributions.empirical_distribution import ECDF
from scipy import interpolate
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
    parser.add_argument('--beta', default=0.5, type=float,
                        help='observation noise')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{}_DistVAE_{}:v{}'.format(
        config["beta"], config["dataset"], config["num"]), type='model')
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
    """alpha-rate"""
    alpha_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    alpha_rate = []
    for j in range(len(dataset.continuous)):
        tmp = []
        for alpha in alpha_levels:
            if len(np.where(alpha_hat[:, j] < alpha)[0]):
                cut1 = np.where(alpha_hat[:, j] < alpha)[0][-1]
            else:
                cut1 = 0
            if len(np.where(alpha < alpha_hat[:, j])[0]):
                cut2 = np.where(alpha < alpha_hat[:, j])[0][0]
            else:
                cut2 = -1
            
            f_inter = interpolate.interp1d(
                [alpha_hat[cut1, j], alpha_hat[cut2, j]],
                [x_linspace[:, j][cut1], x_linspace[:, j][cut2]])
            try:
                tmp.append((test_dataset.x_data[:, j] <= f_inter(alpha)).mean())
            except:
                tmp.append((test_dataset.x_data[:, j] <= x_linspace[:, j][cut2].item()).mean())
        alpha_rate.append(tmp)
    alpha_rate = np.array(alpha_rate).mean(axis=0)
    #%%
    pd.DataFrame(
        np.concatenate([
            alpha_rate[None, :],
            np.abs(alpha_rate - alpha_levels)[None, :]
        ], axis=0).round(3),
        columns=[str(x) for x in alpha_levels]
    ).to_csv('./assets/{}/{}_alpha_rate.csv'.format(config["dataset"], config["dataset"]))
    #%%
    if config["dataset"] == "covtype":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2, 
                               figsize=(3 * config["CRPS_dim"] // 2, 3 * 2))
    elif config["dataset"] == "credit":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2, 
                               figsize=(3 * config["CRPS_dim"] // 2, 3 * 2))
    elif config["dataset"] == "loan":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
    elif config["dataset"] == "adult":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
    elif config["dataset"] == "cabs":
        fig, ax = plt.subplots(1, config["CRPS_dim"], 
                               figsize=(3 * config["CRPS_dim"], 3 * 1))
    elif config["dataset"] == "kings":
        fig, ax = plt.subplots(2, config["CRPS_dim"] // 2 + 1, 
                               figsize=(3 * config["CRPS_dim"] // 2 + 1, 3 * 2))
    else:
        raise ValueError('Not supported dataset!')
    
    orig = dataset.x_data[:, :len(dataset.continuous)] * np.array(dataset.std)
    orig += np.array(dataset.mean)
    orig = pd.DataFrame(orig, columns=dataset.continuous).astype(int)
    
    for k, v in enumerate(dataset.continuous):
        x_linspace_orig = [np.arange(x, y, 1) for x, y in zip(
            [np.quantile(orig.to_numpy()[:, k], q=0.01)],
            [np.quantile(orig.to_numpy()[:, k], q=0.99)])][0]
        if v in dataset.integer:
            ecdf = ECDF(orig[dataset.continuous].to_numpy()[:, k])
            emp = [ecdf(x) for x in x_linspace_orig]
            ax.flatten()[k].step(
                (x_linspace_orig - dataset.mean[k]) / dataset.std[k], 
                emp, where='post',
                label="empirical", linewidth=3.5, color=u'#ff7f0e')
        else:
            q = np.arange(0.01, 1, 0.01)
            ax.flatten()[k].step(
                np.quantile(dataset.x_data[:, k], q=q), 
                q, where='post',
                label="empirical", linewidth=3.5, color=u'#ff7f0e')
        
        ax.flatten()[k].plot(
            x_linspace[:, k], alpha_hat[:, k], 
            label="estimate", linewidth=3.5, linestyle='dashed', color=u'#1f77b4')    
        
        ax.flatten()[k].set_xlabel(v, fontsize=12)
        # ax.flatten()[k].set_ylabel('CDF', fontsize=12)
        ax.flatten()[k].tick_params(axis="x", labelsize=14)
        ax.flatten()[k].tick_params(axis="y", labelsize=14)
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