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
    tags=['DistVAE', 'Calibration'],
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
    
    dataset = TabularDataset()
    test_dataset = TabularDataset(train=False)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    
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
    """real data: covtype dataset"""
    base = pd.read_csv('./data/covtype.csv')
    base = base.sample(frac=1, random_state=0).reset_index(drop=True)
    base = base.dropna(axis=0)
    base = base.iloc[:50000]
    
    continuous = [
        'Elevation', # target variable
        'Aspect', 
        'Slope',
        'Horizontal_Distance_To_Hydrology', 
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
    ]
    base = base[continuous]
    
    df = base.iloc[:45000] # train
    #%%
    MC = 5000
    
    """Quantile Estimation with sampling mechanism"""
    n = 100
    x_linspace_est = np.linspace(
        np.quantile(dataset.x_data[:, j], q=0.01),
        np.quantile(dataset.x_data[:, j], q=0.99),
        n)
    
    # for comparison
    j = 2 # Slope
    alpha_est = torch.zeros((len(x_linspace_est), 1))
    for _ in tqdm.tqdm(range(MC), desc="Estimate CDF..."):
        randn = torch.randn(len(x_linspace_est), config["latent_dim"]) # prior
        with torch.no_grad():
            gamma, beta, _ = model.quantile_parameter(randn)
            x_tmp = torch.from_numpy(x_linspace_est[:, None]).clone()
            alpha_tilde = model._quantile_inverse(x_tmp, gamma, beta, j)
            alpha_est += alpha_tilde
    alpha_est /= MC
    
    x_linspace_est = x_linspace_est * dataset.std[j] + dataset.mean[j]
    #%%
    """Calibration Step 1. Estimate F(x + 0.5), F(x - 0.5)"""
    x_linspace = [np.arange(x-0.5, y+0.5, 1) for x, y in zip(
        [np.quantile(df.to_numpy()[:, j], q=0.01)],
        [np.quantile(df.to_numpy()[:, j], q=0.99)])][0]
    
    alpha_hat = torch.zeros((len(x_linspace), 1))
    for _ in tqdm.tqdm(range(MC), desc="Estimate CDF..."):
        randn = torch.randn(len(x_linspace), config["latent_dim"]) # prior
        with torch.no_grad():
            gamma, beta, _ = model.quantile_parameter(randn)
            x_tmp = torch.from_numpy(x_linspace[:, None]).clone()
            x_tmp -= dataset.mean.to_numpy()[j]
            x_tmp /= dataset.std.to_numpy()[j]
            alpha_tilde = model._quantile_inverse(x_tmp, gamma, beta, j)
            alpha_hat += alpha_tilde
    alpha_hat /= MC
    
    x_linspace = [np.arange(x, y, 1) for x, y in zip(
        [np.quantile(df.to_numpy()[:, j], q=0.01)],
        [np.quantile(df.to_numpy()[:, j], q=0.99)])][0]
    #%%
    """Calibration Step 2. Quantization F(x) = F(x + 0.5) - F(x - 0.5)"""
    alpha_cal = []
    for i in range(len(alpha_hat)-1):
        alpha_cal.append((alpha_hat[i+1] - alpha_hat[i]).item())
    alpha_cal = np.cumsum(alpha_cal)
    #%%
    """Calibration Step 3. Ensure monotonicity"""
    alpha_mono = [0]
    for i in range(1, len(alpha_cal)):
        if alpha_cal[i] < alpha_mono[-1]:
            alpha_mono.append(alpha_mono[-1])
        else:
            alpha_mono.append(alpha_cal[i])
    #%%
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.plot(x_linspace_est, alpha_est, label="estimate")
    ax.step(x_linspace, alpha_mono, label="calibration")
    emp = [stats.percentileofscore(
        df[dataset.continuous].to_numpy()[:, j],
        x) * 0.01 for x in x_linspace]
    ax.step(x_linspace, emp, label="empirical")
    ax.set_xlabel(dataset.continuous[j], fontsize=14)
    ax.set_ylabel('alpha', fontsize=14)
    
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_CDF_calibration.png'.format(config["dataset"], config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'CDF calibration': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%