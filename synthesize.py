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
from sklearn import metrics

from modules.simulation import set_random_seed

from modules.model import VAE
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
    tags=["Synthesize"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    # dataset = "adult"
    dataset = "covtype"
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/VAE(CRPS)/model_{}:v{}'.format(dataset, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert dataset == config["dataset"]
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
    TestTabularDataset = dataset_module.TestTabularDataset
    
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = TestTabularDataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    # print(test_dataset.transformer.output_dimensions)
    
    if config["vgmm"]:
        config["input_dim"] = dataset.transformer.output_dimensions
    else:
        config["input_dim"] = len(dataset.continuous)
        # config["input_dim"] = len(dataset.continuous + dataset.discrete)
    # config["output_dim"] = len(dataset.continuous)
    # config["output_dim"] = len(dataset.continuous + dataset.discrete)
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
    import statsmodels.api as sm
    from sklearn.metrics import f1_score
    from sklearn.ensemble import RandomForestClassifier
    #%%
    """baseline"""
    covariates = [x for x in dataset.train.columns if x not in ['Elevation', 'Slope', 'Cover_Type']]
    
    linreg = sm.OLS(dataset.train['Elevation'], dataset.train[covariates]).fit()
    # print(linreg.summary())
    pred = linreg.predict(test_dataset.test[covariates])
    
    rsq_baseline = (test_dataset.test['Elevation'] - pred).pow(2).sum()
    rsq_baseline /= np.var(test_dataset.test['Elevation']) * len(test_dataset.test)
    rsq_baseline = 1 - rsq_baseline
    print("[Baseline] R-squared: {:.3f}".format(rsq_baseline))
    wandb.log({'R^2 (Baseline)': rsq_baseline})
    #%%
    """1. Inverse transform sampling"""
    n = len(dataset.train)
    randn = torch.randn(n, config["latent_dim"]) # prior
    quantiles = []
    for j in range(len(dataset.continuous)):
        alpha = torch.rand(n, 1)
        with torch.no_grad():
            gamma, beta = model.quantile_parameter(randn)
            quantiles.append(model.quantile_function(alpha, gamma, beta, j))
    quantiles = torch.cat(quantiles, dim=1).numpy()
    ITS = pd.DataFrame(quantiles, columns=dataset.continuous)
    #%%
    linreg = sm.OLS(ITS['Elevation'], ITS[covariates]).fit()
    # print(linreg.summary())
    pred = linreg.predict(test_dataset.test[covariates])
    
    rsq = (test_dataset.test['Elevation'] - pred).pow(2).sum()
    rsq /= np.var(test_dataset.test['Elevation']) * len(test_dataset.test)
    rsq = 1 - rsq
    print("[Inverse transform sampling] R-squared: {:.3f}".format(rsq))
    wandb.log({'R^2 (Inverse transform sampling)': rsq})
    #%%
    # """2. Mean"""
    # from scipy.integrate import quad
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%