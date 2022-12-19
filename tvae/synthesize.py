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

from modules.model import TVAE

from modules.datasets import generate_dataset

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="VAE(CRPS)", 
    entity="anseunghwan",
    tags=["TVAE", "Synthesize"],
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
    
    dataset = 'covtype'
    # dataset = 'credit'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/VAE(CRPS)/TVAE_{}:v{}'.format(dataset, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert dataset == config["dataset"]
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """model"""
    model = TVAE(config, device).to(device)
    
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
    """dataset"""
    dataset, dataloader, transformer = generate_dataset(config, device, random_state=0)
    
    config["input_dim"] = transformer.output_dimensions
    #%%
    if not os.path.exists('./assets/{}'.format(config["dataset"])):
        os.makedirs('./assets/{}'.format(config["dataset"]))
        
    if config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            # 'Slope', 
            # 'Cover_Type'
        ]
        df = df[continuous]
        df = df.dropna(axis=0)
        
        train = df.iloc[2000:, ]
        test = df.iloc[:2000, ]
        
    elif config["dataset"] == 'credit':
        df = pd.read_csv('./data/application_train.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        continuous = [
            'AMT_INCOME_TOTAL', 
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE',
            'REGION_POPULATION_RELATIVE', 
            'DAYS_BIRTH', 
            'DAYS_EMPLOYED', 
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
        ]
        df = df[continuous]
        df = df.dropna(axis=0)
        
        train = df.iloc[:300000]
        test = df.iloc[300000:]
        
    else:
        raise ValueError('Not supported dataset!')
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    steps = len(train) // config["batch_size"] + 1
    data = []
    with torch.no_grad():
        for _ in range(steps):
            mean = torch.zeros(config["batch_size"], config["latent_dim"])
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(device)
            fake = model.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:len(train)]
    sample_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    #%%
    """Machine Learning Efficacy"""
    if config["dataset"] == "covtype":
        target = 'Elevation'
    elif config["dataset"] == "credit":
        target = 'AMT_INCOME_TOTAL'
    else:
        raise ValueError('Not supported dataset!')
    covariates = [x for x in train.columns if x not in [target]]
    #%%
    # Baseline
    std = train.std(axis=0)
    mean = train.mean(axis=0)
    train = (train - mean) / std
    test = (test - mean) / std
    
    if config["dataset"] == 'covtype':
        regr = RandomForestRegressor(random_state=0)
        regr.fit(train[covariates], train[target])
        pred = regr.predict(test[covariates])
    
    elif config["dataset"] == "credit":
        linreg = sm.OLS(train[target], train[covariates]).fit()
        # print(linreg.summary())
        pred = linreg.predict(test[covariates])
            
    else:
        raise ValueError('Not supported dataset!')
    
    rsq_baseline = (test[target] - pred).pow(2).sum()
    rsq_baseline /= np.var(test[target]) * len(test)
    rsq_baseline = 1 - rsq_baseline
    print("[Baseline] R-squared: {:.3f}".format(rsq_baseline))
    wandb.log({'R^2 (Baseline)': rsq_baseline})
    #%%
    # synthetic
    sample_df = (sample_df - sample_df.mean(axis=0)) / sample_df.std(axis=0)
    
    if config["dataset"] == 'covtype':
        regr = RandomForestRegressor(random_state=0)
        regr.fit(sample_df[covariates], sample_df[target])
        pred = regr.predict(test[covariates])
    
    elif config["dataset"] == "credit":
        linreg = sm.OLS(sample_df[target], sample_df[covariates]).fit()
        # print(linreg.summary())
        pred = linreg.predict(test[covariates])
            
    else:
        raise ValueError('Not supported dataset!')
    
    rsq = (test[target] - pred).pow(2).sum()
    rsq /= np.var(test[target]) * len(test)
    rsq = 1 - rsq
    print("[TVAE] R-squared: {:.3f}".format(rsq))
    wandb.log({'R^2 (TVAE)': rsq})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%