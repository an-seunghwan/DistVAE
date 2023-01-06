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
from modules.evaluation import (
    regression_eval,
    classification_eval
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
    project="DistVAE", 
    entity="anseunghwan",
    tags=['DistVAE', 'Synthetic'],
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
    
    dataset = "covtype"
    # dataset = "credit"
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(dataset, config["num"]), type='model')
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
    """Regression"""
    if config["dataset"] == "covtype":
        target = 'Elevation'
    elif config["dataset"] == "credit":
        target = 'AMT_INCOME_TOTAL'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Regression...\n")
    base_r2result = regression_eval(dataset.train, test_dataset.test, target)
    wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_r2result])})
    #%%
    # Inverse Transform Sampling
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    OutputInfo_list = dataset.OutputInfo_list
    n = len(dataset.train)
    with torch.no_grad():
        samples = model.generate_data(n, OutputInfo_list)
    ITS = pd.DataFrame(samples.numpy(), columns=dataset.train.columns)
    
    r2result = regression_eval(ITS, test_dataset.test, target)
    wandb.log({'R^2 (ITS)': np.mean([x[1] for x in r2result])})
    #%%
    # visualization
    fig = plt.figure(figsize=(5, 4))
    plt.plot([x[1] for x in base_r2result], 'o--', label='baseline')
    plt.plot([x[1] for x in r2result], 'o--', label='synthetic')
    plt.ylim(0, 1)
    plt.ylabel('$R^2$', fontsize=13)
    plt.xticks([0, 1, 2], [x[0] for x in base_r2result], fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_MLU_regression.png'.format(config["dataset"], config["dataset"]))
    plt.show()
    plt.close()
    wandb.log({'ML Utility (Regression)': wandb.Image(fig)})
    #%%
    """Classification"""
    if config["dataset"] == "covtype":
        target = 'Cover_Type'
    elif config["dataset"] == "credit":
        target = 'TARGET'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Classification...\n")
    base_f1result = classification_eval(dataset.train, test_dataset.test, target)
    wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
    #%%
    # Inverse Transform Sampling
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    OutputInfo_list = dataset.OutputInfo_list
    n = len(dataset.train)
    with torch.no_grad():
        samples = model.generate_data(n, OutputInfo_list)
    ITS = pd.DataFrame(samples.numpy(), columns=dataset.train.columns)
    
    f1result = classification_eval(ITS, test_dataset.test, target)
    wandb.log({'F1 (ITS)': np.mean([x[1] for x in f1result])})
    #%%
    # visualization
    fig = plt.figure(figsize=(5, 4))
    plt.plot([x[1] for x in base_f1result], 'o--', label='baseline')
    plt.plot([x[1] for x in f1result], 'o--', label='synthetic')
    plt.ylim(0, 1)
    plt.ylabel('$F_1$', fontsize=13)
    plt.xticks([0, 1, 2], [x[0] for x in base_f1result], fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_MLU_classification.png'.format(config["dataset"], config["dataset"]))
    plt.show()
    plt.close()
    wandb.log({'ML Utility (Classification)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%