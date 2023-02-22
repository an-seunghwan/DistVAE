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

import sys
sys.path.append('..')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.simulation import set_random_seed
from modules.model import VAE

from copula_modules.copula import NCECopula
from copula_modules.train import train_copula
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
    tags=['DistVAE', 'Copula'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    # Stage 1
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')
    # parser.add_argument('--beta', default=0.5, type=float,
    #                     help='scale parameter of asymmetric Laplace distribution')
    # parser.add_argument('--seed', type=int, default=1, 
    #                     help='seed for repeatable results')
    
    # Stage 2
    parser.add_argument("--latent_dim", default=2, type=int,
                        help="the latent dimension size")
    
    parser.add_argument('--epochs', default=150, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """DistVAE model (Stage 1) load"""
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_DistVAE_{}:v{}'.format(
    #     config["beta"], config["dataset"], config["num"]), type='model')
    artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
    stage1_config = dict(artifact.metadata.items())
    config["seed"] = config["num"]
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    OutputInfo_list = dataset.OutputInfo_list
    CRPS_dim = sum([x.dim for x in OutputInfo_list if x.activation_fn == 'CRPS'])
    softmax_dim = sum([x.dim for x in OutputInfo_list if x.activation_fn == 'softmax'])
    config["CRPS_dim"] = CRPS_dim
    config["softmax_dim"] = softmax_dim
    config["data_dim"] = len(OutputInfo_list)
    #%%
    model = VAE(stage1_config, device).to(device)
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
    """Copula Model"""
    copula = NCECopula(config, device)
    #%%
    for epoch in range(config["epochs"]):
        logs = train_copula(OutputInfo_list, dataloader, model, copula, config, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """Copula model save"""
    torch.save(copula.model.state_dict(), './assets/Copula_{}.pth'.format(config["dataset"]))
    # artifact = wandb.Artifact('beta{}_Copula_{}'.format(config["beta"], config["dataset"]), 
    #                         type='model',
    #                         metadata=config) # description=""
    artifact = wandb.Artifact('Copula_{}'.format(config["dataset"]), 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file('./assets/Copula_{}.pth'.format(config["dataset"]))
    artifact.add_file('./main.py')
    artifact.add_file('./copula_modules/copula.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%