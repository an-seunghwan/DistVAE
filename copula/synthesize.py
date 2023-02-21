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

from modules.evaluation import (
    regression_eval,
    classification_eval,
    statistical_similarity,
    DCR_metric,
    attribute_disclosure
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
    tags=['DistVAE', 'Copula', 'Synthetic'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='credit', 
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
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_DistVAE_{}:v{}'.format(
        config["beta"], config["dataset"], config["num"]), type='model')
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
    #     config["dataset"], config["num"]), type='model')
    stage1_config = dict(artifact.metadata.items())
    model_dir = artifact.download()
    #%%
    """Copula model load"""
    artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_Copula_{}:v{}'.format(
        config["beta"], config["dataset"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    copula_model_dir = artifact.download()
    
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
    copula = NCECopula(config, device)
    if config["cuda"]:
        model_name = [x for x in os.listdir(copula_model_dir) if x.endswith('pth')][0]
        copula.model.load_state_dict(
            torch.load(
                copula_model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(copula_model_dir) if x.endswith('pth')][0]
        copula.model.load_state_dict(
            torch.load(
                copula_model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    copula.model.eval()
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(copula.model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    """Synthetic Data Generation"""
    n = len(dataset.train)
    burn_in = 1000
    grid_points = 51
    
    a = np.linspace(0, 1, grid_points)
    uv_samples = np.zeros((burn_in, config["data_dim"]))
    uv_samples[0, :] = np.random.uniform(0, 1, config["data_dim"]) # random initialization
    
    z = np.random.normal(size=(1, config["latent_dim"]))
    z = np.repeat(z, repeats=grid_points, axis=0).reshape(-1, config["latent_dim"]) # fixed
    
    for t in tqdm.tqdm(range(1, burn_in), desc="Gibbs Sampling..."):
        for i in range(config["data_dim"]):
            if i == 0:
                # (t-1)th sample -> coordinates 1, ..., d-1
                # (t)th sample -> coordinate 0 : grid point approximate
                uv_i_vector = np.concatenate(
                    (a.reshape(-1, 1), np.repeat(
                        uv_samples[[t-1], i+1:config["data_dim"]],
                        repeats=grid_points, axis=0).reshape(grid_points, -1)), axis=1)
                
            elif i > 0 and i < config["data_dim"] - 1:
                # (t-1)th sample -> coordinates k+1, ..., d-1 where k > 0
                # (t)th sample -> coordinate 0, ..., k-1
                # (t)th sample -> coordinate k : grid point approximate
                uv_i_vector_left = np.concatenate(
                    (np.repeat(
                        uv_samples[[t], 0:i],
                        repeats=grid_points, axis=0).reshape(grid_points, -1),
                    a.reshape(-1, 1)), axis=1)
                uv_i_vector = np.concatenate(
                    (uv_i_vector_left, 
                    np.repeat(
                        uv_samples[[t-1], i+1:config["data_dim"]],
                        repeats=grid_points, axis=0).reshape(grid_points, -1)), axis=1)
                
            else:
                # (t-1)th sample -> coordinates 0, 1, ..., d-2
                # (t)th sample -> coordinate d-1 : grid point approximate
                uv_i_vector = np.concatenate(
                    (np.repeat(
                        uv_samples[[t], 0:i],
                        repeats=grid_points, axis=0).reshape(grid_points, -1),
                    a.reshape(-1, 1)), axis=1)
            
            with torch.no_grad():
                h = copula.model(
                        torch.cat([
                            torch.from_numpy(uv_i_vector).to(torch.float32),
                            torch.from_numpy(z).to(torch.float32)], dim=1))
            conditional_density = h
            conditional_density /= conditional_density.sum()
            
            icdf = copula.inverse_transform_sampling(
                conditional_density.numpy().squeeze(1),
                np.linspace(0, 1, grid_points + 1))
            uv_samples[t, i] = icdf(np.random.uniform())
    #%%
    for i in range(uv_samples.shape[1]):
        plt.plot(uv_samples[:, i])
    #%%
    print("\nStatistical Similarity...\n")
    Dn, W1 = statistical_similarity(
        dataset.train_raw.copy(), syndata.copy(), 
        standardize=True, continuous=dataset.continuous)
    cont_Dn = np.mean(Dn[:config["CRPS_dim"]])
    disc_Dn = np.mean(Dn[config["CRPS_dim"]:])
    cont_W1 = np.mean(W1[:config["CRPS_dim"]])
    disc_W1 = np.mean(W1[config["CRPS_dim"]:])
    
    print('K-S (continuous): {:.3f}'.format(cont_Dn))
    print('1-WD (continuous): {:.3f}'.format(cont_W1))
    print('K-S (discrete): {:.3f}'.format(disc_Dn))
    print('1-WD (discrete): {:.3f}'.format(disc_W1))
    wandb.log({'K-S (continuous)': cont_Dn})
    wandb.log({'1-WD (continuous)': cont_W1})
    wandb.log({'K-S (discrete)': disc_Dn})
    wandb.log({'1-WD (discrete)': disc_W1})
    #%%
    print("\nDistance to Closest Record...\n")
    # standardization of synthetic data
    syndata_ = syndata.copy()
    syndata_[dataset.continuous] -= syndata_[dataset.continuous].mean(axis=0)
    syndata_[dataset.continuous] /= syndata_[dataset.continuous].std(axis=0)
    
    privacy = DCR_metric(
        dataset.train[dataset.continuous].copy(), syndata_[dataset.continuous].copy())
    
    DCR = privacy
    print('DCR (R&S): {:.3f}'.format(DCR[0]))
    print('DCR (R): {:.3f}'.format(DCR[1]))
    print('DCR (S): {:.3f}'.format(DCR[2]))
    wandb.log({'DCR (R&S)': DCR[0]})
    wandb.log({'DCR (R)': DCR[1]})
    wandb.log({'DCR (S)': DCR[2]})
    #%%
    print("\nAttribute Disclosure...\n")
    compromised_idx = np.random.choice(
        range(len(dataset.train_raw)), 
        int(len(dataset.train_raw) * 0.01), 
        replace=False)
    train_raw_ = dataset.train_raw.copy()
    train_raw_[dataset.continuous] -= train_raw_[dataset.continuous].mean(axis=0)
    train_raw_[dataset.continuous] /= train_raw_[dataset.continuous].std(axis=0)
    compromised = train_raw_.iloc[compromised_idx].reset_index().drop(columns=['index'])
    
    # for attr_num in [1, 2, 3, 4, 5]:
    #     if attr_num > len(dataset.continuous): break
    attr_num = 5
    attr_compromised = dataset.continuous[:attr_num]
    for K in [1, 10, 100]:
        acc, f1 = attribute_disclosure(
            K, compromised, syndata_, attr_compromised, dataset)
        print(f'AD F1 (S={attr_num},K={K}): {f1:.3f}')
        wandb.log({f'AD F1 (S={attr_num},K={K})': f1})
        # print(f'AD Accuracy (S={attr_num},K={K}): {acc:.3f}')
        # wandb.log({f'AD Accuracy (S={attr_num},K={K})': acc})
    #%%
    print("\nBaseline: Machine Learning Utility in Regression...\n")
    base_reg = regression_eval(
        dataset.train.copy(), test_dataset.test.copy(), dataset.RegTarget, 
        dataset.mean[dataset.RegTarget], dataset.std[dataset.RegTarget])
    wandb.log({'MARE (Baseline)': np.mean([x[1] for x in base_reg])})
    #%%
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    df_dummy = []
    for d in dataset.discrete:
        df_dummy.append(pd.get_dummies(syndata_[d], prefix=d))
    syndata_ = pd.concat([syndata_.drop(columns=dataset.discrete)] + df_dummy, axis=1)
    reg = regression_eval(
        syndata_.copy(), test_dataset.test.copy(), dataset.RegTarget, 
        syndata.mean()[dataset.RegTarget], syndata.std()[dataset.RegTarget])
    wandb.log({'MARE': np.mean([x[1] for x in reg])})
    #%%
    print("\nBaseline: Machine Learning Utility in Classification...\n")
    base_clf = classification_eval(
        dataset.train.copy(), test_dataset.test.copy(), dataset.ClfTarget)
    wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_clf])})
    #%%
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    clf = classification_eval(
        syndata_.copy(), test_dataset.test.copy(), dataset.ClfTarget)
    wandb.log({'F1': np.mean([x[1] for x in clf])})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%