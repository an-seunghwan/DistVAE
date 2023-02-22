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
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')
    # parser.add_argument('--beta', default=0.5, type=float,
    #                     help='observation noise')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    
    """model load"""
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_DistVAE_{}:v{}'.format(
    #     config["beta"], config["dataset"], config["num"]), type='model')
    artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
    stage1_config = dict(artifact.metadata.items())
    model_dir = artifact.download()
    #%%
    """Copula model load"""
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_Copula_{}:v{}'.format(
    #     config["beta"], config["dataset"], config["num"]), type='model')
    artifact = wandb.use_artifact('anseunghwan/DistVAE/Copula_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
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
    """Gibbs sampling from copula CDF"""
    n = len(dataset.train)
    # z = torch.randn(1, config["latent_dim"])
    z = torch.zeros(1, config["latent_dim"])
    burn_in = 5000
    grid_points = 101
    pseudo_samples = copula.gibbs_sampling(
        z=z, test_size=0, burn_in=burn_in, grid_points=grid_points)    
    pseudo_samples = torch.from_numpy(pseudo_samples).to(z.dtype)
    #%%
    with torch.no_grad():
        gamma, beta, logit = model.quantile_parameter(z)
        
        samples = []
        st = 0
        for j, info in enumerate(OutputInfo_list):
            if info.activation_fn == "CRPS":
                samples.append(
                    model.quantile_function(
                        pseudo_samples, gamma, beta, j)
                )
                
            elif info.activation_fn == "softmax":
                ed = st + info.dim
                out = logit[:, st : ed]
                cdf = nn.Softmax(dim=1)(out).cumsum(dim=1)
                label = pseudo_samples[:, [j]] > cdf
                samples.append(label.sum(dim=1, keepdims=True))
                st = ed

        samples = torch.cat(samples, dim=1)
    #%%
    corr = np.corrcoef(pseudo_samples.t())
    corr[corr == 1] = 0
    x, y = np.where(corr > 0.3)
    #%%
    fig, ax = plt.subplots(2, len(x) // 2 + 1, figsize=(15, 5))
    for i in range(len(x)):
        ax.flatten()[i].scatter(
            pseudo_samples[100:, x[i]], pseudo_samples[100:, y[i]],
            alpha=0.2)
    plt.tight_layout()
    # discard = 1000
    # plt.scatter(
    #     samples[discard:, 3], samples[discard:, 4],
    #     alpha=0.5)
    #%%
    pseudo = []
    with torch.no_grad():
        for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            
            """pseudo-observations"""
            # z, mean, logvar, gamma, beta, logit = model(x_batch, deterministic=True)
            z_ = torch.repeat_interleave(
                z,
                repeats=x_batch.size(0),
                dim=0
            )
            gamma, beta, logit = model.quantile_parameter(z_)
            
            # continuous
            alpha_tilde_list = model.quantile_inverse(x_batch, gamma, beta)
            cont_pseudo = torch.cat(alpha_tilde_list, dim=1)
            # discrete
            disc_pseudo = []
            st = 0
            for j, info in enumerate(OutputInfo_list):
                if info.activation_fn == "CRPS":
                    continue
                elif info.activation_fn == "softmax":
                    ed = st + info.dim
                    out = logit[:, st : ed]
                    cdf = nn.Softmax(dim=1)(out).cumsum(dim=1)
                    x_ = x_batch[:, config["CRPS_dim"] + st : config["CRPS_dim"] + ed]
                    disc_pseudo.append((cdf * x_).sum(axis=1, keepdims=True))
                    st = ed
            disc_pseudo = torch.cat(disc_pseudo, dim=1)
            
            pseudo.append(torch.cat([cont_pseudo, disc_pseudo], dim=1))
    pseudo = torch.cat(pseudo, dim=0)
    pseudo = pseudo[:burn_in, :]
    #%%
    fig, ax = plt.subplots(2, len(x) // 2 + 1, figsize=(15, 5))
    for i in range(len(x)):
        ax.flatten()[i].scatter(
            pseudo[:, x[i]], pseudo[:, y[i]],
            alpha=0.2)
    plt.tight_layout()
    # plt.scatter(
    #     pseudo[:, 3], pseudo[:, 4],
    #     alpha=0.1)
    #%%
    corr = np.corrcoef(pseudo_samples.t())
    corr[corr == 1] = 0
    print(corr[corr > 0.3])
    corr = np.corrcoef(pseudo.t())
    corr[corr == 1] = 0
    print(corr[corr > 0.3])
    #%%
    """XXX"""
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