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
from statsmodels.distributions.empirical_distribution import ECDF

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
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')
    parser.add_argument('--beta', default=0.1, type=float,
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
    if config["beta"] == 0.1:
        artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
            config["dataset"], config["seed"]), type='model')
    else:
        artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{:.1f}_DistVAE_{}:v{}'.format(
            config["beta"], config["dataset"], config["seed"]), type='model')
    
    stage1_config = dict(artifact.metadata.items())
    model_dir = artifact.download()
    #%%
    """Copula model load"""
    if config["beta"] == 0.1:
        artifact = wandb.use_artifact('anseunghwan/DistVAE/Copula_{}:v{}'.format(
            config["dataset"], config["num"]), type='model')
    else:
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
    # n = len(dataset.train)
    n = 1000
    
    """Gibbs sampling from copula CDF"""
    sample_size = 0
    burn_in = 100
    grid_points = 101
    
    samples = []
    for _ in tqdm.tqdm(range(n), desc="Sampling with Copula"):
        z = torch.randn(1, config["latent_dim"])
        # z = torch.zeros(1, config["latent_dim"])
        
        pseudo_samples = copula.gibbs_sampling(
            z=z, sample_size=sample_size, burn_in=burn_in, grid_points=grid_points)
        pseudo_samples = torch.from_numpy(pseudo_samples[[-1], :]).to(z.dtype) # last sample
    
        with torch.no_grad():
            gamma, beta, logit = model.quantile_parameter(z)
            
            tmp = []
            st = 0
            for j, info in enumerate(OutputInfo_list):
                if info.activation_fn == "CRPS":
                    tmp.append(
                        model.quantile_function(
                            pseudo_samples, gamma, beta, j)
                    )
                    
                elif info.activation_fn == "softmax":
                    ed = st + info.dim
                    out = logit[:, st : ed]
                    cdf = nn.Softmax(dim=1)(out).cumsum(dim=1)
                    label = pseudo_samples[:, [j]] > cdf
                    tmp.append(label.sum(dim=1, keepdims=True))
                    st = ed
        samples.append(
            torch.cat(tmp, dim=1))
    samples = torch.cat(samples, dim=0)
    #%%
    """save synthetic data from DisCoVAE"""
    np.save(
        "./assets/{}/beta{}_copula_syndata_{}".format(
            config["dataset"], config["beta"], config["num"]), # num -> seed
        samples.numpy())
    
    copula_syndata = pd.DataFrame(samples.numpy(), columns=dataset.continuous + dataset.discrete)
        
    """un-standardization of synthetic data"""
    copula_syndata[dataset.continuous] = copula_syndata[dataset.continuous] * dataset.std + dataset.mean
    
    """post-process integer columns (calibration)"""
    copula_syndata[dataset.integer] = copula_syndata[dataset.integer].round(0).astype(int)
    copula_syndata[dataset.discrete] = copula_syndata[dataset.discrete].astype(int)
    #%%
    fig, ax = plt.subplots(
        config["CRPS_dim"], config["CRPS_dim"], 
        figsize=(10, 10))
    for i in range(config["CRPS_dim"]):
        ecdf1 = ECDF(samples.numpy()[:, i])
        u1 = ecdf1(samples.numpy()[:, i])
        for j in range(config["CRPS_dim"]):
            ecdf2 = ECDF(samples.numpy()[:, j])
            u2 = ecdf2(samples.numpy()[:, j])
            ax[i, j].scatter(
                u1, u2,
                alpha=0.1)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("./assets/{}/DisCoVAE_CDF_plot.png".format(config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'CDF Plot (DisCoVAE)': wandb.Image(fig)})
    #%%
    """Synthetic Data Generation via DistVAE"""
    syndata = model.generate_data(n, OutputInfo_list, dataset)
    
    fig, ax = plt.subplots(
        config["CRPS_dim"], config["CRPS_dim"], 
        figsize=(10, 10))
    for i in range(config["CRPS_dim"]):
        ecdf1 = ECDF(syndata.to_numpy()[:, i])
        u1 = ecdf1(syndata.to_numpy()[:, i])
        for j in range(config["CRPS_dim"]):
            ecdf2 = ECDF(syndata.to_numpy()[:, j])
            u2 = ecdf2(syndata.to_numpy()[:, j])
            ax[i, j].scatter(
                u1, u2,
                alpha=0.1)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("./assets/{}/DistVAE_CDF_plot.png".format(config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'CDF Plot (DistVAE)': wandb.Image(fig)})
    #%%
    """Grount-truth dataset"""    
    fig, ax = plt.subplots(
        config["CRPS_dim"], config["CRPS_dim"], 
        figsize=(10, 10))
    for i in range(config["CRPS_dim"]):
        ecdf1 = ECDF(dataset.train_raw.to_numpy()[:n, i])
        u1 = ecdf1(dataset.train_raw.to_numpy()[:n, i])
        for j in range(config["CRPS_dim"]):
            ecdf2 = ECDF(dataset.train_raw.to_numpy()[:n, j])
            u2 = ecdf2(dataset.train_raw.to_numpy()[:n, j])
            ax[i, j].scatter(
                u1, u2,
                alpha=0.1)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("./assets/{}/True_CDF_plot.png".format(config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'CDF Plot (True)': wandb.Image(fig)})
    #%%
    # print("\nBaseline: Machine Learning Utility in Regression...\n")
    # base_reg = regression_eval(
    #     dataset.train.copy(), test_dataset.test.copy(), dataset.RegTarget, 
    #     dataset.mean[dataset.RegTarget], dataset.std[dataset.RegTarget])
    # wandb.log({'MARE (Baseline)': np.mean([x[1] for x in base_reg])})
    #%%
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    syndata_ = syndata.copy()
    syndata_[dataset.continuous] -= syndata_[dataset.continuous].mean(axis=0)
    syndata_[dataset.continuous] /= syndata_[dataset.continuous].std(axis=0)
    
    df_dummy = []
    for d in dataset.discrete:
        df_dummy.append(pd.get_dummies(syndata_[d], prefix=d))
    syndata_ = pd.concat([syndata_.drop(columns=dataset.discrete)] + df_dummy, axis=1)
    reg = regression_eval(
        syndata_.copy(), test_dataset.test.copy(), dataset.RegTarget, 
        syndata.mean()[dataset.RegTarget], syndata.std()[dataset.RegTarget])
    wandb.log({'MARE (DistVAE)': np.mean([x[1] for x in reg])})
    #%%
    copula_syndata_ = copula_syndata.copy()
    copula_syndata_[dataset.continuous] -= copula_syndata_[dataset.continuous].mean(axis=0)
    copula_syndata_[dataset.continuous] /= copula_syndata_[dataset.continuous].std(axis=0)
    
    df_dummy = []
    for d in dataset.discrete:
        df_dummy.append(pd.get_dummies(copula_syndata_[d], prefix=d))
    copula_syndata_ = pd.concat([copula_syndata_.drop(columns=dataset.discrete)] + df_dummy, axis=1)
    reg = regression_eval(
        copula_syndata_.copy(), test_dataset.test.copy(), dataset.RegTarget, 
        copula_syndata.mean()[dataset.RegTarget], copula_syndata.std()[dataset.RegTarget])
    wandb.log({'MARE (DisCoVAE)': np.mean([x[1] for x in reg])})
    #%%
    # print("\nBaseline: Machine Learning Utility in Classification...\n")
    # base_clf = classification_eval(
    #     dataset.train.copy(), test_dataset.test.copy(), dataset.ClfTarget)
    # wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_clf])})
    #%%
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    clf = classification_eval(
        syndata_.copy(), test_dataset.test.copy(), dataset.ClfTarget)
    wandb.log({'F1 (DistVAE)': np.mean([x[1] for x in clf])})
    #%%
    clf = classification_eval(
        copula_syndata_.copy(), test_dataset.test.copy(), dataset.ClfTarget)
    wandb.log({'F1 (DisCoVAE)': np.mean([x[1] for x in clf])})
    #%%
    """Correlation"""
    corr = np.corrcoef(syndata_.to_numpy()[:, :config["CRPS_dim"]].T)
    copula_corr = np.corrcoef(copula_syndata_.to_numpy()[:, :config["CRPS_dim"]].T)
    true_corr = np.corrcoef(dataset.x_data[:n, :config["CRPS_dim"]].T)
    #%%
    corr = np.abs(true_corr - corr).mean()
    copula_corr = np.abs(true_corr - copula_corr).mean()
    print(f'Corr MAE (DistVAE): {corr:.3f}')
    wandb.log({'Corr MAE (DistVAE)': corr})
    print(f'Corr MAE (DisCoVAE): {copula_corr:.3f}')
    wandb.log({'Corr MAE (DisCoVAE)': copula_corr})
    
    # the number of correlation direction matched
    corr_matched = ((true_corr > 0) * (corr > 0)).mean()
    copula_corr_matched = ((true_corr > 0) * (copula_corr > 0)).mean()
    print(f'Corr Dir (DistVAE): {corr_matched:.3f}')
    wandb.log({'Corr Dir (DistVAE)': corr_matched})
    print(f'Corr Dir (DisCoVAE): {copula_corr_matched:.3f}')
    wandb.log({'Corr Dir (DisCoVAE)': copula_corr_matched})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%
# print("\nStatistical Similarity...\n")
# Dn, W1 = statistical_similarity(
#     dataset.train_raw.copy(), syndata.copy(), 
#     standardize=True, continuous=dataset.continuous)
# cont_Dn = np.mean(Dn[:config["CRPS_dim"]])
# disc_Dn = np.mean(Dn[config["CRPS_dim"]:])
# cont_W1 = np.mean(W1[:config["CRPS_dim"]])
# disc_W1 = np.mean(W1[config["CRPS_dim"]:])

# print('K-S (continuous): {:.3f}'.format(cont_Dn))
# print('1-WD (continuous): {:.3f}'.format(cont_W1))
# print('K-S (discrete): {:.3f}'.format(disc_Dn))
# print('1-WD (discrete): {:.3f}'.format(disc_W1))
# wandb.log({'K-S (continuous)': cont_Dn})
# wandb.log({'1-WD (continuous)': cont_W1})
# wandb.log({'K-S (discrete)': disc_Dn})
# wandb.log({'1-WD (discrete)': disc_W1})
#%%
# print("\nDistance to Closest Record...\n")
# # standardization of synthetic data
# syndata_ = syndata.copy()
# syndata_[dataset.continuous] -= syndata_[dataset.continuous].mean(axis=0)
# syndata_[dataset.continuous] /= syndata_[dataset.continuous].std(axis=0)

# privacy = DCR_metric(
#     dataset.train[dataset.continuous].copy(), syndata_[dataset.continuous].copy())

# DCR = privacy
# print('DCR (R&S): {:.3f}'.format(DCR[0]))
# print('DCR (R): {:.3f}'.format(DCR[1]))
# print('DCR (S): {:.3f}'.format(DCR[2]))
# wandb.log({'DCR (R&S)': DCR[0]})
# wandb.log({'DCR (R)': DCR[1]})
# wandb.log({'DCR (S)': DCR[2]})
#%%
# print("\nAttribute Disclosure...\n")
# compromised_idx = np.random.choice(
#     range(len(dataset.train_raw)), 
#     int(len(dataset.train_raw) * 0.01), 
#     replace=False)
# train_raw_ = dataset.train_raw.copy()
# train_raw_[dataset.continuous] -= train_raw_[dataset.continuous].mean(axis=0)
# train_raw_[dataset.continuous] /= train_raw_[dataset.continuous].std(axis=0)
# compromised = train_raw_.iloc[compromised_idx].reset_index().drop(columns=['index'])

# # for attr_num in [1, 2, 3, 4, 5]:
# #     if attr_num > len(dataset.continuous): break
# attr_num = 5
# attr_compromised = dataset.continuous[:attr_num]
# for K in [1, 10, 100]:
#     acc, f1 = attribute_disclosure(
#         K, compromised, syndata_, attr_compromised, dataset)
#     print(f'AD F1 (S={attr_num},K={K}): {f1:.3f}')
#     wandb.log({f'AD F1 (S={attr_num},K={K})': f1})
#     # print(f'AD Accuracy (S={attr_num},K={K}): {acc:.3f}')
#     # wandb.log({f'AD Accuracy (S={attr_num},K={K})': acc})
#%%