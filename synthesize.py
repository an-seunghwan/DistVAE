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
    classification_eval,
    goodness_of_fit,
    privacy_metrics
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
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    if config["dataset"] == 'credit':
        latents = []
        dataloader_ = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        for (x_batch) in tqdm.tqdm(iter(dataloader_), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            
            with torch.no_grad():
                mean, logvar = model.get_posterior(x_batch)
            latents.append(mean)
        latents = torch.cat(latents, dim=0).numpy()
        labeles = dataset.train['TARGET_1'].to_numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(
            latents[labeles == 0, 0], latents[labeles == 0, 1],
            s=30, c='blue', alpha=0.5,
            label="0")
        ax[0].scatter(
            latents[labeles == 1, 0], latents[labeles == 1, 1],
            s=30, c='red', alpha=0.5,
            label="1")
        ax[0].set_xlim(-4, 4)
        ax[0].set_ylim(-4, 4)
        ax[0].set_xlabel('$z_1$', fontsize=18)
        ax[0].set_ylabel('$z_2$', fontsize=18)
        ax[0].legend()
        
        ax[1].bar(
            [0, 1], 
            [(labeles == 0).mean(), (labeles == 1).mean()])
        ax[1].set_xticks([0, 1], ["0", "1"])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel('labels', fontsize=18)
        ax[1].set_ylabel('proportion', fontsize=18)
        
        plt.tight_layout()
        plt.savefig('./assets/{}/{}_latent_space_and_label_ratio.png'.format(config["dataset"], config["dataset"]))
        # plt.show()
        plt.close()
    #%%
    """Inverse Transform Sampling"""
    OutputInfo_list = dataset.OutputInfo_list
    n = len(dataset.train)
    with torch.no_grad():
        samples = model.generate_data(n, OutputInfo_list)
    ITS = pd.DataFrame(samples.numpy(), columns=dataset.train.columns)
    
    # un-standardization of synthetic data
    ITS[dataset.continuous] = ITS[dataset.continuous] * dataset.std + dataset.mean
    #%%
    # standardization of synthetic data
    ITS_mean = ITS[dataset.continuous].mean(axis=0)
    ITS_std = ITS[dataset.continuous].std(axis=0)
    ITS_scaled = ITS.copy()
    ITS_scaled[dataset.continuous] = (ITS[dataset.continuous] - ITS_mean) / ITS_std
    #%%
    """Goodness of Fit""" # only continuous
    print("\nGoodness of Fit...\n")
    
    Dn, W1 = goodness_of_fit(config, dataset.train.to_numpy(), ITS_scaled.to_numpy())
    
    print('Goodness of Fit (Kolmogorov): {:.3f}'.format(Dn))
    print('Goodness of Fit (1-Wasserstein): {:.3f}'.format(W1))
    wandb.log({'Goodness of Fit (Kolmogorov)': Dn})
    wandb.log({'Goodness of Fit (1-Wasserstein)': W1})
    #%%
    """Privacy Preservability""" # only continuous
    print("\nPrivacy Preservability...\n")
    
    privacy = privacy_metrics(dataset.train[dataset.continuous], ITS_scaled[dataset.continuous])
    
    DCR = privacy[0, :3]
    print('DCR (R&S): {:.3f}'.format(DCR[0]))
    print('DCR (R): {:.3f}'.format(DCR[1]))
    print('DCR (S): {:.3f}'.format(DCR[2]))
    wandb.log({'DCR (R&S)': DCR[0]})
    wandb.log({'DCR (R)': DCR[1]})
    wandb.log({'DCR (S)': DCR[2]})
    
    NNDR = privacy[0, 3:]
    print('NNDR (R&S): {:.3f}'.format(NNDR[0]))
    print('NNDR (R): {:.3f}'.format(NNDR[1]))
    print('NNDR (S): {:.3f}'.format(NNDR[2]))
    wandb.log({'NNDR (R&S)': NNDR[0]})
    wandb.log({'NNDR (R)': NNDR[1]})
    wandb.log({'NNDR (S)': NNDR[2]})
    #%%
    # dataset.train[dataset.continuous].hist(figsize=(10, 10))
    # ITS[dataset.continuous].hist(figsize=(10, 10))
    #%%
    """Regression"""
    if config["dataset"] == "covtype":
        target = 'Elevation'
    elif config["dataset"] == "credit":
        target = 'AMT_CREDIT'
    elif config["dataset"] == "loan":
        target = 'Age'
    elif config["dataset"] == "adult":
        target = 'age'
    elif config["dataset"] == "cabs":
        target = 'Trip_Distance'
    elif config["dataset"] == "kings":
        target = 'long'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # standardization except for target variable
    real_train = dataset.train.copy()
    real_test = test_dataset.test.copy()
    real_train[target] = real_train[target] * dataset.std[target] + dataset.mean[target]
    real_test[target] = real_test[target] * dataset.std[target] + dataset.mean[target]
    
    cont = [x for x in dataset.continuous if x not in [target]]
    ITS_scaled = ITS.copy()
    ITS_scaled[cont] = (ITS_scaled[cont] - ITS_mean[cont]) / ITS_std[cont]
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Regression...\n")
    base_reg = regression_eval(real_train, real_test, target)
    wandb.log({'MAPE (Baseline)': np.mean([x[1] for x in base_reg])})
    # wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_reg])})
    #%%
    # Inverse Transform Sampling
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    reg = regression_eval(ITS_scaled, real_test, target)
    wandb.log({'MAPE (ITS)': np.mean([x[1] for x in reg])})
    # wandb.log({'R^2 (ITS)': np.mean([x[1] for x in reg])})
    #%%
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot([x[1] for x in base_reg], 'o--', label='baseline')
    # plt.plot([x[1] for x in reg], 'o--', label='synthetic')
    # plt.ylim(0, 100)
    # plt.ylabel('MAPE', fontsize=13)
    # # plt.ylim(0, 1)
    # # plt.ylabel('$R^2$', fontsize=13)
    # plt.xticks([0, 1, 2], [x[0] for x in base_reg], fontsize=13)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/{}_MLU_regression.png'.format(config["dataset"], config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'ML Utility (Regression)': wandb.Image(fig)})
    #%%
    """Classification"""
    if config["dataset"] == "covtype":
        target = 'Cover_Type'
    elif config["dataset"] == "credit":
        target = 'TARGET'
    elif config["dataset"] == "loan":
        target = 'Personal Loan'
    elif config["dataset"] == "adult":
        target = 'income'
    elif config["dataset"] == "cabs":
        target = 'Surge_Pricing_Type'
    elif config["dataset"] == "kings":
        target = 'condition'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Classification...\n")
    base_clf = classification_eval(dataset.train, test_dataset.test, target)
    wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_clf])})
    #%%
    ITS_scaled = ITS.copy()
    ITS_scaled[dataset.continuous] = (ITS_scaled[dataset.continuous] - ITS_mean[dataset.continuous]) / ITS_std[dataset.continuous]
    
    # Inverse Transform Sampling
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    clf = classification_eval(ITS_scaled, test_dataset.test, target)
    wandb.log({'F1 (ITS)': np.mean([x[1] for x in clf])})
    #%%
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot([x[1] for x in base_clf], 'o--', label='baseline')
    # plt.plot([x[1] for x in clf], 'o--', label='synthetic')
    # plt.ylim(0, 1)
    # plt.ylabel('$F_1$', fontsize=13)
    # plt.xticks([0, 1, 2], [x[0] for x in base_clf], fontsize=13)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/{}_MLU_classification.png'.format(config["dataset"], config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'ML Utility (Classification)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%