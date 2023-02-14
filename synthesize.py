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
    merge_discrete,
    regression_eval,
    classification_eval,
    goodness_of_fit,
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
    # parser.add_argument('--beta', default=0.5, type=float,
    #                     help='observation noise')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    # artifact = wandb.use_artifact('anseunghwan/DistVAE/beta{}_DistVAE_{}:v{}'.format(
    #     config["beta"], config["dataset"], config["num"]), type='model')
    artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
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
    """Goodness of Fit""" 
    print("\nGoodness of Fit...\n")
    
    cut_points = merge_discrete(dataset.train.to_numpy(), config["CRPS_dim"])
    ITS_cut_points = merge_discrete(ITS_scaled.to_numpy(), config["CRPS_dim"])
    
    Dn, W1 = goodness_of_fit(config["CRPS_dim"], dataset.train.to_numpy(), ITS_scaled.to_numpy(), cut_points, ITS_cut_points)
    cont_Dn = np.mean(Dn[:config["CRPS_dim"]])
    disc_Dn = np.mean(Dn[config["CRPS_dim"]:])
    cont_W1 = np.mean(W1[:config["CRPS_dim"]])
    disc_W1 = np.mean(W1[config["CRPS_dim"]:])
    
    print('K-S (continuous): {:.3f}'.format(cont_Dn))
    print('K-S (discrete): {:.3f}'.format(disc_Dn))
    print('1-WD (continuous): {:.3f}'.format(cont_W1))
    print('1-WD (discrete): {:.3f}'.format(disc_W1))
    wandb.log({'K-S (continuous)': cont_Dn})
    wandb.log({'K-S (discrete)': disc_Dn})
    wandb.log({'1-WD (continuous)': cont_W1})
    wandb.log({'1-WD (discrete)': disc_W1})
    
    # Dn, W1 = goodness_of_fit(config, dataset.train.to_numpy(), ITS_scaled.to_numpy())
    
    # print('Goodness of Fit (Kolmogorov): {:.3f}'.format(Dn))
    # print('Goodness of Fit (1-Wasserstein): {:.3f}'.format(W1))
    # wandb.log({'Goodness of Fit (Kolmogorov)': Dn})
    # wandb.log({'Goodness of Fit (1-Wasserstein)': W1})
    #%%
    """Privacy Preservability""" # only continuous
    print("\nDistance to Closest Record...\n")
    
    privacy = DCR_metric(dataset.train[dataset.continuous], ITS_scaled[dataset.continuous])
    
    DCR = privacy
    # DCR = privacy[0, :3]
    print('DCR (R&S): {:.3f}'.format(DCR[0]))
    print('DCR (R): {:.3f}'.format(DCR[1]))
    print('DCR (S): {:.3f}'.format(DCR[2]))
    wandb.log({'DCR (R&S)': DCR[0]})
    wandb.log({'DCR (R)': DCR[1]})
    wandb.log({'DCR (S)': DCR[2]})
    
    # NNDR = privacy[0, 3:]
    # print('NNDR (R&S): {:.3f}'.format(NNDR[0]))
    # print('NNDR (R): {:.3f}'.format(NNDR[1]))
    # print('NNDR (S): {:.3f}'.format(NNDR[2]))
    # wandb.log({'NNDR (R&S)': NNDR[0]})
    # wandb.log({'NNDR (R)': NNDR[1]})
    # wandb.log({'NNDR (S)': NNDR[2]})
    #%%
    print("\nAttribute Disclosure...\n")
    
    cut_points = merge_discrete(ITS_scaled.to_numpy(), config["CRPS_dim"])
    
    compromised_idx = np.random.choice(range(len(dataset.train)), 
                                       int(len(dataset.train) * 0.01), 
                                       replace=False)
    compromised = dataset.train.iloc[compromised_idx]
    #%%
    for attr_num in [1, 2, 3, 4, 5]:
        if attr_num > len(dataset.continuous): break
        attr_compromised = dataset.continuous[:attr_num]
        for K in [1, 10, 100]:
            acc, f1 = attribute_disclosure(
                K, compromised, ITS_scaled, attr_compromised, cut_points, config["CRPS_dim"]
            )
            print(f'AD Accuracy (S={attr_num},K={K}):', acc)
            print(f'AD F1 (S={attr_num},K={K}):', f1)
            wandb.log({f'AD Accuracy (S={attr_num},K={K})': acc})
            wandb.log({f'AD F1 (S={attr_num},K={K})': f1})
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
    wandb.log({'MARE (Baseline)': np.mean([x[1] for x in base_reg])})
    # wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_reg])})
    #%%
    # Inverse Transform Sampling
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    reg = regression_eval(ITS_scaled, real_test, target)
    wandb.log({'MARE': np.mean([x[1] for x in reg])})
    # wandb.log({'R^2': np.mean([x[1] for x in reg])})
    #%%
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot([x[1] for x in base_reg], 'o--', label='baseline')
    # plt.plot([x[1] for x in reg], 'o--', label='synthetic')
    # plt.ylim(0, 1)
    # plt.ylabel('MARE', fontsize=13)
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
    wandb.log({'F1': np.mean([x[1] for x in clf])})
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