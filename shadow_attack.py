#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.simulation import set_random_seed
from modules.model import VAE
from modules.train import train_VAE

from sklearn.metrics import precision_score, recall_score
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
    project="DistVAE", 
    entity="anseunghwan",
    tags=['DistVAE', 'Privacy', 'Attack'],
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
    model_dirs = []
    for n in tqdm.tqdm(range(10), desc="Loading trained shadow models..."):
        num = config["num"] * 10 + n
        artifact = wandb.use_artifact('anseunghwan/DistVAE/shadow_DistVAE_{}:v{}'.format(config["dataset"], num), type='model')
        for key, item in artifact.metadata.items():
            config[key] = item
        model_dir = artifact.download()
        model_dirs.append(model_dir)
    
    if not os.path.exists('./privacy/{}'.format(config["dataset"])):
        os.makedirs('./privacy/{}'.format(config["dataset"]))
    
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
    #%%
    for k in range(len(model_dirs)):
        model_dir = model_dirs[k]
        
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
        # shadow data
        class ShadowTabularDataset(Dataset): 
            def __init__(self, shadow_data):
                self.x_data = shadow_data.to_numpy()
                
            def __len__(self): 
                return len(self.x_data)

            def __getitem__(self, idx): 
                x = torch.FloatTensor(self.x_data[idx])
                return x
        
        targets = []
        shadow_data = []
        for k in range(len(model_dirs)):
            df = pd.read_csv(f'./privacy/{config["dataset"]}/train_{config["seed"]}_synthetic{k}.csv', index_col=0)
            targets.append(df[[x for x in df.columns if x.startswith(target)]].to_numpy().argmax(axis=1))
            shadow_data.append(ShadowTabularDataset(df))
        targets_test = []
        shadow_data_test = []
        for k in range(len(model_dirs)):
            df = pd.read_csv(f'./privacy/{config["dataset"]}/test_{config["seed"]}_synthetic{k}.csv', index_col=0)
            targets_test.append(df[[x for x in df.columns if x.startswith(target)]].to_numpy().argmax(axis=1))
            shadow_data_test.append(ShadowTabularDataset(df))
        #%%
        """training latent variables"""
        latents = []
        for k in range(len(model_dirs)):
            dataloader = DataLoader(shadow_data[k], batch_size=config["batch_size"], shuffle=False)
            
            zs = []
            for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                with torch.no_grad():
                    mean, logvar = model.get_posterior(x_batch)
                zs.append(mean)
            zs = torch.cat(zs, dim=0)
            latents.append(zs)
        #%%
        """test latent variables"""
        latents_test = []
        for k in range(len(model_dirs)):
            dataloader = DataLoader(shadow_data_test[k], batch_size=config["batch_size"], shuffle=False)
            
            zs = []
            for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
                if config["cuda"]:
                    x_batch = x_batch.cuda()
                with torch.no_grad():
                    mean, logvar = model.get_posterior(x_batch)
                zs.append(mean)
            zs = torch.cat(zs, dim=0)
            latents_test.append(zs)
        #%%
        """attack training records"""
        target_num = dataset.train[[x for x in df.columns if x.startswith(target)]].shape[1]
        attack_training = {}
        for t in range(target_num):
            tmp1 = []
            for k in range(len(model_dirs)):
                tmp1.append(latents[k].numpy()[[targets[k] == t][0], :])
            tmp1 = np.concatenate(tmp1, axis=0)
            tmp1 = np.concatenate([tmp1, np.ones((len(tmp1), 1))], axis=1)
            
            tmp2 = []
            for k in range(len(model_dirs)):
                tmp2.append(latents_test[k].numpy()[[targets_test[k] == t][0], :])
            tmp2 = np.concatenate(tmp2, axis=0)
            tmp2 = np.concatenate([tmp2, np.zeros((len(tmp2), 1))], axis=1)
            
            tmp = np.concatenate([tmp1, tmp2], axis=0)
            
            attack_training[t] = tmp
        #%%
        """training attack model"""
        from sklearn.ensemble import GradientBoostingClassifier
        attackers = {}
        for k in range(target_num):
            clf = GradientBoostingClassifier(random_state=0).fit(
                attack_training[k][:, :config["latent_dim"]], 
                attack_training[k][:, -1])
            attackers[k] = clf
        #%%
        artifact = wandb.use_artifact('anseunghwan/DistVAE/DistVAE_{}:v{}'.format(config["dataset"], config["seed"]), type='model')
        for key, item in artifact.metadata.items():
            config[key] = item
        model_dir = artifact.download()
        
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
        
        dataset = TabularDataset()
        test_dataset = TabularDataset(train=False)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        #%%
        """Ground-truth training latent variables"""
        gt_latents = []
        for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            with torch.no_grad():
                mean, logvar = model.get_posterior(x_batch)
            gt_latents.append(mean)
        gt_latents = torch.cat(gt_latents, dim=0)
        #%%
        """Ground-truth test latent variables"""
        gt_latents_test = []
        for (x_batch) in tqdm.tqdm(iter(test_dataloader), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            with torch.no_grad():
                mean, logvar = model.get_posterior(x_batch)
            gt_latents_test.append(mean)
        gt_latents_test = torch.cat(gt_latents_test, dim=0)
        #%%
        """attacker accuracy"""
        gt_targets = dataset.train[[x for x in df.columns if x.startswith(target)]].to_numpy().argmax(axis=1)
        gt_targets_test = test_dataset.test[[x for x in df.columns if x.startswith(target)]].to_numpy().argmax(axis=1)
        
        gt_latents = gt_latents[:len(gt_latents_test), :]
        gt_targets = gt_targets[:len(gt_latents_test)]
        
        pred = []
        for t in range(target_num):
            pred.append(attackers[t].predict(gt_latents[gt_targets == t]))
        for t in range(target_num):
            pred.append(attackers[t].predict(gt_latents_test[gt_targets_test == t]))
        pred = np.concatenate(pred)
        
        gt = np.zeros((len(pred), ))
        gt[:len(gt_latents)] = 1
        
        precision = precision_score(gt, pred)
        recall = recall_score(gt, pred)
        
        print('MI Precision:', precision)
        print('MI Recacll:', recall)
        wandb.log({'MI Precision' : precision})
        wandb.log({'MI Recacll' : recall})
    #%%    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%