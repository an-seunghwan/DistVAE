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
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
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
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["Tabular", "TVAE", "Inference"],
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
    
    # dataset = 'loan'
    dataset = 'adult'
    # dataset = 'covtype'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/CausalDisentangled/TVAE_{}:v{}'.format(dataset, config["num"]), type='model')
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
    
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    if config["dataset"] == 'loan':
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
        continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        df = df[continuous]
        
        df_ = (df - df.mean(axis=0)) / df.std(axis=0)
        train = df_.iloc[:4000]
        test = df_.iloc[4000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        df = df[continuous]
        
        train = df.iloc[:40000]
        test = df.iloc[40000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            'Slope', 
            'Cover_Type']
        df = df[continuous]
        df = df.dropna(axis=0)
        
        train = df.iloc[2000:, ]
        test = df.iloc[:2000, ]
        
        i_test = 'fisherz'
        
    else:
        raise ValueError('Not supported dataset!')
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.05, 
            indep_test=i_test) 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    """train dataset representation"""
    train_recon = []
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        with torch.no_grad():
            out = model(x_batch, deterministic=True)
        train_recon.append(out[-1])
    train_recon = torch.cat(train_recon, dim=0)
    train_recon = transformer.inverse_transform(train_recon, model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : train dataset representation"""
    train_recon = train_recon[df.columns]
    cg = pc(data=train_recon.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    trainSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    trainSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    print('SHD (Train): {}'.format(trainSHD))
    wandb.log({'SHD (Train)': trainSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train_recon.columns)
    pdy.write_png('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    steps = len(train) // config["batch_size"] + 1
    data = []
    with torch.no_grad():
        for _ in range(steps):
            mean = torch.zeros(config["batch_size"], config["node"])
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(device)
            fake = model.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:len(train)]
    sample_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    #%%
    """PC algorithm : synthetic dataset"""
    sample_df = sample_df[df.columns]
    cg = pc(data=sample_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    sampleSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    sampleSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    print('SHD (Sample): {}'.format(sampleSHD))
    wandb.log({'SHD (Sample)': sampleSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """Machine Learning Efficacy"""
    
    # Baseline
    if config["dataset"] == 'loan':
        covariates = [x for x in train.columns if x != 'CCAvg']
        linreg = sm.OLS(train['CCAvg'], train[covariates]).fit()
        pred = linreg.predict(test[covariates])
        rsq_baseline = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
        
        print("Baseline R-squared: {:.2f}".format(rsq_baseline))
        wandb.log({'R^2 (Baseline)': rsq_baseline})
        
    elif config["dataset"] == 'adult':
        covariates = [x for x in train.columns if x != 'income']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train[covariates], train['income'])
        pred = clf.predict(test[covariates])
        # logistic = sm.Logit(train['income'], train[covariates]).fit()
        # pred = logistic.predict(test[covariates])
        pred = (pred > 0.5).astype(float)
        f1_baseline = f1_score(test['income'], pred)
        
        print("Baseline F1: {:.2f}".format(f1_baseline))
        wandb.log({'F1 (Baseline)': f1_baseline})
    
    elif config["dataset"] == 'covtype':
        covariates = [x for x in train.columns if x != 'Cover_Type']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train[covariates], train['Cover_Type'])
        pred = clf.predict(test[covariates])
        f1_baseline = f1_score(test['Cover_Type'].to_numpy(), pred, average='micro')
        # acc_baseline = clf.score(test[covariates], test['Cover_Type'])
        
        print("Baseline F1: {:.2f}".format(f1_baseline))
        wandb.log({'F1 (Baseline)': f1_baseline})
    
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # synthetic
    if config["dataset"] == 'loan':
        covariates = [x for x in sample_df.columns if x != 'CCAvg']
        sample_df[covariates] = (sample_df[covariates] - sample_df[covariates].mean(axis=0)) / sample_df[covariates].std(axis=0)
        
        covariates = [x for x in sample_df.columns if x != 'CCAvg']
        linreg = sm.OLS(sample_df['CCAvg'], sample_df[covariates]).fit()
        pred = linreg.predict(test[covariates])
        rsq = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
        
        print("{} R-squared: {:.2f}".format(config["dataset"], rsq))
        wandb.log({'R^2 (Sample)': rsq})
        
    elif config["dataset"] == 'adult':
        covariates = [x for x in sample_df.columns if x != 'income']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(sample_df[covariates], sample_df['income'])
        pred = clf.predict(test[covariates])
        # logistic = sm.Logit(sample_df['income'], sample_df[covariates]).fit()
        # pred = logistic.predict(test[covariates])
        pred = (pred > 0.5).astype(float)
        f1 = f1_score(test['income'], pred)
        
        print("{} F1: {:.2f}".format(config["dataset"], f1))
        wandb.log({'F1 (Sample)': f1})
    
    elif config["dataset"] == 'covtype':
        covariates = [x for x in train.columns if x != 'Cover_Type']
        
        clf = RandomForestClassifier(random_state=0)
        clf.fit(sample_df[covariates], sample_df['Cover_Type'])
        pred = clf.predict(test[covariates])
        f1 = f1_score(test['Cover_Type'].to_numpy(), pred, average='micro')
        # acc = clf.score(test[covariates], test['Cover_Type'])
        
        print("{} F1: {:.2f}".format(config["dataset"], f1))
        wandb.log({'F1 (Sample)': f1})
    
    else:
        raise ValueError('Not supported dataset!')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%