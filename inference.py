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
    project="VAE(CRPS)", 
    entity="anseunghwan",
    tags=["Inference"],
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
    
    # dataset = "covtype"
    dataset = "credit"
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/VAE(CRPS)/model_{}:v{}'.format(dataset, config["num"]), type='model')
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
    TestTabularDataset = dataset_module.TestTabularDataset
    
    dataset = TabularDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = TestTabularDataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    # print(test_dataset.transformer.output_dimensions)
    
    if config["vgmm"]:
        config["input_dim"] = dataset.transformer.output_dimensions
    else:
        config["input_dim"] = len(dataset.continuous)
        # config["input_dim"] = len(dataset.continuous + dataset.discrete)
    # config["output_dim"] = len(dataset.continuous)
    # config["output_dim"] = len(dataset.continuous + dataset.discrete)
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
    # """3D visualization of quantile function"""
    # if not os.path.exists("./assets/{}/latent_quantile".format(config["dataset"])): 
    #     os.makedirs("./assets/{}/latent_quantile".format(config["dataset"]))
            
    # xs = torch.linspace(-2, 2, steps=30)
    # ys = torch.linspace(-2, 2, steps=30)
    # x, y = torch.meshgrid(xs, ys)
    # grid_z = torch.cat([x.flatten()[:, None], y.flatten()[:, None]], axis=1)
    
    # j = 1
    # alpha = 0.5
    # for j in range(config["input_dim"]):
    #     quantiles = []
    #     for alpha in np.linspace(0.1, 0.9, 9):
    #         with torch.no_grad():
    #             gamma, beta = model.quantile_parameter(grid_z)
    #             quantiles.append(model.quantile_function(alpha, gamma, beta, j))
        
    #     fig = plt.figure(figsize=(6, 4))
    #     ax = fig.gca(projection='3d')
    #     for i in range(len(quantiles)):
    #         ax.plot_surface(x.numpy(), y.numpy(), quantiles[i].reshape(x.shape).numpy())
    #         ax.set_xlabel('$z_1$', fontsize=14)
    #         ax.set_ylabel('$z_2$', fontsize=14)
    #         ax.set_zlabel('{}'.format(dataset.continuous[j]), fontsize=14)
    #     ax.view_init(30, 60)
    #     plt.tight_layout()
    #     plt.savefig('./assets/{}/latent_quantile/latent_quantile_{}.png'.format(config["dataset"], j))
    #     # plt.show()
    #     plt.close()
    #     wandb.log({'latent space ~ quantile': wandb.Image(fig)})
    #%%
    # """latent space"""
    # latents = []
    # for (x_batch) in tqdm.tqdm(iter(dataloader)):
    #     if config["cuda"]:
    #         x_batch = x_batch.cuda()
        
    #     with torch.no_grad():
    #         mean, logvar = model.get_posterior(x_batch)
    #     latents.append(mean)
    # latents = torch.cat(latents, axis=0)
    
    # fig = plt.figure(figsize=(3, 3))
    # plt.scatter(latents[:, 0], latents[:, 1], 
    #             alpha=0.7, s=1)
    # plt.xlabel('$z_1$', fontsize=14)
    # plt.ylabel('$z_2$', fontsize=14)
    # plt.tight_layout()
    # plt.savefig('./assets/{}/latent.png'.format(config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'latent space': wandb.Image(fig)})
    #%%
    """Empirical quantile plot"""
    q = np.arange(0.01, 1, 0.01)
    fig, ax = plt.subplots(3, config["input_dim"] // 3, figsize=(config["input_dim"], 6))
    for k, v in enumerate(dataset.continuous):
        ax.flatten()[k].plot(q, np.quantile(dataset.x_data[:, k], q=q))
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.tight_layout()
    plt.savefig('./assets/{}/empirical_quantile.png'.format(config["dataset"]))
    # plt.show()
    plt.close()
    #%%
    """Quantile Estimation with sampling mechanism"""
    n = 100
    x_linspace = np.linspace(
        [np.quantile(dataset.x_data[:, k], q=0.01) for k in range(len(dataset.continuous))],
        [np.quantile(dataset.x_data[:, k], q=0.99) for k in range(len(dataset.continuous))],
        n)
    x_linspace = torch.from_numpy(x_linspace)
    
    alpha_hat = torch.zeros((n, len(dataset.continuous)))
    for _ in range(n):
        randn = torch.randn(n, config["latent_dim"]) # prior
        with torch.no_grad():
            gamma, beta = model.quantile_parameter(randn)
            alpha_tilde_list = model.quantile_inverse(x_linspace, gamma, beta)
            alpha_hat += torch.cat(alpha_tilde_list, dim=1)
    alpha_hat /= n
    #%%
    fig, ax = plt.subplots(3, config["input_dim"] // 3, figsize=(config["input_dim"], 6))
    for k, v in enumerate(dataset.continuous):
        ax.flatten()[k].plot(alpha_hat[:, k], x_linspace[:, k], label="sampled")
        ax.flatten()[k].plot(q, np.quantile(dataset.x_data[:, k], q=q), label="empirical")
        ax.flatten()[k].set_xlabel('alpha')
        ax.flatten()[k].set_ylabel(v)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/sampling_estimated_quantile.png'.format(config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'Estimated quantile (sampling mechanism)': wandb.Image(fig)})
    #%%
    # n = 100
    # q = np.arange(0.01, 0.99, 0.01)
    # j = 0
    # randn = torch.randn(n, config["latent_dim"]) # prior
    # fig, ax = plt.subplots(1, config["input_dim"], figsize=(2*config["input_dim"], 2))
    # for k, v in enumerate(dataset.continuous):
    #     quantiles = []
    #     for alpha in q:
    #         with torch.no_grad():
    #             gamma, beta = model.quantile_parameter(randn)
    #             quantiles.append(model.quantile_function(alpha, gamma, beta, k))
    #     ax.flatten()[k].plot(q, np.quantile(dataset.x_data[:, k], q=q), label="empirical")
    #     ax.flatten()[k].plot(q, [x.mean().item() for x in quantiles], label="prior")
    #     ax.flatten()[k].set_xlabel('alpha')
    #     ax.flatten()[k].set_ylabel(v)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/prior_estimated_quantile.png'.format(config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'Estimated quantile (prior)': wandb.Image(fig)})
    # #%%
    # n = 100
    # q = np.arange(0.01, 0.99, 0.01)
    # j = 0
    # idx = np.random.choice(range(len(latents)), n, replace=False)
    # fig, ax = plt.subplots(1, config["input_dim"], figsize=(2*config["input_dim"], 2))
    # for k, v in enumerate(dataset.continuous):
    #     quantiles = []
    #     for alpha in q:
    #         with torch.no_grad():
    #             gamma, beta = model.quantile_parameter(latents[idx, :]) # aggregated
    #             quantiles.append(model.quantile_function(alpha, gamma, beta, k))
    #     ax.flatten()[k].plot(q, np.quantile(dataset.x_data[:, k], q=q), label="empirical")
    #     ax.flatten()[k].plot(q, [x.mean().item() for x in quantiles], label="aggregated")
    #     ax.flatten()[k].set_xlabel('alpha')
    #     ax.flatten()[k].set_ylabel(v)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/aggregated_estimated_quantile.png'.format(config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'Estimated quantile (aggregated)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%