#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from statsmodels.distributions.empirical_distribution import ECDF
import torch
from modules.toy_NCECopula import NCECopula
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    
    parser.add_argument('--grid_points', type=int, default=21, 
                        help='Number of points to approximate conditional distribution in Gibbs sampling')
    parser.add_argument('--data_dim', default=2,
                        help='d-dimension')
    
    parser.add_argument('--batch_size', default=256,
                        help='Number of data samples to train on at once')
    parser.add_argument('--iterations', default=1000,
                        help='Number of iterations to train for')
    parser.add_argument('--lr', default=0.005,
                        help='learning rate')
  
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """data generation"""
    n = 10000
    sigma = 0.1
    t = torch.randn(n, 1)
    x = torch.cat(
        [torch.sin(t) + torch.randn(n, 1)*sigma,
        t * torch.cos(t) + torch.randn(n, 1)*sigma],
        dim=1
    )

    ecdf = [ECDF(x[:, 0]), ECDF(x[:, 1])]
    pseudo = torch.cat(
        [torch.from_numpy(ecdf[0](x[:, 0])[:, None]), 
        torch.from_numpy(ecdf[1](x[:, 1])[:, None])],
        dim=1)
    pseudo = pseudo.to(torch.float32)

    assert config["data_dim"] == pseudo.size(1)
    #%%
    """Copula with NCE"""
    copula = NCECopula(config, device)
    copula.train(pseudo)
    uv_samples = copula.gibbs_sampling(test_size=n)
    #%%
    """visualization"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(pseudo[:, 0], pseudo[:, 1],
                alpha=0.5)
    ax[0].set_title('Ground-Truth', fontsize=15)
    ax[1].scatter(uv_samples[:, 0], uv_samples[:, 1],
                alpha=0.5)
    ax[1].set_title('NCE', fontsize=15)
    plt.tight_layout()
    plt.savefig('./assets/toy_sampled.png')
    # plt.show()
    plt.close()
    #%%
    import seaborn as sns
    df_samples = pd.DataFrame(
        uv_samples, columns=['U1', 'U2']
    )
    sns.jointplot(
        x="U1", y="U2",
        alpha=0.5,
        data=df_samples,
        height=5);
    plt.tight_layout()
    plt.savefig('./assets/toy_joint.png')
    # plt.show()
    plt.close()
#%%
if __name__ == '__main__':
    main()
#%%