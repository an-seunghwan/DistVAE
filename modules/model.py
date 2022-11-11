#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["input_dim"], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, config["latent_dim"] * 2),
        ).to(device)
        
        """spline"""
        # self.M = 10
        self.delta = torch.arange(0, 1.1, step=0.1).view(1, -1).to(device)
        self.M = self.delta.size(1)
        self.spline = nn.Sequential(
            nn.Linear(config["latent_dim"], 4),
            nn.ReLU(),
            nn.Linear(4, config["input_dim"] * (1 + self.M)),
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar, deterministic=False):
        if deterministic:
            z = mean
        else:
            noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
            z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input, deterministic=False):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar, deterministic=deterministic)
        return z, mean, logvar
    
    def quantile_parameter(self, z):
        h = self.spline(z)
        h = torch.split(h, 1 + self.M, dim=1)
        
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
        return gamma, beta
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(alpha.to(self.device) - self.delta > 0,
                                                alpha.to(self.device) - self.delta,
                                                torch.zeros(()).to(self.device))).sum(axis=1, keepdims=True)
    
    def forward(self, input, deterministic=False):
        z, mean, logvar = self.encode(input, deterministic=deterministic)
        gamma, beta = self.quantile_parameter(z)
        return z, mean, logvar, gamma, beta
#%%
def main():
    #%%
    config = {
        "input_dim": 10,
        "latent_dim": 2,
    }
    
    model = VAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(10, config["input_dim"])
    
    z, mean, logvar, gamma, beta = model(batch)
    
    assert z.shape == (10, config["latent_dim"])
    assert mean.shape == (10, config["latent_dim"])
    assert logvar.shape == (10, config["latent_dim"])
    assert gamma[0].shape == (10, 1)
    assert len(gamma) == config["input_dim"]
    assert beta[0].shape == (10, model.M)
    assert len(beta) == config["input_dim"]
    
    print("Model pass test!")
#%%
if __name__ == '__main__':
    main()
#%%