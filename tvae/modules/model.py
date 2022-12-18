#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class TVAE(nn.Module):
    def __init__(self, config, device):
        super(TVAE, self).__init__()
        
        self.config = config
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["input_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, config["node"] * 2),
        ).to(device)
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"], 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, config["input_dim"]),
        ).to(device)
        self.sigma = nn.Parameter(torch.ones(config["input_dim"]) * 0.1)
        
    def get_posterior(self, input):
        h = self.encoder(nn.Flatten()(input)) # [batch, node * 2]
        mean, logvar = torch.split(h, self.config["node"], dim=1)
        return mean, logvar
    
    def encode(self, input, deterministic=False):
        mean, logvar = self.get_posterior(input)
        
        """Latent Generating Process"""
        if deterministic:
            latent = mean
        else:
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            latent = mean + torch.exp(logvar / 2) * noise
        return mean, logvar, latent
    
    def forward(self, input, deterministic=False):
        """encoding"""
        mean, logvar, latent = self.encode(input, deterministic=deterministic)
        
        """decoding"""
        xhat = self.decoder(latent)
        
        return mean, logvar, latent, xhat
#%%
def main():
    #%%
    config = {
        "input_dim": 5,
        "n": 10,
        "node": 3,
    }
    """TVAE"""
    model = TVAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["input_dim"])
    
    mean, logvar, latent, xhat = model(batch)
    
    assert mean.shape == (config["n"], config["node"])
    assert logvar.shape == (config["n"], config["node"])
    assert latent.shape == (config["n"], config["node"])
    assert xhat.shape == (config["n"], config["input_dim"])
    
    print("TVAE pass test!")
    print()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%