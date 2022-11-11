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
            nn.Linear(8, config["node"] * 2),
        ).to(device)
        
        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["node"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 3*config["image_size"]*config["image_size"]),
            nn.Tanh()
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(nn.Flatten()(input)) # [batch, node * 2]
        mean, logvar = torch.split(h, self.config["node"], dim=1)
        return mean, logvar
    
    def transform(self, input, log_determinant=False):
        latent = torch.matmul(input, self.I_B_inv) # [batch, node], input = epsilon (exogenous variables)
        orig_latent = latent.clone()
        latent = torch.split(latent, 1, dim=1) # [batch, 1] x node
        latent = list(map(lambda x, layer: layer(x, log_determinant=log_determinant), latent, self.flows)) # input = (I-B^T)^{-1} * epsilon
        logdet = [x[1] for x in latent]
        latent = [x[0] for x in latent]
        return orig_latent, latent, logdet
    
    def encode(self, input, deterministic=False, log_determinant=False):
        mean, logvar = self.get_posterior(input)
        
        """Latent Generating Process"""
        if deterministic:
            epsilon = mean
        else:
            noise = torch.randn(input.size(0), self.config["node"]).to(self.device) 
            epsilon = mean + torch.exp(logvar / 2) * noise
        orig_latent, latent, logdet = self.transform(epsilon, log_determinant=log_determinant)
        
        return mean, logvar, epsilon, orig_latent, latent, logdet
    
    def forward(self, input, deterministic=False, log_determinant=False):
        """encoding"""
        mean, logvar, epsilon, orig_latent, latent, logdet = self.encode(input, 
                                                                         deterministic=deterministic,
                                                                         log_determinant=log_determinant)
        
        """decoding"""
        xhat = self.decoder(torch.cat(latent, dim=1))
        xhat = xhat.view(-1, self.config["image_size"], self.config["image_size"], 3)
        
        """Alignment"""
        _, _, _, _, align_latent, _ = self.encode(input, 
                                                  deterministic=True, 
                                                  log_determinant=log_determinant)
        
        return mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat
#%%
class Discriminator(nn.Module):
    def __init__(self, config, device='cpu'):
        super(Discriminator, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(3*config["image_size"]*config["image_size"] + config["node"], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1),
        ).to(device)

    def forward(self, x, z):
        x = x.view(-1, 3*self.config["image_size"]*self.config["image_size"])
        x = torch.cat((x, z), dim=1)
        return self.net(x)
#%%
def main():
    #%%
    config = {
        "image_size": 64,
        "n": 10,
        "node": 4,
        "flow_num": 4,
        "inverse_loop": 100,
        "scm": 'linear',
        "factor": [1, 1, 2],
    }
    
    B = torch.zeros(config["node"], config["node"])
    B[:2, 2:] = 1
    #%%
    """CAD-VAE"""
    mask = []
    # light
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[:20, ...] = 1
    mask.append(m)
    # angle
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[20:51, ...] = 1
    mask.append(m)
    # shadow
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    
    model = GAM(B, mask, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(batch)
    
    inverse_diff = torch.abs(sum([x - y for x, y in zip(torch.split(orig_latent, 1, dim=1), 
                                                        model.inverse(latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"]) < 1e-5
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(batch)
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat_separated, xhat = model(batch, log_determinant=True)
    
    assert mean.shape == (config["n"], config["node"])
    assert logvar.shape == (config["n"], config["node"])
    assert epsilon.shape == (config["n"], config["node"])
    assert orig_latent.shape == (config["n"], config["node"])
    assert latent[0].shape == (config["n"], 1)
    assert len(latent) == config["node"]
    assert logdet[0].shape == (config["n"], 1)
    assert len(logdet) == config["node"]
    assert align_latent[0].shape == (config["n"], 1)
    assert len(align_latent) == config["node"]
    assert xhat.shape == (config["n"], config["image_size"], config["image_size"], 3)
    
    # deterministic behavior
    out1 = model(batch, deterministic=False)
    out2 = model(batch, deterministic=True)
    assert (out1[0] - out2[0]).abs().mean() == 0 # mean
    assert (out1[1] - out2[1]).abs().mean() == 0 # logvar
    assert (torch.cat(out1[4], dim=1) - torch.cat(out2[4], dim=1)).abs().mean() != 0 # latent
    assert (torch.cat(out1[6], dim=1) - torch.cat(out2[6], dim=1)).abs().mean() == 0 # align_latent
    
    print("CAD-VAE pass test!")
    print()
    #%%
    """Baseline VAE"""
    model = VAE(B, config, 'cpu')
    discriminator = Discriminator(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    for x in discriminator.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(batch)
    inverse_diff = torch.abs(sum([x - y for x, y in zip(torch.split(orig_latent, 1, dim=1), 
                                                        model.inverse(latent))]).sum())
    assert inverse_diff / (config["n"] * config["node"]) < 1e-5
    
    info = discriminator(batch, epsilon)
    assert info.shape == (config["n"], 1)
    
    mean, logvar, epsilon, orig_latent, latent, logdet, align_latent, xhat = model(batch, log_determinant=True)
    
    assert mean.shape == (config["n"], config["node"])
    assert logvar.shape == (config["n"], config["node"])
    assert epsilon.shape == (config["n"], config["node"])
    assert orig_latent.shape == (config["n"], config["node"])
    assert latent[0].shape == (config["n"], 1)
    assert len(latent) == config["node"]
    assert logdet[0].shape == (config["n"], 1)
    assert len(logdet) == config["node"]
    assert align_latent[0].shape == (config["n"], 1)
    assert len(align_latent) == config["node"]
    assert xhat.shape == (config["n"], config["image_size"], config["image_size"], 3)
    
    # deterministic behavior
    out1 = model(batch, deterministic=False)
    out2 = model(batch, deterministic=True)
    assert (out1[0] - out2[0]).abs().mean() == 0 # mean
    assert (out1[1] - out2[1]).abs().mean() == 0 # logvar
    assert (torch.cat(out1[4], dim=1) - torch.cat(out2[4], dim=1)).abs().mean() != 0 # latent
    assert (torch.cat(out1[6], dim=1) - torch.cat(out2[6], dim=1)).abs().mean() == 0 # align_latent
    
    print("Baseline VAE pass test!")
    print()
    #%%
    """Baseline Classifier"""
    mask = []
    # light
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[:20, ...] = 1
    mask.append(m)
    # angle
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[20:51, ...] = 1
    mask.append(m)
    # shadow
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    m = torch.zeros(config["image_size"], config["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    
    model = Classifier(mask, config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["image_size"], config["image_size"], 3)
    
    pred = model(batch)
    
    assert pred.shape == (config["n"], config["node"])
    
    print("Baseline Classifier pass test!")
    print()
    #%%
    """Downstream Classifier"""
    model = DownstreamClassifier(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(config["n"], config["node"])
    
    pred = model(batch)
    
    assert pred.shape == (config["n"], 1)
    
    print("Downstream Classifier pass test!")
#%%
if __name__ == '__main__':
    main()
#%%