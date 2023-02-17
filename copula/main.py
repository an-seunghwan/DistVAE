#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from statsmodels.distributions.empirical_distribution import ECDF

from modules.model import CopulaCDF
from modules.train import train_function
#%%
torch.random.manual_seed(1)

n = 1000
sigma = 0.1
t = torch.randn(n, 1)
x = torch.cat(
    [torch.sin(t) + torch.randn(n, 1)*sigma,
    t * torch.cos(t) + torch.randn(n, 1)*sigma],
    dim=1
)
#%%
ecdf = [ECDF(x[:, 0]), ECDF(x[:, 1])]
pseudo = torch.cat(
    [torch.from_numpy(ecdf[0](x[:, 0])[:, None]), 
     torch.from_numpy(ecdf[1](x[:, 1])[:, None])],
    dim=1)
pseudo = pseudo.to(torch.float32)
#%%
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1])
#%%
plt.figure(figsize=(5, 5))
plt.scatter(pseudo[:, 0], pseudo[:, 1])
#%%
config = {
    "epochs": 1000,
    "step": 0.005,
    "lr": 0.1,
    "threshold": 1e-5
}
#%%
model = CopulaCDF(config)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config["lr"]
)
#%%
for epoch in range(config["epochs"]):
    loss = train_function(pseudo, model, config, optimizer)
    
    if epoch % 50 == 0:
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', Loss: {:.4f}'.format(loss)])
        print(print_input)
#%%
with torch.no_grad():
    alpha = torch.rand(1000, 1)
    h = [model.spline1]
    gamma = [h_[:, [0]] for h_ in h]
    beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
    u1 = model.quantile_function(alpha, gamma, beta, 0)
    
    alpha = torch.rand(1000, 1)
    h = [model.spline2(u1)]
    gamma = [h_[:, [0]] for h_ in h]
    beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
    u2 = model.quantile_function(alpha, gamma, beta, 0)
#%%
plt.scatter(u1, u2)
#%%