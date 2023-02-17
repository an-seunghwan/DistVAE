#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
#%%
class CopulaCDF(nn.Module):
    def __init__(self, config):
        super(CopulaCDF, self).__init__()
        
        self.config = config
        
        """spline"""
        self.delta = torch.arange(0, 1 + config["step"], step=config["step"]).view(1, -1)
        self.M = self.delta.size(1) - 1
        
        self.spline1 = nn.Parameter(
            torch.randn(1, 1 + (self.M + 1)),
            requires_grad=True)
        self.spline2 = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 1 + (self.M + 1)),
        )
    
    def quantile_parameter(self, u):
        h = [self.spline1,
             self.spline2(u)]
        gamma = [h_[:, [0]] for h_ in h]
        beta = [nn.Softplus()(h_[:, 1:]) for h_ in h] # positive constraint
        return gamma, beta
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(alpha - self.delta > 0,
                                                alpha - self.delta,
                                                torch.zeros(()))).sum(axis=1, keepdims=True)
        
    def _quantile_inverse(self, x, gamma, beta, j):
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(delta_ - self.delta > 0,
                            delta_ - self.delta,
                            torch.zeros(()))
        mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(axis=-1).squeeze(0).t()
        mask = torch.where(mask <= x, 
                        mask, 
                        torch.zeros(())).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma[j]
        alpha_tilde += (mask * beta[j] * self.delta).sum(axis=1, keepdims=True)
        alpha_tilde /= (mask * beta[j]).sum(axis=1, keepdims=True) + 1e-6
        alpha_tilde = torch.clip(alpha_tilde, self.config["threshold"], 1) # numerical stability
        return alpha_tilde

    def quantile_inverse(self, x, gamma, beta):
        alpha_tilde_list = []
        for j in range(x.size(1)):
            alpha_tilde = self._quantile_inverse(x[:, [j]], gamma, beta, j)
            alpha_tilde_list.append(alpha_tilde)
        return alpha_tilde_list
    
    def forward(self, u):
        gamma, beta = self.quantile_parameter(u)
        return gamma, beta
#%%