#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_VAE(dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'quantile': [],
        'KL': [],
    }
    # for debugging
    for i in range(config["latent_dim"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        z, mean, logvar, gamma, beta = model(x_batch.tanh())
        
        loss_ = []
        
        """alpha_tilde"""
        j = 0
        alpha_tilde_list = []
        for j in range(config["input_dim"]):
            delta_ = model.delta.unsqueeze(2).repeat(1, 1, model.M + 1)
            delta_ = torch.where(delta_ - model.delta > 0,
                                delta_ - model.delta,
                                torch.zeros(()).to(device))
            mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(axis=-1).squeeze(0).t()
            # mask = [model.quantile_function(d, gamma, beta, j) for d in model.delta[0]]
            # mask = torch.cat(mask, axis=1)
            mask = torch.where(mask <= x_batch[:, [j]], 
                            mask, 
                            torch.zeros(()).to(device)).type(torch.bool).type(torch.float)
            alpha_tilde = x_batch[:, [j]] - gamma[j]
            alpha_tilde += (mask * beta[j] * model.delta).sum(axis=1, keepdims=True)
            alpha_tilde /= (mask * beta[j]).sum(axis=1, keepdims=True) + 1e-6
            alpha_tilde = torch.clip(alpha_tilde, config["threshold"], 1) # numerical stability
            alpha_tilde_list.append(alpha_tilde)
        
        """loss"""
        j = 0
        total_loss = 0
        for j in range(config["input_dim"]):
            term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
            term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
            
            loss = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
            loss += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
            loss += (beta[j] * term).sum(axis=1, keepdims=True)
            loss *= 0.5
            total_loss += loss.mean()
        # print(loss.mean())
        loss_.append(('quantile', total_loss))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        ### posterior variance: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        for i in range(config["latent_dim"]):
            loss_.append(('posterior_variance{}'.format(i+1), var_[i]))
        
        loss = total_loss + config["beta"] * KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%