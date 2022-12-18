#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy
#%%
def train(output_info_list, dataset, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
    }
    # for debugging
    logs['activated'] = []
    
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        mean, logvar, latent, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        start = 0
        recon = 0
        for column_info in output_info_list:
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':
                    end = start + span_info.dim
                    std = model.sigma[start]
                    residual = x_batch[:, start] - torch.tanh(xhat[:, start])
                    recon += (residual ** 2 / 2 / (std ** 2)).mean()
                    recon += torch.log(std)
                    start = end
                else:
                    end = start + span_info.dim
                    recon += cross_entropy(
                        xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim=-1), reduction='mean')
                    start = end
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        ### activated: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        loss_.append(('activated', (var_ < 0.1).sum()))
        
        loss = recon + KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        # model.sigma.data.clamp_(0.01, 0.1)
        model.sigma.data.clamp_(config["sigma_range"][0], config["sigma_range"][1])
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%