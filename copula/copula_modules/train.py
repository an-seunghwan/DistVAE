#%%
import tqdm

import torch
from torch import nn
#%%
def train_copula(OutputInfo_list, dataloader, model, copula, config):
    logs = {
        'loss': [], 
    }
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        copula.optimizer.zero_grad()
        
        with torch.no_grad():
            """pseudo-observations"""
            z, mean, logvar, gamma, beta, logit = model(x_batch, deterministic=False)
            # continuous
            alpha_tilde_list = model.quantile_inverse(x_batch, gamma, beta)
            cont_pseudo = torch.cat(alpha_tilde_list, dim=1)
            # discrete
            disc_pseudo = []
            st = 0
            for j, info in enumerate(OutputInfo_list):
                if info.activation_fn == "CRPS":
                    continue
                elif info.activation_fn == "softmax":
                    ed = st + info.dim
                    out = logit[:, st : ed]
                    cdf = nn.Softmax(dim=1)(out).cumsum(dim=1)
                    x_ = x_batch[:, config["CRPS_dim"] + st : config["CRPS_dim"] + ed]
                    disc_pseudo.append((cdf * x_).sum(axis=1, keepdims=True))
                    st = ed
            disc_pseudo = torch.cat(disc_pseudo, dim=1)
            # all covariates
            pseudo = torch.cat([cont_pseudo, disc_pseudo], dim=1)
        
        # Noise contrastive estimation
        uniform = torch.rand(
            pseudo.size(0), config["data_dim"]) # noise
        
        true = copula.model(
            torch.cat([pseudo, z], dim=1))
        noise = copula.model(
            torch.cat([uniform, z], dim=1))
        
        loss = - (torch.log(true + 1e-6) + torch.log(1 - noise + 1e-6)).mean() # numerical stability
            
        loss.backward()
        copula.optimizer.step()
            
        """accumulate losses"""
        logs['loss'] = logs.get('loss') + [loss.item()]
    
    return logs
#%%