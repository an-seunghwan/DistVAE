#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_VAE(OutputInfo_list, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'quantile': [],
        'KL': [],
    }
    # for debugging
    logs['activated'] = []
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        z, mean, logvar, gamma, beta, logit = model(x_batch)
        
        loss_ = []
        
        """alpha_tilde"""
        alpha_tilde_list = model.quantile_inverse(x_batch, gamma, beta)
        
        """loss"""
        j = 0
        st = 0
        total_loss = 0
        # tmp1 = []
        # tmp2 = []
        for j, info in enumerate(OutputInfo_list):
            if info.activation_fn == "CRPS":
                term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
                term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
                
                loss = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
                loss += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
                loss += (beta[j] * term).sum(axis=1, keepdims=True)
                loss *= 0.5
                total_loss += loss.mean()
                # tmp1.append(x_batch[:, [j]])
            
            elif info.activation_fn == "softmax":
                ed = st + info.dim
                _, targets = x_batch[:, config["CRPS_dim"] + st : config["CRPS_dim"] + ed].max(dim=1)
                out = logit[:, st : ed]
                # tmp1.append(x_batch[:, config["CRPS_dim"] + st : config["CRPS_dim"] + ed])
                # tmp2.append(out)
                total_loss += nn.CrossEntropyLoss()(out, targets)
                st = ed
        
        # assert (torch.cat(tmp1, dim=1) - x_batch).sum().item() == 0
        # assert (torch.cat(tmp2, dim=1) - logit).sum().item() == 0
                
        loss_.append(('quantile', total_loss))
        
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
        
        loss = total_loss + config["beta"] * KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
# alpha_tilde_list = []
# for j in range(config["input_dim"]):
#     delta_ = model.delta.unsqueeze(2).repeat(1, 1, model.M + 1)
#     delta_ = torch.where(delta_ - model.delta > 0,
#                         delta_ - model.delta,
#                         torch.zeros(()).to(device))
#     mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(axis=-1).squeeze(0).t()
#     # mask = [model.quantile_function(d, gamma, beta, j) for d in model.delta[0]]
#     # mask = torch.cat(mask, axis=1)
#     mask = torch.where(mask <= x_batch[:, [j]], 
#                     mask, 
#                     torch.zeros(()).to(device)).type(torch.bool).type(torch.float)
#     alpha_tilde = x_batch[:, [j]] - gamma[j]
#     alpha_tilde += (mask * beta[j] * model.delta).sum(axis=1, keepdims=True)
#     alpha_tilde /= (mask * beta[j]).sum(axis=1, keepdims=True) + 1e-6
#     alpha_tilde = torch.clip(alpha_tilde, config["threshold"], 1) # numerical stability
#     alpha_tilde_list.append(alpha_tilde)
#%%