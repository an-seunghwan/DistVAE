#%%
import torch
#%%
def train_function(pseudo, model, config, optimizer):
    optimizer.zero_grad()
    
    gamma, beta = model(pseudo[:, [0]])

    """alpha_tilde"""
    alpha_tilde_list = model.quantile_inverse(pseudo, gamma, beta)
    
    """loss"""
    j = 0
    total_loss = 0
    for j in range(pseudo.size(1)):
        term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
        term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
        
        loss = (2 * alpha_tilde_list[j]) * pseudo[:, [j]]
        loss += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
        loss += (beta[j] * term).sum(axis=1, keepdims=True)
        loss *= 0.5
        total_loss += loss.mean()
            
    total_loss.backward()
    optimizer.step()
    
    return total_loss
#%%