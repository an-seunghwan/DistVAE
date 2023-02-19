#%%
"""
Reference:
[1] https://github.com/tonellolab/CODINE-copula-estimator/blob/main/CODINE_Gaussian.py
"""
#%%
import tqdm
import numpy as np
import scipy.interpolate as interpolate

import torch
from torch import nn
#%%
class NCECopula():
    def __init__(self, config, device):
        self.config = config
        
        """model"""
        self.model = nn.Sequential(
            nn.Linear(config["data_dim"], 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 1),
            nn.Sigmoid()
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config["lr"]
        )
        
    def train(self, data, verbose=True):
        self.model.train()
        
        print(self.model)
        
        """training: Noise contrastive estimation"""
        for i in range(self.config["iterations"]):
            self.optimizer.zero_grad()

            idx = np.random.choice(
                range(len(data)), self.config["batch_size"], replace=False)
            uniform = torch.rand(
                self.config["batch_size"], self.config["data_dim"])

            true = self.model(data[idx, :])
            noise = self.model(uniform)

            loss = - (true.log() + (1 - noise).log()).mean()

            loss.backward()
            self.optimizer.step()
            
            if verbose:
                if i % 100 == 0:
                    print_input = "[Iter {:03d}]".format(i + 1)
                    print_input += ''.join([', Loss: {:.4f}'.format(loss)])
                    print(print_input)
                    
    def inverse_transform_sampling(self, hist, bin_edges):
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist)
        inv_cdf = interpolate.interp1d(cum_values, bin_edges)
        return inv_cdf
    
    """Gibbs sampling"""
    def gibbs_sampling(self, test_size):
        self.model.eval()

        a = np.linspace(0, 1, self.config["grid_points"])
        uv_samples = np.zeros((test_size, self.config["data_dim"]))
        uv_samples[0, :] = np.random.uniform(0, 1, self.config["data_dim"]) # random initialization
        
        for t in tqdm.tqdm(range(1, test_size), desc="Gibbs Sampling..."):
            for i in range(self.config["data_dim"]):
                if i == 0:
                    # (t-1)th sample -> coordinates 1, ..., d-1
                    # (t)th sample -> coordinate 0 : grid point approximate
                    uv_i_vector = np.concatenate(
                        (a.reshape(-1, 1), np.repeat(
                            uv_samples[t-1, i+1:self.config["data_dim"]],
                            repeats=self.config["grid_points"], axis=0).reshape(-1, 1)),axis=1)
                    
                elif i > 0 and i < self.config["data_dim"]-1:
                    # (t-1)th sample -> coordinates k+1, ..., d-1 where k > 0
                    # (t)th sample -> coordinate 0, ..., k-1
                    # (t)th sample -> coordinate k : grid point approximate
                    uv_i_vector_left = np.concatenate(
                        (np.repeat(
                            uv_samples[t, 0:i],
                            repeats=self.config["grid_points"], axis=0).reshape(-1, 1),
                        a.reshape(-1, 1)), axis=1)
                    uv_i_vector = np.concatenate(
                        (uv_i_vector_left, 
                        np.repeat(
                            uv_samples[t-1, i+1:self.config["data_dim"]],
                            repeats=self.config["grid_points"], axis=0).reshape(-1, 1)), axis=1)
                    
                else:
                    # (t-1)th sample -> coordinates 0, 1, ..., d-2
                    # (t)th sample -> coordinate d-1 : grid point approximate
                    uv_i_vector = np.concatenate(
                        (np.repeat(
                            uv_samples[t, 0:i],
                            repeats=self.config["grid_points"], axis=0).reshape(-1, 1),
                        a.reshape(-1, 1)), axis=1)
                
                with torch.no_grad():
                    h = self.model(torch.from_numpy(uv_i_vector).to(torch.float32))
                conditional_density = h / (1 - h)
                conditional_density /= conditional_density.sum()
                
                icdf = self.inverse_transform_sampling(
                    conditional_density.numpy().squeeze(1),
                    np.linspace(0, 1, self.config["grid_points"] + 1))
                uv_samples[t, i] = icdf(np.random.uniform())

        return uv_samples
#%%