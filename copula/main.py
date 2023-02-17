#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from statsmodels.distributions.empirical_distribution import ECDF
#%%
np.random.seed(1)

n = 1000
sigma = 0.1
t = np.random.normal(size=(n, 1))
x = np.concatenate(
    [np.sin(t) + np.random.normal(scale=sigma, size=(n, 1)),
    t * np.cos(t) + np.random.normal(scale=sigma, size=(n, 1))],
    axis=1
)
#%%
ecdf = [ECDF(x[:, 0]), ECDF(x[:, 1])]
pseudo = np.concatenate(
    [ecdf[0](x[:, 0])[:, None], 
     ecdf[1](x[:, 1])[:, None]],
    axis=1)
#%%
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1])
#%%
plt.figure(figsize=(5, 5))
plt.scatter(pseudo[:, 0], pseudo[:, 1])
#%%

#%%