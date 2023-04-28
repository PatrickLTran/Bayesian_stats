#%%
#import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import norm
import pymc

#%%

df = pd.read_csv('/home/docker/Bayesian_stats-main/linear_regression/Howell1.csv',sep=';')
df_18 = df[df['age'] >= 18]
h_bar = np.mean(df_18.height)

#%%
# ####
"""

mu = a + b(h_i - h_bar)

"""
#%%
h = df_18.height.values
with pymc.Model() as model:
    a = pymc.Normal('intercept', mu=60,sigma=10)
    b = pymc.Lognormal('slope', mu=0, sigma=1)
    mu = a + b*(h-h_bar)
    sigma = pymc.HalfNormal('Sigma', sigma=10)
    y_obs = df_18.weight.values

with model:
    trace = pymc.sample(1000,tune=1000)

print(pymc.summary(trace, var_names=['intercept', 'slope']))

# %%
