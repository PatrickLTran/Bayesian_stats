#%%
import tensorflow as tf
import pandas as pd
import numpy as np


#%%

df = pd.read_csv('/home/docker/stats_rethinking/linear_regression/Howell1.csv',sep=';')
df_18 = df[df['age'] >= 18]

#%%
# ####
"""

y = a + b(h_i - h_)

"""
# %%
h_bar = np.mean(df_18.height)

#%%

alpa = np.random.normal(loc=60,scale=10,size=5)