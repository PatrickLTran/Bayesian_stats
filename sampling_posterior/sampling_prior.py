#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
# %%
# grid estimation functions
# grid points is number of samples you want to use to approximate
def uniform_prior(grid_points):
    """
    Return  uniform prior distribution 
    parameters:
        grid pints: array of prior values
    Return:
        Uniform density of prior values
    """
    
    return np.repeat(5, grid_points)
#%%
def truncated_prior(grid_points, trunc_point=0.5):
    """
    Returns Truncated prior density

            Parameters:
                grid_points (numpy.array): Array of prior values
                trunc_point (double): Value where the prior is truncated

            Returns:
                density (numpy.array): Truncated density of prior values
    """
    return (np.linspace(0, 1, grid_points) >= trunc_point).astype(int)


def double_exp_prior(grid_points):
    """
    Returns Double Exponential prior density

            Parameters:
                grid_points (numpy.array): Array of prior values

            Returns:
                density (numpy.array): Double Exponential density of prior values
    """
    return np.exp(-5 * abs(np.linspace(0, 1, grid_points) - 0.5))

def binom_post_grid_approx(prior_func, grid_points=5, success=6, tosses=9):
    """
    Returns the grid approximation of posterior distribution with binomial likelihood.

            Parameters:
                    prior_func (function): A function that returns the likelihood of the prior
                    grid_points (int): Number of points in the prior grid
                    successes (int): Number of successes
                    tosses (int): number of tosses

            Returns:
                    p_grid (numpy.array): Array of prior values
                    posterior (numpy.array): Likelihood (density) of prior values
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = prior_func(grid_points)

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior
# %%
p_grid, posterior = binom_post_grid_approx(uniform_prior, grid_points=100, success=6, tosses=9)
samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)
# %%
_, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.plot(samples, "o", alpha=0.2)
ax0.set_xlabel("sample number")
ax0.set_ylabel("proportion water (p)")
az.plot_kde(samples, ax=ax1)
ax1.set_xlabel("proportion water (p)")
ax1.set_ylabel("density");
# %%
# get highest density above 50%
az.hdi(samples, hdi_prob=0.5)
# %%
dummy_w = stats.binom.rvs(n=9, p=0.7, size=int(1e5))
# dummy_w = stats.binom.rvs(n=9, p=0.6, size=int(1e4))
# dummy_w = stats.binom.rvs(n=9, p=samples)
bar_width = 0.1
plt.hist(dummy_w, bins=np.arange(0, 11) - bar_width / 2, width=bar_width)
plt.xlim(0, 9.5)
plt.xlabel("dummy water count")
plt.ylabel("Frequency");
# %%

###########################################################

"""
Assume globe toss example with observations
W L W W W L W L W
"""
obs = np.array([1,0,1,1,1,0,1,0,1])

with pm.Model() as sample_model:
    p = pm.Uniform("p")
    y = pm.Bernoulli("y", p=p, observed=obs)
    trace = pm.sample(1000,chains=4)
    az.plot_trace(trace)
    az.plot_forest(trace, var_names = ["p"])
# %%
samples = trace["posterior"]['p'].values
# %%
"""
percentage of sample below 20% quantile
"""
np.sum(samples < 0.2) / samples.size
# %%
"""
percentage of sample above 80% quantile
"""
np.sum(samples > 0.8) / samples.size
#%%
"""
percentage of sample between 20% and 80% quantile
"""
np.sum((samples > 0.2 ) & (samples < 0.8)) / samples.size

# %%
"""
p
"""
hpd_interval = az.hdi(samples, hdi_prob=0.66)

# %%
print(
    str(
        1
        - np.sum(samples > hpd_interval[1]) / samples.size
        - np.sum(samples < hpd_interval[0]) / samples.size
    )
)

# %%
