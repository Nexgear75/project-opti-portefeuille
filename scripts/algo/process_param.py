"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
import pandas as pd
import numpy as np
from scripts.utils.const import DAY_TRADING


def process_param(df, day_trading=DAY_TRADING):
    """
    process mu & sigma from a given data frame    

    input: 
        (pandas: data frame): df
        (int): day_trading

    output:
        (numpy: array[float]): mu: rendement
        (numpy: array[array[float]]): sigma: matrice de covariance
        (int): N: number of values
    """
    
    # df.pct_change() : (price_t / price_{t-1}) - 1
    daily_yield = df.pct_change().dropna()
    
    # rendement
    # mu = mean(daily_yield) * day_trading
    mu = daily_yield.mean() * day_trading
    mu_vector = mu.to_numpy()
    
    # matrice de covariance
    # Sigma = covariance(daily_yield) * day_trading
    sigma = daily_yield.cov() * day_trading
    sigma_matrix = sigma.to_numpy()
    
    # number of values
    # N (variables de décision)
    N = len(df.columns) 
    
    print(f"Calcul des paramètres terminé pour N={N} actifs.")
    # print(f"Période des rendements: {daily_yield.index.min().date()} to {daily_yield.index.max().date()}")
    
    return mu_vector, sigma_matrix, N

""" test process_param
from get_data import get_df

df = get_df()
print(type(df))

mu, sigma, N = process_param(df)

print(f"mu\n {mu.round(4)}")
print(f"\nSigma (matrix {N}x{N}):\n {sigma.round(4)}")
"""
