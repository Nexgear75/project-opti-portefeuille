"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
import numpy as np

def get_risk(weights,sigma):
    """
        process risk
        function 2 minimize the risk (f2(w) = w^T * sigma * w) 

        input:
            (list[float]): weights
            (numpy: array[array[float]]): sigma

        output:
            (list[float]): f2(w)
    """
    return np.dot(weights.T,np.dot(sigma,weights))
