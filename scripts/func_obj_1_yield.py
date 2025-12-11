"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
import numpy as np

def get_yield(weights, mu):
    """
        process the yield
        fuction 1 minimize de negative yield (f1(w) = -w^T * mu)

        input:
            (list[float]): weights
            (numpy: array[float]): mu
        output:
            (list[float]): f1(w)
    """
    return -np.dot(weights,mu)
