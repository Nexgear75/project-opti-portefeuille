"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
from scripts.utils.const import DEFAULT_PROP_COST
import numpy as np

def get_trans_cost(weights_t, weights_t_p_1, prop_cost = DEFAULT_PROP_COST, trace = False):
    """
        process the transaction cost
        
        input:
            (list[float]): weights_t: previous weights repartion
            (list[float]): weights_t_p_1: next increment weights repartion
            float: prop_cost

        output:
            (float): cost
    """
    n = len(weights_t)
    if n != len(weights_t_p_1):
        if trace:
            print("unable to process the transaction_cost\n    The length of both weight list must be the same.")
            return []
    weights_t = np.asarray(weights_t, np.float32)
    weights_t_p_1 = np.asarray(weights_t_p_1, np.float32)
    weight_diff = np.sum(np.abs(weights_t_p_1 - weights_t))
    cost = prop_cost * weight_diff
    return cost
