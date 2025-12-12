"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
from scripts.utils.const import DEFAULT_PROP_COST

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
    weight_diff = 0
    for i in range(n):
        weight_diff += (weights_t[i] - weights_t_p_1[i]) if weights_t[i] >= weights_t_p_1[i] else (weights_t_p_1[i] - weights_t[i])
    cost = prop_cost * weight_diff
    return cost
