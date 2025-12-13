"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime
    
"""
from scripts.algo.func_obj_1_yield import get_yield
from scripts.algo.func_obj_2_risk import get_risk
from scripts.algo.func_obj_3_trans_cost import get_trans_cost

def get_pareto(selection,mu,sigma,dim = 2):
    """
        process the pareto selection

        input:
            (list[list[float]]): selection: list of weights
            (numpy: array[float]): mu: rendement
            (numpy: array[array[float]]): sigma: matrice de covariance
            (int): dim: number of goal functions 2 / 3

        output:
            (list[list[float]]): pareto_selection: list of weights
    """
    pareto_selection = []
    pareto_selection.append(selection[0])
    if dim == 3:
        base_weights = []
        for i in range(len(selection[0])):
            base_weights.append(1/len(selection[0]))

    # each element of the selection
    for i in range(1,len(selection)):
        weights = selection[i]
        pareto_opt = True
        dominate = []

        # is comared wit all the pareto optimal elements
        for k in range(len(pareto_selection)):
            pareto_element = pareto_selection[k]
            f_1_w = get_yield(weights,mu)
            f_1_pareto = get_yield(pareto_element,mu)
            f_2_w = get_risk(weights,sigma)
            f_2_pareto = get_risk(pareto_element,sigma)

            if dim == 3:
                f_3_w = get_trans_cost(base_weights,weights)
                f_3_pareto = get_trans_cost(base_weights,pareto_element)

                if (f_3_pareto <= f_3_w and f_2_pareto <= f_2_w and f_1_pareto <= f_1_w):
                    # pareto element dominate our element
                    if (f_3_pareto < f_3_w or f_2_pareto < f_2_w or f_1_pareto < f_1_w):
                        pareto_opt = False
                        break

                if (f_3_pareto >= f_3_w and f_2_pareto >= f_2_w and f_1_pareto >= f_1_w):
                    # individual dom pareto element
                    if (f_3_pareto > f_3_w or f_2_pareto > f_2_w or f_1_pareto > f_1_w):
                        dominate.append(k)
            else :

                if (f_2_pareto <= f_2_w and f_1_pareto <= f_1_w):
                    # pareto element dominate our element
                    if (f_2_pareto < f_2_w or f_1_pareto < f_1_w):
                        pareto_opt = False 
                        break

                if (f_2_pareto >= f_2_w and f_1_pareto >= f_1_w):
                    # individual dom pareto element
                    if (f_2_pareto > f_2_w or f_1_pareto > f_1_w):
                        dominate.append(k)

        # all the dominated elements ar deleted from the pareto optimal selection
        for k in sorted(dominate, reverse = True):
            pareto_selection.pop(k)

        if pareto_opt:
            pareto_selection.append(weights)

    return pareto_selection
