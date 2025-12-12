"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""

from scripts.algo.genetic.initiate import initiate
from scripts.algo.genetic.select import select_2dim_nom
from scripts.algo.genetic.reproduce import reproduce
from scripts.algo.func_obj_1_yield import get_yield
from scripts.algo.func_obj_2_risk import get_risk
from scripts.utils.const import DEFAULT_POP_SIZE, DEFAULT_MAX_ITER

def gen_alg_2dim(mu,sigma,N,population_size = DEFAULT_POP_SIZE,selection_size = 20,max_iteration = DEFAULT_MAX_ITER,selection = "nom",trace = False):
    """
        genetic algo to generate some weights

        input:
            (numpy: array[float]): mu
            (numpy: array[array[float]]): sigma
            (int): N
            (int): population_size: for population generation and reproduction
            (int): selection_size: for selection
            (int): max_iteration: in case de selection does not reach a stabilization
            (String): selection: norm: nominal selection
            (Bool): trace

        output:
            (list[list[float]]): selection: the last selection (sorted best -> worst)
    """

    if trace:
        print("population initiate")
    population = initiate(N)
    selection_nm2 = []
    selection_nm1 = []
    stagnate = False
    i = 0
    while stagnate == False and i < max_iteration:
        i += 1
        if trace:
            print("iteration -",i)
            print("selection - ",end="")
        selection = select_2dim_nom(population,get_yield,get_risk,mu,sigma,10)
        if trace:
            print(len(selection),"-",len(selection[0]))
        if selection == selection_nm1 and selection_nm1 == selection_nm2:
            stagnate = True
            break
        else:
            selection_nm2 = selection_nm1
            selection_nm1 = selection
        if trace:
            print("reproduction & mutation & replacement - ",end="")
        population = reproduce(selection,N)
        if trace:
            print(len(population),"-",len(population[0]))
    return selection
