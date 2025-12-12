"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""

def select_2dim_nom(population,func_obj_1,func_obj_2,mu,sigma,pool_size):
    """
        nominal selection
        select the best individual among the population

        input:
            (list[list[float]]): population
            (func_obj -> float): func_obj_1, func_obj_2: function to MINIMIZE (input weights list)
            (numpy: array[float]): mu: rendement
            (numpy: array[array[float]]): sigma: matrice de covariance
            (int): pool_size: number of individual we want to select
            
        output:
            (list[list[float]]): selection

    """
    selection = []
    performances = []
    N = len(population)

    # we get the performances of each individual (distance from the origin) 
    for individual in population:
        x = float(func_obj_1(individual,mu))
        y = float(func_obj_2(individual,sigma))
        perf_idx = (x**2+y**2)**(1/2)
        performances.append(perf_idx)

    # we select the pool_size best individuals
    for i in range(pool_size):
        min = 1000
        for k in range(len(performances)):
            if performances[k]<=min:
                min = performances[k]
                i_min = k
        selection.append(population[i_min])
        performances.pop(i_min)
        population.pop(i_min)

    return selection
