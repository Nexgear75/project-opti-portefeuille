"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
from random import randint, shuffle
from scripts.utils.const import DEFAULT_POP_SIZE

def initiate(N,pool_size = DEFAULT_POP_SIZE):
    """
        initiate a population of solution 

        input:
            (int): N: Size of the weights list
            (int): pool_size: size of the batch of solution
    
        output:
            (list[list[float]]): population: list of weight list
    """

    population = []
    for i in range (pool_size):
        population.append(gen_individual(N))

    return population

def gen_individual(N):
    """
        generate a random individual

        input:
            (int): N: Size of the weights list
        
        output:
            (list[float]): individual: weights list
    """
    individual = []
    for i in range(N):
        individual.append(0)
    for i in range(N):
        if sum(individual)<1:
            individual[i] = randint(0,1000)/1000
        k = 0    
        while sum(individual)>1:
            if k >= 50:
                individual[i] = 0
                individual[i] = 1 - sum(individual)
            else:
                individual[i] = randint(0,1000)/1000
                k+=1
    if sum(individual) != 1:
        individual[randint(0,N-1)] += 1 - sum(individual)
    shuffle(individual)
    return individual

