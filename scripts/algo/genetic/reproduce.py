"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime

"""
from random import randint
from scripts.utils.const import DEFAULT_POP_SIZE

def reproduce(selection,N,pool_size = DEFAULT_POP_SIZE,trace = False):
    """
        generate an all new population out of the selected sample
        we conservate both parents and children in the new population of size N
        we integrate a mutation factor in the reproduction
        we do a partial replacement (parent are kept in the n+1 population)
        
        input:
            (list[list[float]]): selection
            (int): N
        
        output:
            (list[list[float]]): population
    """
    population = selection
    len(selection)
    while(len(population ) < pool_size):
        if trace:
            print("generate individual n ",len(population)+1)
        parent_A = selection[randint(0,len(selection)-1)]
        parent_B = selection[randint(0,len(selection)-1)]
        individual = []
        # we have two reproduction way, distance wise or mean wise 
        if randint(0,1)==0:
            # distance wise high trait reproduction
            if trace : 
                print("reproduce in distance")
            for i in range(N):
                x = parent_A[i]
                y = parent_B[i]
                individual.append((x**2+y**2)**(1/2))
        else :
            # mean wise low trait reproduction
            if trace :
                print("reproduce by mean")
            for i in range(N):
                x = parent_A[i]
                y = parent_B[i]
                individual.append((x+y)/2)

        # mutation
        if trace :
            print("mutate")
        for i in range(N):
            og_val = individual[i]
            if sum(individual)<1:
                individual[i] += randint(0,1000)/1000 
            k = 0    
            while sum(individual)>1:
                if k >= 10:
                    individual[i] = og_val
                    break
                else :
                    individual[i] += randint(0,1000)/1000
                    k += 1
        if sum(individual) != 1:
            individual[randint(0,N-1)] += 1 - sum(individual)
        if sum(individual) > 1:
            if trace:
                print("mutation failed, we restart it")
            population.append(individual) 
        if trace :
            print("done")
    return population
