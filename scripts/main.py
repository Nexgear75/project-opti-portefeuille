"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime
    
    Main py file, here we call the functions running the pipe line

"""

from scripts.algo.genetic.gen_alg import gen_alg_2dim, gen_alg_3dim
from scripts.algo.ngsa2 import alg_ngsa2
from scripts.algo.process_param import process_param
from scripts.utils.load_data import get_data
from scripts.algo.get_pareto import get_pareto


def use_alg(algo = "gen",dim = 2,trace = False):
    """
        using the genetic algo

        input:
            (String): algo: gen / ngsa2
            (int): dim: number of goal function 2 / 3
            (Bool): trace

        output:
            (list[dict{
                        "status": (Bool): success state
                        "weights": (dict{name : weight}): wallet
                        "metrics": (dict{})
                        }]): result: result list top of the solution (ruled by the inputs in the alg function)
    """
    if trace:
        print("collecting data")
    df, map = get_data()
    if trace:
        print("get parameters from data")
    mu, sigma, N = process_param(df)
    if trace:
        print("weight processing through optimization alg")

    if True : # try:
        if algo == "gen":
            if dim == 2:
                selection = gen_alg_2dim(mu,sigma,N)
            else:
                selection = gen_alg_3dim(mu,sigma,N)
        elif algo == "ngsa2":
            selection = alg_ngsa2(df,mu,sigma,dim = dim)
        if trace:
            print("alg ran successfuly")
        # except:
        print("something went rong with the genetic algo")
    if trace:
        print(len(selection)," elements selected")
        print("processing pareto front")
    selection = get_pareto(selection,mu,sigma,dim)
    if trace:
        print(len(selection)," pereto optimal elements")
        print("preparing the result to send")
    result = []
    for obj in selection:
        obj_result = {}
        obj_result["status"] = True
        obj_result["weights"] = {}
        for i in range(len(obj)):
            weight = obj[i]
            column_name = df.columns[i]
            obj_result["weights"][column_name] = weight
        obj_result["metrics"] = {}
        result.append(obj_result)
    return result

result = []
result.append(use_alg(algo = "gen", dim = 2, trace = True))
result.append(use_alg(algo = "gen", dim = 3, trace = True))
result.append(use_alg(algo = "ngsa2", dim = 2, trace = True))
result.append(use_alg(algo = "ngsa2", dim = 3, trace = True))

for dict in result:
    weights = dict[0]["weights"]
    print(max(weights, key=weights.get))





"""
def test_f1():
    from scripts.utils.load_data import get_data
    from scripts.algo.process_param import process_param
    from scripts.algo.func_obj_1_yield import get_yield

    df, sector_mapl = get_data()
    mu,sigma,N = process_param(df)
    weight = []

    for i in range(N):
        weight.append(1/N)

    f_1 = get_yield(weight,mu)
    print(type(f_1))
    print(f_1)
"""

"""
def test_f2():
    from scripts.utils.load_data import get_data
    from scripts.algo.process_param import process_param
    from scripts.algo.func_obj_2_risk import get_risk

    df, sector_mapl = get_data()
    mu,sigma,N = process_param(df)
    weight = []

    for i in range(N):
        weight.append(1/N)

    f_2 = get_risk(weight,sigma)
    print(type(f_2))
    print(f_2)
"""
