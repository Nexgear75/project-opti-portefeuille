"""

    Multi-criteria optimisation project
    Fougeroux Alex & Robert Paul-Aime
    
    Main py file, here we call the functions running the pipe line

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

test_f2()
