from scripts.algo.genetic.gen_alg import gen_alg_2dim
from scripts.algo.process_param import process_param
from scripts.utils.load_data import get_data


df, map = get_data()
mu, sigma, N = process_param(df)
selection = gen_alg_2dim(mu,sigma,N,trace = True)
print(selection)
