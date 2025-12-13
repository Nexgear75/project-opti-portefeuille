import numpy as np
import pandas as pd
# Importations spécifiques à l'optimisation
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from scripts.algo.process_param import process_param
from scripts.utils.load_data import get_data
from scripts.algo.func_obj_1_yield import get_yield
from scripts.algo.func_obj_2_risk import get_risk
from scripts.algo.func_obj_3_trans_cost import get_trans_cost

from scripts.utils.const import DEFAULT_POP_SIZE,DEFAULT_MAX_ITER

df,map = get_data()
mu, sigma, N = process_param(df)
# ---------------------------------------------------

class WalletProblem(Problem):
    """
    Définit le problème d'optimisation de portefeuille pour pymoo (Niveau 1: Bi-objectif).
    Variables de décision: N poids d'actifs (w).
    Objectifs: f1 (rendement, à minimiser) et f2 (risque, à minimiser).
    Contraintes: C_Base (somme(w)=1 et w_i >= 0).
    """

    def __init__(self, mu, sigma, dim = 2):
        self.mu = mu
        self.sigma = sigma
        self.N = len(mu)
        self.dim = dim
        
        super().__init__(
            n_var=self.N,
            n_obj=self.dim,  # goal func
            n_constr= 1 if self.dim == 2 else 2,
            xl=np.zeros(self.N), # w_i >= 0
            xu=np.ones(self.N) # w_i <= 1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Fonction d'évaluation appelée par l'algorithme génétique.
        X est une matrice où chaque ligne est un portefeuille (w).
        """
        F = np.zeros((X.shape[0], self.n_obj)) # Goal matrix
        G = np.zeros((X.shape[0], self.n_constr)) # constraints matrix

        for i, w in enumerate(X):
            # Goal function 1: yield
            f1 = get_yield(w,self.mu)
            # Goal function 2: risk
            f2 = get_risk(w, self.sigma)
            # Goal function 3: transition cost
            

            F[i, 0] = f1
            F[i, 1] = f2

            if self.dim == 3:
                base_w = []
                for j in range(self.N):
                    base_w.append(1/self.N)
                f3 = get_trans_cost(base_w,w)
                F[i, 2] = f3
            
            # weights sum = 1
            G[i, 0] = np.abs(np.sum(w) - 1.0) 
                
            if self.dim == 3:
                nb_not0 = np.sum(w>0.0001) # delta tolerence = 0.0001 here
                G[i,1] = np.abs(nb_not0 - 5) # K = 5
            
        out["F"] = F
        out["G"] = G


# --- 3. CONFIGURATION ET EXÉCUTION DE L'ALGORITHME NSGA-II ---
def alg_ngsa2(df,mu,sigma,dim = 2,trace = False):
    """docstr
    """
    if trace:
        print("initiating")
    problem = WalletProblem(mu, sigma, dim = dim)

    algorithm = NSGA2(
        pop_size=100, 
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=(1.0 / problem.N), eta=20),
        eliminate_duplicates=True 
    )

    if trace:
        print("optimizing")
    result = minimize(
        problem,
        algorithm,
        ('n_gen', DEFAULT_MAX_ITER), # stop after max iteration
        seed=1,
        verbose=trace
    )

    selection = result.pop.get("X")
    return selection

"""
# --- 4. ANALYSE ET PRÉSENTATION DES RÉSULTATS ---
print("\n--- Résultats de l'Optimisation ---")

if result.F is not None:
    # Récupérer les objectifs du front de Pareto
    F = result.F
    
    # Le Rendement (f1) doit être affiché comme positif
    rendements = -F[:, 0] * 100 # En %
    # Le Risque (f2) est la Variance. On affiche l'Écart-type (volatilité)
    volatilite = np.sqrt(F[:, 1]) * 100 # En %
    
    # Créer un DataFrame pour une analyse facile
    df_front_pareto = pd.DataFrame({
        'Rendement (%)': rendements,
        'Volatilité (%)': volatilite,
        'Risque (Variance)': F[:, 1]
    }).sort_values(by='Volatilité (%)')
    
    print(f"Nombre de solutions sur le Front de Pareto: {len(df_front_pareto)}")
    print("\nFront de Pareto (Extrait) :\n", df_front_pareto.head())

    # --- VISUALISATION (IMPORTANT POUR LE RAPPORT) ---
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_front_pareto['Volatilité (%)'], df_front_pareto['Rendement (%)'], s=10, label='Solutions Non-Dominées (Front de Pareto)')
    plt.xlabel('Volatilité (Risque, Écart-type annuel en %)')
    plt.ylabel('Rendement Annuel Attendu (%)')
    plt.title('Frontière Efficace de Markowitz (Niveau 1)')
    plt.grid(True)
    plt.legend()
    plt.show() # Ceci affichera le graphique
    """
