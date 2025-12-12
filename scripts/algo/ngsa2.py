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

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.N = len(mu)
        
        # Le problème a N variables (les poids w_i)
        # 0 <= w_i <= 1
        super().__init__(
            n_var=self.N,
            n_obj=2,  # f1 (rendement) et f2 (risque)
            n_constr=1, # Une seule contrainte d'égalité (somme des poids = 1)
            xl=np.zeros(self.N), # Limite inférieure de w_i : 0
            xu=np.ones(self.N) # Limite supérieure de w_i : 1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Fonction d'évaluation appelée par l'algorithme génétique.
        X est une matrice où chaque ligne est un portefeuille (w).
        """
        F = np.zeros((X.shape[0], self.n_obj)) # Matrice pour stocker les objectifs
        G = np.zeros((X.shape[0], self.n_constr)) # Matrice pour stocker les contraintes

        for i, w in enumerate(X):
            # Objectif 1: Rendement (f1 = -w^T * mu) - À MINIMISER
            # Note: w est déjà la ligne i de X
            f1 = -np.dot(w, self.mu) 
            f1 = get_yield(w,self.mu)
            # Objectif 2: Risque (f2 = w^T * Sigma * w) - À MINIMISER
            f2 = np.dot(w.T, np.dot(self.sigma, w))
            f2 = get_risk(w, self.sigma)

            F[i, 0] = f1
            F[i, 1] = f2
            
            # Contrainte 1 (G): Contrainte d'égalité. 
            # La somme des poids doit être égale à 1.
            # pymoo gère les contraintes comme G(x) <= 0. 
            # Pour une égalité (somme=1), on la transforme en: |somme(w) - 1| <= epsilon
            # On utilise une pénalité légère pour l'égalité: |somme(w) - 1|
            # Si le résultat est 0, la contrainte est satisfaite.
            # G[i, 0] = |np.sum(w) - 1|
            # Une façon plus simple de traiter l'égalité est souvent de la laisser non gérée 
            # dans G et de la faire gérer par un opérateur de réparation (Repair) pour les AG, 
            # mais pour l'instant, nous utilisons la contrainte d'inégalité stricte pour l'égalité.
            
            # En théorie, pour G(x) = 0, on utilise deux contraintes d'inégalité:
            # G1(x) = somme(w) - 1 <= 0  (somme <= 1)
            # G2(x) = 1 - somme(w) <= 0  (somme >= 1)
            # Puisque NSGA-II est moins performant sur l'égalité stricte, nous utilisons un 
            # simple écart de la somme à 1 comme contrainte de violation G (à minimiser à 0).
            # Note: Dans pymoo, une contrainte G > 0 est une violation.
            G[i, 0] = np.abs(np.sum(w) - 1.0) 
            
        out["F"] = F
        out["G"] = G


# --- 3. CONFIGURATION ET EXÉCUTION DE L'ALGORITHME NSGA-II ---
def alg_ngsa2(df,mu,sigma,trace = False):
    """docstr
    """
    print()
    # 3.1 Définir le problème
    problem = WalletProblem(mu, sigma)

    # 3.2 Définir l'algorithme
    algorithm = NSGA2(
        pop_size=100, # Taille de la population
        sampling=FloatRandomSampling(), # Echantillonnage
        crossover=SBX(prob=0.9, eta=15), # Croisement
        mutation=PM(prob=(1.0 / problem.N), eta=20),
        eliminate_duplicates=True # Éliminer les doublons
    )

    # 3.3 Optimisation (génération du front de Pareto)
    print("\n--- Début de l'optimisation NSGA-II (Niveau 1) ---")
    result = minimize(
        problem,
        algorithm,
        ('n_gen', DEFAULT_MAX_ITER), # Arrêter après 200 générations
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
