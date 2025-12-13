# Optimisation de Portefeuille Multi-Critère (NSGA-II)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Terminé-success?style=for-the-badge)

Une application interactive d'aide à la décision pour la gestion de portefeuille financier. Ce projet implémente des algorithmes d'optimisation avancés (NSGA-II) pour résoudre le problème complexe du compromis Rendement / Risque / Coûts de transaction.

---

## Fonctionnalités Clés

* **Sélection d'Actifs :** Choix manuel ou génération aléatoire diversifiée parmi les constituants du S&P 500.
* **Algorithmes :** Comparaison possible entre :
  * **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) : Approche multi-objectifs complète.
  * **Algorithme Génétique Standard** : Approche alternative.
* **Visualisation Interactive :**
  * Frontière de Pareto dynamique (Plotly).
  * Graphique Sunburst pour l'allocation sectorielle (Diversification).
  * Tableaux de performance détaillés (Rendement, Volatilité, Sharpe).
* **Personnalisation :** Définition d'un rendement cible ($r_{min}$) en temps réel.

---

## Stack Technique

* **Langage :** Python
* **Interface Utilisateur :** Streamlit
* **Manipulation de Données :** Pandas, NumPy
* **Visualisation :** Plotly Express & Graph Objects
* **Logique Métier :** Implémentation custom de NSGA-II (Python pur)

---

## Installation et Lancement

Suivez ces étapes pour faire tourner le projet sur votre machine locale.

### 1. Cloner le projet

```bash
git clone [https://github.com/VOTRE_PSEUDO/project-opti-portefeuille.git](https://github.com/VOTRE_PSEUDO/project-opti-portefeuille.git)
cd project-opti-portefeuille
```

### 2. Créer un environnement virtuel (Recommandé)
* Sur Mac/Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

* Sur Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

L'application doit être lancée depuis la racine du projet pour garantir le chargement correct des modules.

```bash
streamlit run src/app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501/`

## Structure du projet

```
project-opti-portefeuille/
├── data/               # Fichiers CSV (S&P 500 par secteurs)
├── scripts/            # Cœur algorithmique
│   ├── algo/           # Implémentation NSGA-II et Génétique
│   ├── utils/          # Chargement des données
│   └── main.py         # Point d'entrée logique
├── src/
│   └── app.py          # Interface Streamlit (Frontend)
├── requirements.txt    # Liste des dépendances
└── README.md           # Documentation
```
