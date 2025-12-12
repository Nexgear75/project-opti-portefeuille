import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Import de get data (problème pour python de trouver les fichiers)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from scripts.load_data import get_data
except ImportError as e:
    st.error(f"Erreur d'import : {e}")
    st.stop()

# Configuration de la page

st.set_page_config(
    page_title="Optimisateur de Portefeuille", page_icon="📈", layout="wide"
)

# Chargement des données


@st.cache_data
def load_data():
    return get_data("data")


try:
    prices_df, sector_map = load_data()
    all_tickers = prices_df.columns.tolist()
except Exception as e:
    st.error(f"Erreur critique lors du chargement des données : {e}")
    st.stop()

# Fonction de simulation (à finir avec le code de PA)


def mock_markowitz_optimization(selected_assets, r_min):
    """
    Génère un résultat aléatoire respectant le format d'entrée (a update plus tard)
    input:
        Liste d'actifs, Rendement min
    output:
        Dictionnaire {status, weights, metrics}
    """

    if not selected_assets:
        return {"status": False}

    # Génération de poids aléatoires qui somment à 1 (comme condition réel)
    n = len(selected_assets)
    weights = np.random.random(n)
    weights /= weights.sum()

    # Création du dictionnaire de poids
    weights_dict = dict(zip(selected_assets, weights))

    # Simulation des métriques de rendements et de risque
    simulated_return = r_min + np.random.uniform(0.01, 0.05)
    simulated_volatility = simulated_return * 0.8 + np.random.uniform(0.01, 0.03)

    return {
        "status": True,
        "weights": weights_dict,  # <--- LA RECETTE
        "metrics": {  # <--- LES CHIFFRES
            "expected_return": simulated_return,
            "volatility": simulated_volatility,
        },
    }


# Sidebar
st.sidebar.title("Paramètres")

# Sélection des actifs
selected_assets = st.sidebar.multiselect(
    "1. Sélection des Actifs",
    options=all_tickers,
    default=all_tickers[:5] if len(all_tickers) > 5 else all_tickers,
)

# Sélection du rendement minimal
r_min_input = st.sidebar.slider(
    "2. Rendement Annuel Minimal visé (%)",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=0.5,
)
r_min_val = r_min_input / 100.0

# Bouton de lancement
run_btn = st.sidebar.button("Lancer l'Optimisation", type="primary")

# Page principale
st.title("Dashboard d'Optimisation de Portefeuille")

# Création des onglets
tab1, tab2, tab3 = st.tabs(
    ["Analyse de Marché", "Portefeuille Optimal", "Détails du Projet"]
)

with tab1:
    st.header("Aperçu des données historiques")
    if selected_assets:
        subset = prices_df[selected_assets]
        normalized_df = subset / subset.iloc[0] * 100

        st.line_chart(normalized_df)

        with st.expander("Voir les données brutes"):
            st.dataframe(subset.tail(10))

    else:
        st.info("Veuillez sélectionner des actifs dans la barre latérale.")

with tab2:
    if run_btn and selected_assets:
        with st.spinner("Calcul du portefeuille optimal en cours..."):
            resultat = mock_markowitz_optimization(selected_assets, r_min_val)

            if resultat["status"]:
                metrics = resultat["metrics"]
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Rendement Espéré",
                    f"{metrics['expected_return']:.2%}",
                    delta="Annuel",
                )
                col2.metric(
                    "Risque (Volatilité)",
                    f"{metrics['volatility']:.2%}",
                    delta_color="inverse",
                )
                col3.metric(
                    "Ratio de Sharpe (Est.)",
                    f"{metrics['expected_return'] / metrics['volatility']:.2f}",
                )

                st.divider()

                col_chart, col_table = st.columns([2, 1])

                with col_chart:
                    st.subheader("Allocation Sectorielle & Actifs")

                    # Transformation du dictionnaire de poids en dataframe
                    weights_data = resultat["weights"]
                    df_weights = pd.DataFrame(
                        list(weights_data.items()), columns=["Ticker", "Poids"]
                    )

                    df_weights["Secteur"] = df_weights["Ticker"].map(sector_map)

                    df_weights = df_weights[df_weights["Poids"] > 0.001]

                    # Graphique sunburst
                    fig_sun = px.sunburst(
                        df_weights,
                        path=["Secteur", "Ticker"],
                        values="Poids",
                        color="Secteur",
                        title="Répartition Macro-économique du Portefeuille",
                    )
                    st.plotly_chart(fig_sun, width="stretch")

                    # Graphique de barres
                    fig_bar = px.bar(
                        df_weights,
                        x="Poids",
                        y="Ticker",
                        orientation="h",
                        color="Secteur",
                        title="Allocation des Actifs",
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                with col_table:
                    st.subheader("Détail des Poids")
                    st.dataframe(
                        df_weights.sort_values(
                            by="Poids", ascending=False
                        ).style.format({"Poids": "{:.2%}"}),
                        width="stretch",
                        hide_index=True,
                    )

                st.divider()
                st.subheader("Positionnement sur la Frontière Efficiente")

                # Génération de faux points pour dessiner la courbe
                # A fournir par PA
                fake_risks = np.linspace(0.05, 0.30, 50)
                fake_returns = np.log(fake_risks * 10) * 0.10 + 0.05

                fig_pareto = px.scatter(
                    x=fake_risks,
                    y=fake_returns,
                    labels={"x": "Risque (Volatilité)", "y": "Rendement Espéré"},
                    title="Frontière Efficiente (Théorique)",
                )

                fig_pareto.add_trace(
                    go.Scatter(
                        x=[metrics["volatility"]],
                        y=[metrics["expected_return"]],
                        mode="markers+text",
                        marker=dict(color="red", size=15, symbol="star"),
                        name="Portefeuille Sélectionné",
                        text=["Votre Portefeuille"],
                        textposition="top center",
                    )
                )
                st.plotly_chart(fig_pareto, width="stretch")
            else:
                st.error("L'optimisation à échoué. Essayez d'assouplir les contraintes")
    elif not selected_assets:
        st.info("Sélectionnez des actifs pour commencer.")
    else:
        st.info("Cliquez sur 'Lancer l'optimisation' dans la barre latérale.")

with tab3:
    st.markdown("""
    Ce dashboard implémente une optimisation de portefeuille multi-critère basée sur :
    1. **Modèle de Markowitz (Moyenne-Variance)** pour le niveau 1.
    2. **Algorithme NSGA-II** pour l'intégration des coûts de transaction (Niveau 2).
    
    ### Sources de Données
    Les données proviennent des constituants du S&P 500 classés par secteurs GICS.
    """)
