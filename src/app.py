import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import random


# Gestion des imports dans les autres fichiers
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from scripts.utils.load_data import get_data
    from scripts.main import use_alg_ngsa2
    from scripts.algo.process_param import process_param
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


def run_real_optimization(prices_df, r_min_target):
    """
    ADAPTATEUR : Connecte l'algo NSGA-II de ton collègue à l'interface Streamlit.
    """
    mu, sigma, N = process_param(prices_df)

    try:
        raw_results = use_alg_ngsa2(dim=3, trace=False)
    except Exception as e:
        st.error(f"Erreur dans l'algo NSGA-II : {e}")
        return {"status": False}

    processed_candidates = []

    for item in raw_results:
        weights_dict = item["weights"]

        w_vec = np.array([weights_dict.get(col, 0.0) for col in prices_df.columns])

        total_sum = np.sum(w_vec)
        if total_sum > 0:
            w_vec = w_vec / total_sum

            weights_dict = {
                col: val for col, val in zip(prices_df.columns, w_vec) if val > 0
            }

        ret = np.dot(w_vec, mu)
        risk = np.sqrt(np.dot(w_vec.T, np.dot(sigma, w_vec)))

        processed_candidates.append(
            {
                "weights": weights_dict,
                "metrics": {"expected_return": ret, "volatility": risk},
            }
        )

    valid_portfolios = [
        p
        for p in processed_candidates
        if p["metrics"]["expected_return"] >= r_min_target
    ]

    if valid_portfolios:
        best_portfolio = min(valid_portfolios, key=lambda x: x["metrics"]["volatility"])
    else:
        best_portfolio = max(
            processed_candidates, key=lambda x: x["metrics"]["expected_return"]
        )
        st.warning(
            f"Attention : L'algorithme n'a pas trouvé de solution atteignant {r_min_target:.1%}. Voici la meilleure solution disponible."
        )

    return {
        "status": True,
        "weights": best_portfolio["weights"],
        "metrics": best_portfolio["metrics"],
        "all_front": processed_candidates,
    }


st.sidebar.title("Paramètres")

if "asset_multiselect" not in st.session_state:
    st.session_state["asset_multiselect"] = (
        all_tickers[:5] if len(all_tickers) > 5 else all_tickers
    )


def generate_random_portfolio():
    """Génère une sélection diversifiée de 10 à 20 actifs"""
    target_size = random.randint(10, 20)
    new_selection = []

    sectors_dict = {}
    for ticker in all_tickers:
        sec = sector_map.get(ticker, "Unknown")
        if sec not in sectors_dict:
            sectors_dict[sec] = []
        sectors_dict[sec].append(ticker)

    for sec, tickers in sectors_dict.items():
        if tickers:
            new_selection.append(random.choice(tickers))

    remaining_slots = target_size - len(new_selection)
    if remaining_slots > 0:
        pool = [t for t in all_tickers if t not in new_selection]
        if pool:
            extras = random.sample(pool, min(len(pool), remaining_slots))
            new_selection.extend(extras)

    return new_selection


if st.sidebar.button("Sélection Aléatoire Diversifiée", type="secondary"):
    st.session_state["asset_multiselect"] = generate_random_portfolio()
    st.rerun()

selected_assets = st.sidebar.multiselect(
    "1. Sélection des Actifs",
    options=all_tickers,
    key="asset_multiselect",
    help="Utilisez le bouton ci-dessus pour générer un panier varié.",
)

r_min_input = st.sidebar.slider(
    "2. Rendement Annuel Minimal visé (%)",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=0.5,
)
r_min_val = r_min_input / 100.0

st.sidebar.divider()

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
        with st.spinner(
            "Optimisation NSGA-II en cours (cela peut prendre du temps)..."
        ):
            subset_df = prices_df[selected_assets]

            resultat = run_real_optimization(subset_df, r_min_val)

            if resultat["status"]:
                metrics = resultat["metrics"]
                col1, col2, col3 = st.columns(3)

                col1.metric("Rendement Espéré", f"{metrics['expected_return']:.2%}")
                col2.metric("Risque (Volatilité)", f"{metrics['volatility']:.2%}")
                sharpe = (
                    metrics["expected_return"] / metrics["volatility"]
                    if metrics["volatility"] > 0
                    else 0
                )
                col3.metric("Ratio de Sharpe (Est.)", f"{sharpe:.2f}")

                st.divider()

                col_chart, col_table = st.columns([2, 1])

                with col_chart:
                    st.subheader("Allocation Sectorielle & Actifs")

                    weights_data = resultat["weights"]
                    df_weights = pd.DataFrame(
                        list(weights_data.items()), columns=["Ticker", "Poids"]
                    )

                    df_weights["Secteur"] = df_weights["Ticker"].map(sector_map)

                    df_weights = df_weights[df_weights["Poids"] > 0.001]

                    fig_sun = px.sunburst(
                        df_weights,
                        path=["Secteur", "Ticker"],
                        values="Poids",
                        color="Secteur",
                        title="Répartition Macro-économique",
                    )
                    st.plotly_chart(fig_sun, width="stretch")

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
                st.subheader("Frontière de Pareto (Résultats de l'Algo)")

                all_points = resultat.get("all_front", [])
                x_risk = [p["metrics"]["volatility"] for p in all_points]
                y_ret = [p["metrics"]["expected_return"] for p in all_points]

                chosen_risk = resultat["metrics"]["volatility"]
                chosen_ret = resultat["metrics"]["expected_return"]

                fig_pareto = go.Figure()

                # Nuage de points
                fig_pareto.add_trace(
                    go.Scatter(
                        x=x_risk,
                        y=y_ret,
                        mode="markers",
                        name="Solutions NSGA-II",
                        marker=dict(color="blue", opacity=0.5),
                    )
                )

                # Point sélectionné
                fig_pareto.add_trace(
                    go.Scatter(
                        x=[chosen_risk],
                        y=[chosen_ret],
                        mode="markers+text",
                        name="Sélectionné",
                        text=["Votre Choix"],
                        textposition="top center",
                        marker=dict(color="red", size=15, symbol="star"),
                    )
                )

                fig_pareto.update_layout(
                    xaxis_title="Risque (Volatilité)",
                    yaxis_title="Rendement Espéré",
                )

                st.plotly_chart(fig_pareto, width="stretch")

with tab3:
    st.markdown("""
    Ce dashboard implémente une optimisation de portefeuille multi-critère basée sur :
    1. **Modèle de Markowitz (Moyenne-Variance)** pour le niveau 1.
    2. **Algorithme NSGA-II** pour l'intégration des coûts de transaction (Niveau 2).
    
    ### Sources de Données
    Les données proviennent des constituants du S&P 500 classés par secteurs GICS.
    """)
