import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_raw_csvs(folder_path: str):
    """
    Fonction interne : Scanne le dossier et charge les fichers bruts
    output:
        liste de DataFrame et un mapping secteur.
    """

    if not os.path.exists(folder_path):
        console.print(
            f"[bold red]Erreur : Le dossier '{folder_path}' n'existe pas.[/bold red]"
        )
        return None, None

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        console.print(
            f"[bold red]Aucun fichier CSV trouvé dans '{folder_path}'.[/bold red]"
        )
        return None, None

    all_series = []
    sector_map = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            description="Lecture des fichiers CSV...", total=len(csv_files)
        )

        for file in csv_files:
            sector_name = file.replace(".csv", "").replace("_", "")
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

                for ticker in df.columns:
                    clean_ticker = ticker.strip()
                    series = df[ticker]
                    series.name = clean_ticker

                    all_series.append(series)
                    sector_map[clean_ticker] = sector_name

            except Exception as e:
                console.print(
                    f"[yellow]Attention : Impossible de lire {file}. Erreur: {e}[/yellow]"
                )

            progress.advance(task)

    return all_series, sector_map


def clean_and_merge(all_series, missing_threshold=0.1):
    """
    Fonction interne : Fusionne, nettoie les trous et retire les aberrations
    """
    if not all_series:
        return pd.DataFrame()

    with console.status("[bold green]Fusion et nettoyage des données en cours..."):
        full_df = pd.concat(all_series, axis=1)

        initial_shape = full_df.shape

        limit = len(full_df) * missing_threshold
        df_cleaned = full_df.dropna(axis=1, thresh=len(full_df) - limit)

        df_cleaned = df_cleaned.ffill()

        df_cleaned = df_cleaned.dropna(axis=0)

        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        df_cleaned = df_cleaned[df_cleaned > 0].dropna()

    removed_assets = initial_shape[1] - df_cleaned.shape[1]

    table = Table(title="Rapport de Nettoyage des Données")
    table.add_column("Métrique", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="magenta")

    table.add_row("Actifs initiaux trouvés", str(initial_shape[1]))
    table.add_row("Actifs supprimés (Trop de NaN)", str(removed_assets))
    table.add_row("Actifs finaux conservés", str(df_cleaned.shape[1]))
    table.add_row(
        "Période temporelle",
        f"{df_cleaned.index.min().date()} à {df_cleaned.index.max().date()}",
    )
    table.add_row("Nombre de jours de trading", str(len(df_cleaned)))

    console.print(table)

    return df_cleaned


def get_data(folder_path="data/"):
    """
    Récupère, nettoie et retourne les données prêtes à l'emploi.
    output:
        prices_df (pd.DataFrame): Matrice des prix ajustés nettoyés
        sector_map (dict): Dictionnaire {Ticker -> Secteur}
    """

    console.rule("[bold blue]Démarrage du module de données[/bold blue]")

    all_series, sector_map = load_raw_csvs(folder_path)

    if not all_series:
        return pd.DataFrame(), {}

    prices_df = clean_and_merge(all_series)

    final_tickers = prices_df.columns.tolist()
    final_sector_map = {k: v for k, v in sector_map.items() if k in final_tickers}

    console.print(
        f"[bold green]Données chargées avec succès ![/bold green] ({len(final_tickers)} actifs prêts)"
    )
    console.rule()

    return prices_df, final_sector_map
