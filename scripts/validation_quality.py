"""
validation_quality.py
=====================

Audit de qualite des donnees ERA5 traitees. Genere :
    docs/qualite_donnees.md   (rapport markdown pour le rendu)
    data/processed/qa_*.png   (visualisations diagnostiques)

A executer apres preprocessing.py.

Equivalent R (S1) : summary(), ggplot+geom_boxplot, na.count
Equivalent Python : pandas.describe(), seaborn.boxplot, scipy.stats.zscore
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed" / "era5_daily.parquet"
DOC_OUT = PROJECT_ROOT / "docs" / "qualite_donnees.md"
QA_DIR = PROJECT_ROOT / "data" / "processed"

VARIABLES = {
    "t2m": "Temperature 2m (degC)",
    "d2m": "Point de rosee 2m (degC)",
    "rh":  "Humidite relative (%)",
    "tp":  "Precipitations 12 UTC (mm/h)",
}


def main() -> None:
    if not PROCESSED.exists():
        raise FileNotFoundError(
            f"Introuvable : {PROCESSED}\nLancez d'abord scripts/preprocessing.py"
        )

    df = pd.read_parquet(PROCESSED)
    print(f"Donnees chargees : {len(df):,} lignes")

    lines = []
    lines.append("# Qualite des donnees ERA5 (1980-2024)\n")
    lines.append(f"**Source :** Climate Data Store (Copernicus / ECMWF)  ")
    lines.append(f"**Dataset :** `reanalysis-era5-single-levels`  ")
    lines.append(f"**Resolution :** 0.25 deg x 0.25 deg (~28 km)  ")
    lines.append(f"**Frequence :** journaliere (12:00 UTC)  ")
    lines.append(f"**Zone :** 28-46 N / -10-40 E  ")
    lines.append(f"**Lignes :** {len(df):,}  ")
    lines.append(f"**Periode :** {df['date'].min().date()} -> {df['date'].max().date()}\n")

    # 1. Taux de NA
    lines.append("## 1. Taux de valeurs manquantes\n")
    lines.append("| Variable | NA total | % NA |")
    lines.append("|---|---:|---:|")
    for v in VARIABLES:
        n_na = int(df[v].isna().sum())
        pct = 100 * n_na / len(df)
        lines.append(f"| `{v}` | {n_na:,} | {pct:.2f} % |")
    lines.append("")

    # 2. Statistiques descriptives par zone
    lines.append("## 2. Statistiques descriptives par zone\n")
    for zone in df["zone"].unique():
        lines.append(f"### {zone.title()}\n")
        sub = df[df["zone"] == zone][list(VARIABLES.keys())]
        lines.append("```")
        lines.append(sub.describe().round(3).to_string())
        lines.append("```\n")

    # 3. Detection valeurs aberrantes (IQR + z-score)
    lines.append("## 3. Detection des valeurs aberrantes\n")
    lines.append("Methode : IQR (1.5 x interquartile) et |z| > 3.\n")
    lines.append("| Variable | Outliers IQR | Outliers z>3 |")
    lines.append("|---|---:|---:|")
    for v in VARIABLES:
        x = df[v].dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        n_iqr = int(((x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)).sum())
        n_z = int((np.abs(stats.zscore(x)) > 3).sum())
        lines.append(f"| `{v}` | {n_iqr:,} | {n_z:,} |")
    lines.append("")

    # 4. Visualisations
    lines.append("## 4. Visualisations diagnostiques\n")

    # Boxplots par variable, par zone
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(4*len(VARIABLES), 4))
    for ax, (v, label) in zip(axes, VARIABLES.items()):
        sns.boxplot(data=df, x="zone", y=v, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
    fig.suptitle("Distribution par zone (1980-2024)", fontweight="bold")
    fig.tight_layout()
    f1 = QA_DIR / "qa_boxplots.png"
    fig.savefig(f1, dpi=120)
    plt.close(fig)
    lines.append(f"![Boxplots]({f1.name})\n")

    # Boxplots saisonniers (Tunisie uniquement)
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(4*len(VARIABLES), 4))
    sub = df[df["zone"] == "tunisie"]
    for ax, (v, label) in zip(axes, VARIABLES.items()):
        sns.boxplot(data=sub, x="season", y=v,
                    order=["DJF","MAM","JJA","SON"], ax=ax)
        ax.set_title(label)
    fig.suptitle("Saisonnalite Tunisie (1980-2024)", fontweight="bold")
    fig.tight_layout()
    f2 = QA_DIR / "qa_seasonal.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)
    lines.append(f"![Saisonnalite]({f2.name})\n")

    # Continuite temporelle : couverture annuelle
    coverage = (df.groupby(["year", "zone"])
                  .size()
                  .reset_index(name="n_days"))
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=coverage, x="year", y="n_days", hue="zone", ax=ax)
    ax.set_title("Couverture temporelle (jours / an)")
    ax.set_ylabel("Nb de jours")
    fig.tight_layout()
    f3 = QA_DIR / "qa_coverage.png"
    fig.savefig(f3, dpi=120)
    plt.close(fig)
    lines.append(f"![Couverture]({f3.name})\n")

    # 5. Limites et biais
    lines.append("## 5. Limites et biais identifies\n")
    lines.append("- **Resolution spatiale 0.25 deg** : un pixel = ~28 km. ")
    lines.append("  Les phenomenes locaux (ilots de chaleur urbains, microclimats ")
    lines.append("  cotiers fins) sont moyennes.")
    lines.append("- **Pas de temps : 1 valeur quotidienne a 12:00 UTC** : on perd ")
    lines.append("  le cycle diurne. Acceptable pour les tendances long terme, ")
    lines.append("  insuffisant pour les pics de chaleur instantanes.")
    lines.append("- **ERA5 = reanalyse, pas observation directe** : assimilation ")
    lines.append("  d'observations + modele. Les zones a faible densite stations ")
    lines.append("  (Sahara, Mediterranee centrale) ont une incertitude plus elevee.")
    lines.append("- **`tp` est l'accumulation horaire a 12:00** : ce n'est pas le ")
    lines.append("  total quotidien. Pour le total quotidien il faudrait sommer 24 ")
    lines.append("  pas horaires.")

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport ecrit : {DOC_OUT}")
    print(f"Figures      : {f1.name}, {f2.name}, {f3.name}")


if __name__ == "__main__":
    main()
