# Climat Méditerranée — Tunisie & Europe (1980–2024)

Analyse climatique end-to-end sur données de réanalyse **ERA5 (Copernicus/ECMWF)** :
acquisition, nettoyage, exploration statistique, dashboard interactif et modèles
prédictifs comparés, sur 45 ans de température, humidité et précipitations dans
le bassin méditerranéen.

**Auteur :** Bacem Ben Hamad

---

## Question scientifique

> **Comment les vagues de chaleur et le stress hydrique ont-ils évolué entre 1980 et 2024 sur le bassin méditerranéen ? La rive sud (Tunisie) et la rive nord (Europe méridionale) se réchauffent-elles au même rythme, et quelles sont les relations entre température, humidité et précipitations ?**

## Résultats clés

| Indicateur | Tunisie | Europe méridionale | Méditerranée |
|---|---:|---:|---:|
| Tendance T° (Mann-Kendall, p < 0,001) | +0,45 °C/déc | +0,51 °C/déc | +0,46 °C/déc |
| Jours d'été > 30 °C (1980-2000 → 2001-2024) | 72,8 → 80,1 j/an (+10 %) | 0,0 → 0,1 j/an | 3,3 → 20,8 j/an |
| Corrélation T° / humidité relative | -0,70 | -0,65 | -0,80 |

- Réchauffement net et statistiquement significatif sur tout le bassin, légèrement plus rapide côté nord (effet continental), mais conséquences plus marquées côté sud (climat déjà aride).
- Pas de tendance significative sur les précipitations à 45 ans — le stress hydrique progresse surtout via l'évapotranspiration (T° ↑, humidité ↓), pas via la baisse des pluies.
- Module prédictif séparé (`notebooks/projet_prediction.ipynb`) : SARIMA et OLS+saisonnalité battent un baseline naïf saisonnier (RMSE 1,22 °C vs 1,63 °C) sur le test 2015-2024, et projettent un réchauffement compatible avec les scénarios CMIP6 (SSP2-4.5) jusqu'en 2030.

## Données

- **Source :** Climate Data Store (CDS) — Copernicus / ECMWF
- **Dataset :** ERA5 hourly data on single levels (`reanalysis-era5-single-levels`)
- **Variables (3) :**
  - `2m_temperature` — température à 2 m
  - `total_precipitation` — précipitations totales
  - `2m_dewpoint_temperature` — point de rosée (→ humidité relative dérivée, formule de Magnus-Tetens)
- **Zone englobante :** 28°N – 46°N / -10°E – 40°E
- **3 sous-zones d'analyse :**
  - **Tunisie** (rive sud) : 30–37.5°N / 7.5–12°E
  - **Europe méridionale** (rive nord) : 40–46°N / -10–40°E (Espagne, sud France, Italie, Balkans, Grèce)
  - **Méditerranée** (bassin entier, référence)
- **Résolution :** 0.25° × 0.25° (~28 km), 1 valeur quotidienne à 12 UTC
- **Format :** NetCDF, lu via `xarray`

## Approche technique

- **Migration R → Python** documentée dans `docs/migration_R_Python.md` (table de correspondance dplyr→pandas, ggplot→plotly, lm→statsmodels…) — chaque traitement original est explicitement réimplémenté.
- **Statistiques de tendance** : régression OLS, pente de Sen (robuste aux outliers) et test de Mann-Kendall (standard OMM), anomalies vs climatologie de référence WMO 1981–2010.
- **Visualisations** : warming stripes (Ed Hawkins), analyse phénologique de la saison chaude, heatmaps de corrélation, dashboard Dash interactif (filtres temporels, carte spatiale, KPI).
- **Prédiction** : comparaison de 4 modèles (naïf saisonnier, OLS+saisonnalité, SARIMA, Random Forest) avec validation temporelle 2015-2024 et projection 2025-2030.

## Structure du projet

```
co2_emmission_analyse/
├── data/
│   ├── raw/              # NetCDF téléchargés depuis CDS (gitignored)
│   └── processed/        # Datasets nettoyés (parquet, gitignored)
├── scripts/
│   ├── 01_download_era5_test.py    # Test 1 mois
│   ├── 02_inspect_era5_test.py     # Inspection xarray
│   ├── 03_download_era5_full.py    # 1980-2024
│   ├── preprocessing.py            # Nettoyage + agrégations
│   ├── validation_quality.py       # Qualité des données
│   └── dashboard.py                # Application Dash
├── notebooks/
│   ├── projet_final.ipynb          # Notebook principal — analyse descriptive
│   ├── projet_prediction.ipynb     # Notebook complémentaire — modèles prédictifs
│   └── _archive/                   # Versions intermédiaires
├── dashboard/                # Assets CSS Dash
├── docs/
│   ├── qualite_donnees.md          # généré par validation_quality.py
│   └── migration_R_Python.md
├── tests/
│   └── test_humidity.py            # `pytest tests/` ou run direct
├── README.md
├── requirements.txt
└── .gitignore
```

## Lancement

### 1. Configuration CDS (une seule fois)

Créer `~/.cdsapirc` (Windows : `C:\Users\<user>\.cdsapirc`) :

```
url: https://cds.climate.copernicus.eu/api
key: <votre_cle_API>
```

Puis accepter les terms sur https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels.

### 2. Installation

```bash
pip install cdsapi xarray netCDF4 cfgrib pandas numpy scikit-learn statsmodels matplotlib seaborn dash plotly
```

### 3. Téléchargement test

```bash
python scripts/01_download_era5_test.py
python scripts/02_inspect_era5_test.py
```

### 4. Dashboard

```bash
python scripts/dashboard.py
# Ouvrir http://127.0.0.1:8050
```

### 5. Tests

```bash
python -m pytest tests/ -v
```

Tests inclus :
- `test_rh_saturation` : RH(T = Td) ≈ 100 %
- `test_rh_in_bounds` : RH ∈ [0, 100]
- `test_heatwave_*` : détection des épisodes consécutifs

## Notebook prédictif

Le notebook `notebooks/projet_prediction.ipynb` (séparé pour ne pas alourdir
le notebook principal) prolonge l'analyse descriptive par une **comparaison
de 4 modèles** sur la série mensuelle de température en Tunisie :

| Modèle | RMSE test 2015-2024 | Commentaire |
|---|---:|---|
| Naïf saisonnier (baseline) | 1,63 °C | À battre obligatoirement |
| OLS + saisonnalité | 1,22 °C | Tendance +0,43 °C/déc, R² = 0,97 |
| **SARIMA(1,1,1)×(1,1,1,12)** | **1,22 °C** | + intervalle de confiance natif |
| Random Forest (lags + sin/cos) | 1,45 °C | Souffre de l'extrapolation |

Le notebook couvre : décomposition STL, train/test split temporel, métriques
MAE/RMSE/MAPE, intervalles de confiance, **projection 2025-2030** et
comparaison aux scénarios CMIP6 du GIEC.

## Limites des données

- ERA5 est une **réanalyse** (observations + modèle physique), pas une observation directe — densité de stations faible au Maghreb → incertitude plus élevée que sur l'Europe occidentale (Hersbach et al., 2020).
- Résolution 0.25° (~28 km) : microclimats côtiers/urbains lissés.
- 1 valeur quotidienne à 12 UTC : cycle diurne perdu, acceptable pour les tendances long terme, insuffisant pour les pics instantanés.
- Précipitations : une seule valeur horaire (12 UTC) par jour, pas un cumul journalier ; pas de masque terre-mer sur les zones — les tendances de pluie sont indicatives.
