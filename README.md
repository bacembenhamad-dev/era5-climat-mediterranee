# Projet S2 — Analyse climatique Méditerranée & Tunisie (1980–2024)

**Cours :** ANALYSE DES DONNÉES — ESSAI (Pr. Ghazi Bel Mufti)
**Étudiant :** Bacem Ben Ahmad
**Période :** 6 semaines (S2)

---

## Question scientifique

> **Comment les vagues de chaleur et le stress hydrique ont-ils évolué entre 1980 et 2024 sur le bassin méditerranéen ? La rive sud (Tunisie) et la rive nord (Europe méridionale) se réchauffent-elles au même rythme, et quelles sont les relations entre température, humidité et précipitations ?**

---

## Données

- **Source :** Climate Data Store (CDS) — Copernicus / ECMWF
- **Dataset :** ERA5 hourly data on single levels (`reanalysis-era5-single-levels`)
- **Variables (3) :**
  - `2m_temperature` — température à 2 m
  - `total_precipitation` — précipitations totales
  - `2m_dewpoint_temperature` — point de rosée (→ humidité relative dérivée)
- **Zone englobante :** 28°N – 46°N / -10°E – 40°E
- **3 sous-zones d'analyse :**
  - **Tunisie** (rive sud) : 30–37.5°N / 7.5–12°E
  - **Europe méridionale** (rive nord) : 40–46°N / -10–40°E (Espagne, sud France, Italie, Balkans, Grèce)
  - **Méditerranée** (bassin entier, référence)
- **Résolution :** 0.25° × 0.25° (~28 km)
- **Format :** NetCDF, lu via `xarray`

## Migration R → Python

Le pipeline R du semestre 1 (CO₂ anthropique) est documenté dans
`docs/migration_R_Python.md` et chaque traitement R est explicitement
réimplémenté en Python (table de correspondance dans le rapport).

## Structure du projet

```
co2_emmission_v2/
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
│   ├── migration_R_Python.md
│   ├── soutenance_plan.md
│   └── soutenance_slides.html      # slides imprimables (Print → PDF)
├── tests/
│   └── test_humidity.py            # `pytest tests/` ou run direct
├── legacy/                   # artefacts S1 OWID — hors scope S2 (non touché)
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note sur `legacy/`** : ce dossier conserve les artefacts du semestre 1
> (analyse OWID des émissions CO₂ anthropiques). Il est volontairement laissé
> dans le dépôt pour tracer l'évolution R→Python du sujet, mais il n'entre
> pas dans la grille d'évaluation S2.

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

### 6. Slides de soutenance (PDF)

```bash
# Ouvrir docs/soutenance_slides.html dans le navigateur
# puis Fichier → Imprimer → Enregistrer en PDF (format paysage)
```

## Calendrier (6 semaines)

| Semaine | Objectif | Statut |
|---------|----------|--------|
| 1 | Question + acquisition CDS + 1ʳᵉ lecture | ✅ |
| 2 | Nettoyage, NA, aberrantes | ✅ |
| 3 | Transformation, agrégation, features | ✅ |
| 4 | Analyse exploratoire + viz statiques | ✅ |
| 5 | Dashboard Dash (filtres + carte spatiale + KPI) | ✅ |
| 5 bis | **Module prédictif** (SARIMA, OLS, RF) → 2030 | ✅ |
| 6 | Soutenance | en cours |

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

## Critères d'évaluation

| Critère | Pondération |
|---------|------------:|
| Pertinence question scientifique | 25 % |
| Variété des variables climatiques | 20 % |
| Migration R → Python | 20 % |
| Tableau de bord Dash | 20 % |
| Regard critique sur données | 10 % |
| Présentation orale (10+5 min) | 5 % |
