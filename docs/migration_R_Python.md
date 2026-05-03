# Migration R → Python

Ce document trace explicitement la correspondance entre le pipeline R du
semestre 1 et son équivalent Python pour le projet S2.

## Changement de scope (volontaire et documenté)

Le pipeline R du S1 traitait les **émissions CO₂ anthropiques** (dataset
`co2-fossil-plus-land-use.csv`). Le projet S2 élargit le scope aux
**variables climatiques observées** (réanalyse ERA5) afin de répondre à
l'exigence "≥ 3 variables climatiques" du cahier des charges et d'utiliser
la source obligatoire CDS Copernicus.

| Aspect | S1 (R) | S2 (Python) |
|--------|--------|-------------|
| Source | OWID — `co2-fossil-plus-land-use.csv` | CDS Copernicus — ERA5 réanalyse |
| Format | CSV | NetCDF (xarray) |
| Variables | CO₂ fossile + land-use + total | T° 2m + précipitations + point de rosée |
| Granularité | pays × année | grille 0.25° × heure |
| Dimensions | tabulaire | tableaux multi-dim (time, lat, lon) |

## Table de correspondance des traitements

| # | Traitement R (S1) | Code R | Équivalent Python (S2) | Code Python |
|---|--|--|--|--|
| 1 | Chargement données | `read.csv()` | Lecture NetCDF | `xr.open_dataset()` |
| 2 | Inspection structure | `str(df)`, `head()` | `ds`, `ds.info()` | `xarray.Dataset` |
| 3 | Renommage colonnes | `dplyr::rename()` | Renommage variables | `ds.rename()` |
| 4 | Filtrage | `dplyr::filter()` | Sélection | `ds.sel(time=...)`, `ds.where()` |
| 5 | Agrégation | `group_by() %>% summarise()` | Agrégation temporelle | `ds.resample(time='1D').mean()` |
| 6 | Pivot/melt | `tidyr::pivot_longer()` | Reshape | `ds.to_dataframe()`, `pd.melt()` |
| 7 | Gestion NA | `is.na()`, `na.omit()` | Détection / imputation | `ds.isnull()`, `ds.interpolate_na()` |
| 8 | Détection valeurs aberrantes | IQR + boxplot ggplot2 | IQR + z-score | `pd.DataFrame.quantile`, `scipy.stats.zscore` |
| 9 | Statistiques globales | `summary()` | Résumé | `ds.describe()`, `da.mean()`, `da.std()` |
| 10 | Calcul per capita / dérivés | `mutate(... = .x / pop)` | Variables dérivées | xarray arithmetic |
| 11 | Carte choroplèthe | `ggplot + geom_polygon` | Carte 2D climat | `xr.plot.pcolormesh`, Plotly choropleth |
| 12 | Série temporelle | `ggplot + geom_line` | Time series | `matplotlib`, `plotly.express.line` |
| 13 | Boxplot saisonnier | `geom_boxplot` | Boxplot par saison | `seaborn.boxplot` |
| 14 | Heatmap décennale | `geom_tile` | Heatmap | `seaborn.heatmap`, Plotly |
| 15 | Régression linéaire | `lm()`, `predict()` | OLS / tendance | `statsmodels.OLS`, `np.polyfit` |
| 16 | Dashboard | `shinydashboard` | **Dashboard Dash (imposé)** | `dash`, `plotly` |

## Difficultés techniques rencontrées (à documenter pendant le projet)

À remplir au fur et à mesure :

- [ ] Gestion des dimensions multiples (time × lat × lon) absente en R/dplyr
- [ ] Encodage Windows cp1252 → forcer `PYTHONIOENCODING=utf-8`
- [ ] ...
