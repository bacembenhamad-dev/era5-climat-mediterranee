# Plan de soutenance — 10 min + 5 min Q&A

**Format imposé par le prof** : 10 min de présentation + 5 min de questions.

## Structure (10 min)

| Temps | Slide(s) | Contenu | Points clés à mentionner |
|------:|---------|---------|-------------------------|
| **0-1 min** | 1-2 | **Question scientifique + zone** | Vagues de chaleur & stress hydrique en Tunisie/Méditerranée 1980-2024. Pourquoi : pertinence locale (Tunisie), bassin sensible au changement climatique, données ERA5 disponibles. |
| **1-3 min** | 3-5 | **Source données CDS + qualité** | ERA5 réanalyse, 0.25°×0.25°, 3 variables (T2m, dewpoint→RH, précipitations). Limites : résolution spatiale, 1 valeur/jour à 12h UTC, biais de réanalyse. Montrer `qualite_donnees.md`. |
| **3-5 min** | 6-8 | **Pipeline R → Python** | Table de correspondance (dplyr→pandas, ggplot→plotly, lm→statsmodels). Difficultés : passage de tabulaire (R/CSV) à multi-dim (xarray/NetCDF), zip CDS contenant 2 NetCDF, encodage Windows cp1252. |
| **5-8 min** | 9 | **Démo dashboard EN DIRECT** | Lancer `python scripts/dashboard.py`. Montrer : filtre temporel, sélection variable, KPI jours>30°C, tendance annuelle, boxplot saisonnier, heatmap. |
| **8-10 min** | 10-11 | **Résultats + critique** | Tendance T° (X°C/décennie). Évolution canicules. Stress hydrique : précipitations ↓ en JJA. Critique : assimilation modèle, manque de stations sol au Sahara, pas de cycle diurne. |

## Slides à produire (PDF)

1. **Page de garde** : Titre, nom, date, ESSAI, Pr. Bel Mufti
2. **Question scientifique** : 1 phrase + carte de la zone
3. **Données CDS** : capture du portail Copernicus + tableau des variables
4. **Pipeline d'acquisition** : diagramme `cdsapi → ZIP → NetCDF → xarray`
5. **Qualité données** : taux de NA, IQR, boxplots saisonniers
6. **Migration R → Python** : tableau de correspondance (slide pleine page)
7. **Difficultés techniques** : 3 anecdotes (zip CDS, encodage, dimensions)
8. **Résultats clés** : tendance T° + nombre de jours canicule (graphes)
9. **Démo Dashboard** : screenshot du Dashboard (en cas de souci de démo live)
10. **Regard critique** : limites des données ERA5 + biais
11. **Conclusion** : réponse à la question scientifique + perspectives

## Préparation pré-soutenance

### J-1
- [ ] Tester le dashboard sur la machine de présentation (vérifier port 8050 libre)
- [ ] Préparer un screenshot HD du dashboard en backup si la démo plante
- [ ] Imprimer le `qualite_donnees.md` au cas où
- [ ] Synchroniser le dépôt GitHub (README à jour)

### Jour J
- [ ] Lancer `python scripts/dashboard.py` AVANT la présentation, garder l'onglet ouvert
- [ ] Avoir le fichier `era5_daily.parquet` prêt en local
- [ ] Apporter clé USB avec slides PDF + screenshot dashboard

## Anticiper les questions du prof

| Question probable | Réponse |
|---|---|
| *Pourquoi ERA5 et pas ERA5-Land ?* | ERA5 couvre océan + terre (Méditerranée nécessite l'océan), 0.25° suffisant pour vagues de chaleur. ERA5-Land aurait été 0.1° mais terre uniquement. |
| *Pourquoi 1 valeur/jour à 12h UTC ?* | Compromis taille de données / précision. Pour les tendances long terme c'est suffisant. Pour le pic de chaleur instantané il faudrait 4×/jour minimum. |
| *Comment validez-vous votre humidité relative dérivée ?* | Formule de Magnus-Tetens (référence métérologique standard). Bornée à [0,100]. Comparable à celle des stations Météo-France/INM. |
| *Pourquoi pas de scikit-learn / modèle ML ?* | Le projet est exploratoire et descriptif. Une régression linéaire `statsmodels` suffit pour quantifier la tendance. ML aurait été disproportionné. |
| *Quel est le biais ERA5 connu sur cette zone ?* | Sous-représentation en stations sol au Maghreb → incertitude plus élevée que sur Europe occidentale. Documenté dans la littérature (Hersbach 2020). |
| *Combien de temps pour le téléchargement ?* | ~50 min pour 45 ans × ~21 MB = 940 MB total. Script reprend automatiquement si interruption. |

## Critères d'évaluation rappelés

| Critère | % | Comment marquer des points |
|---|--:|---|
| Pertinence question scientifique | 25 % | Bien justifier la zone Tunisie + lien climat local |
| Variété variables climatiques | 20 % | 3 variables + 1 dérivée (RH) = 4 |
| Migration R → Python | 20 % | Table de correspondance explicite + difficultés |
| Tableau de bord Dash | 20 % | Démo fluide + KPI + filtres + interactivité |
| Regard critique | 10 % | Limites ERA5, résolution, biais, alternatives |
| Présentation orale | 5 % | Respecter 10 min, parler clair |
