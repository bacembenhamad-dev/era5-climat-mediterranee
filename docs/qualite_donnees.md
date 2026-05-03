# Qualite des donnees ERA5 (1980-2024)

**Source :** Climate Data Store (Copernicus / ECMWF)  
**Dataset :** `reanalysis-era5-single-levels`  
**Resolution :** 0.25 deg x 0.25 deg (~28 km)  
**Frequence :** journaliere (12:00 UTC)  
**Zone :** 28-46 N / -10-40 E  
**Lignes :** 49,311  
**Periode :** 1980-01-01 -> 2024-12-31

## 1. Taux de valeurs manquantes

| Variable | NA total | % NA |
|---|---:|---:|
| `t2m` | 0 | 0.00 % |
| `d2m` | 0 | 0.00 % |
| `rh` | 0 | 0.00 % |
| `tp` | 0 | 0.00 % |

## 2. Statistiques descriptives par zone

### Mediterranee

```
             t2m        d2m         rh         tp
count  16437.000  16437.000  16437.000  16437.000
mean      20.152      9.069     55.451      0.060
std        6.525      4.405      5.111      0.037
min        6.771     -2.155     43.847      0.001
25%       14.185      5.272     51.349      0.031
50%       19.917      9.014     54.737      0.055
75%       26.308     13.242     59.285      0.083
max       32.529     17.420     73.693      0.297
```

### Tunisie

```
             t2m        d2m         rh         tp
count  16437.000  16437.000  16437.000  16437.000
mean      24.098      9.338     45.050      0.025
std        7.441      4.482      9.495      0.065
min        5.730     -5.557     20.632      0.000
25%       17.439      5.807     38.111      0.000
50%       24.093      9.494     43.577      0.003
75%       30.745     13.186     50.897      0.018
max       41.293     20.370     86.578      1.332
```

### Europe

```
             t2m        d2m         rh         tp
count  16437.000  16437.000  16437.000  16437.000
mean      15.699      8.738     66.042      0.104
std        6.773      5.451      6.119      0.071
min       -0.776     -7.451     47.867      0.000
25%        9.825      4.292     61.471      0.049
50%       15.371      8.731     65.685      0.091
75%       21.854     13.698     70.415      0.145
max       30.273     19.114     85.149      0.496
```

## 3. Detection des valeurs aberrantes

Methode : IQR (1.5 x interquartile) et |z| > 3.

| Variable | Outliers IQR | Outliers z>3 |
|---|---:|---:|
| `t2m` | 0 | 0 |
| `d2m` | 1 | 15 |
| `rh` | 56 | 2 |
| `tp` | 1,627 | 744 |

## 4. Visualisations diagnostiques

![Boxplots](qa_boxplots.png)

![Saisonnalite](qa_seasonal.png)

![Couverture](qa_coverage.png)

## 5. Limites et biais identifies

- **Resolution spatiale 0.25 deg** : un pixel = ~28 km. 
  Les phenomenes locaux (ilots de chaleur urbains, microclimats 
  cotiers fins) sont moyennes.
- **Pas de temps : 1 valeur quotidienne a 12:00 UTC** : on perd 
  le cycle diurne. Acceptable pour les tendances long terme, 
  insuffisant pour les pics de chaleur instantanes.
- **ERA5 = reanalyse, pas observation directe** : assimilation 
  d'observations + modele. Les zones a faible densite stations 
  (Sahara, Mediterranee centrale) ont une incertitude plus elevee.
- **`tp` est l'accumulation horaire a 12:00** : ce n'est pas le 
  total quotidien. Pour le total quotidien il faudrait sommer 24 
  pas horaires.