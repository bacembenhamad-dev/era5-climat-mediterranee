"""
preprocessing.py
================

Pipeline de pretraitement ERA5 (1980-2024).

Etapes :
1. Charge tous les NetCDF annuels (instant + accum) -> dataset xarray fusionne
2. Conversions d'unites :
       t2m, d2m : K -> degres C
       tp       : m -> mm (cumul horaire a 12:00 UTC, donc mm/h)
3. Variable derivee : humidite relative (RH) a partir de t2m et d2m
       (formule de Magnus-Tetens)
4. Detection valeurs aberrantes (IQR + z-score) sur chaque variable
5. Gestion des NA :
       - rapport du taux de NA par variable
       - interpolation lineaire le long de l'axe temporel sur les variables
         instantanees (t2m, d2m), reset a 0 pour la precipitation cumulee
6. Agregation spatiale : moyenne sur la zone "Tunisie" (33-37 N, 8-12 E)
   et sur la zone "Mediterranee complete" pour comparaison
7. Sauvegarde en parquet : data/processed/era5_daily.parquet

Equivalent R (S1) : dplyr::filter, mutate, summarise, na.omit, IQR
Equivalent Python : xarray + pandas + scipy.stats
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5_full"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Zones (latitudes ERA5 decroissantes -> slice(haut, bas))
TUNISIA_BOX = dict(latitude=slice(37.5, 30.0), longitude=slice(7.5, 12.0))
# Europe meridionale = rive nord de la Mediterranee (Espagne, Italie,
# sud France, Balkans, Grece). Permet une comparaison rive sud (Tunisie)
# vs rive nord (Europe) du bassin mediterraneen.
EUROPE_BOX = dict(latitude=slice(46.0, 40.0), longitude=slice(-10.0, 40.0))


def load_all_years() -> xr.Dataset:
    """Charge et concatene tous les NetCDF annuels."""
    instant_files = sorted(RAW_DIR.glob("*/instant.nc"))
    accum_files = sorted(RAW_DIR.glob("*/accum.nc"))

    if not instant_files or not accum_files:
        raise FileNotFoundError(
            f"Aucun NetCDF dans {RAW_DIR}.\n"
            "Lancez d'abord scripts/03_download_era5_full.py"
        )

    print(f"Chargement de {len(instant_files)} fichiers instantanes...")
    ds_inst = xr.open_mfdataset(instant_files, combine="by_coords")
    print(f"Chargement de {len(accum_files)} fichiers accumules...")
    ds_acc = xr.open_mfdataset(accum_files, combine="by_coords")

    ds = xr.merge([ds_inst, ds_acc], compat="override")
    print(f"Dataset fusionne : {dict(ds.sizes)}")
    return ds


def convert_units(ds: xr.Dataset) -> xr.Dataset:
    """K -> Celsius pour t2m, d2m ; m -> mm pour tp."""
    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] - 273.15
        ds["t2m"].attrs["units"] = "degC"
    if "d2m" in ds:
        ds["d2m"] = ds["d2m"] - 273.15
        ds["d2m"].attrs["units"] = "degC"
    if "tp" in ds:
        # ERA5 tp = cumul de precipitation depuis le pas precedent.
        # Avec time=12:00 UTC sur le produit horaire, c'est mm sur l'heure
        # ecoulee, PAS le total quotidien. Documente dans qualite_donnees.md.
        ds["tp"] = ds["tp"] * 1000  # m -> mm
        ds["tp"].attrs["units"] = "mm/h (12 UTC)"
        ds["tp"].attrs["long_name"] = "Total precipitation, hourly accum at 12 UTC"
    return ds


def interpolate_missing(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolation lineaire des trous le long de l'axe temporel pour les
    variables instantanees (t2m, d2m). Pour la precipitation, un trou
    correspond a "pas de pluie sur ce pas" -> on remplit par 0.
    """
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    for v in ("t2m", "d2m"):
        if v in ds and ds[v].isnull().any():
            ds[v] = ds[v].interpolate_na(dim=time_coord, method="linear")
    if "tp" in ds:
        ds["tp"] = ds["tp"].fillna(0.0)
    return ds


def add_relative_humidity(ds: xr.Dataset) -> xr.Dataset:
    """
    Calcule l'humidite relative (%) via Magnus-Tetens.
    RH = 100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))
    avec T et Td en degres C.
    """
    T = ds["t2m"]
    Td = ds["d2m"]
    rh = 100.0 * (
        np.exp((17.625 * Td) / (243.04 + Td))
        / np.exp((17.625 * T) / (243.04 + T))
    )
    rh = rh.clip(0, 100)
    rh.attrs = {"units": "%", "long_name": "Relative humidity (Magnus-Tetens)"}
    ds["rh"] = rh
    return ds


def quality_report(ds: xr.Dataset) -> pd.DataFrame:
    """Rapport NA + statistiques + bornes IQR."""
    rows = []
    for vname in ds.data_vars:
        var = ds[vname]
        n_total = int(var.size)
        n_nan = int(var.isnull().sum().values)
        q1 = float(var.quantile(0.25).values)
        q3 = float(var.quantile(0.75).values)
        iqr = q3 - q1
        rows.append({
            "variable": vname,
            "units": var.attrs.get("units", ""),
            "n_total": n_total,
            "n_nan": n_nan,
            "pct_nan": round(100 * n_nan / n_total, 3),
            "min": float(var.min().values),
            "mean": float(var.mean().values),
            "max": float(var.max().values),
            "q1": q1,
            "q3": q3,
            "iqr_low": q1 - 1.5 * iqr,
            "iqr_high": q3 + 1.5 * iqr,
        })
    return pd.DataFrame(rows)


def aggregate_spatial_yearly(ds: xr.Dataset) -> pd.DataFrame:
    """
    Conserve la dimension spatiale : moyenne annuelle par (year, lat, lon)
    pour chaque variable. Utilise pour les cartes du dashboard.
    """
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    yearly = ds.groupby(f"{time_coord}.year").mean(dim=time_coord)
    df = yearly.to_dataframe().reset_index()
    keep = ["year", "latitude", "longitude", "t2m", "d2m", "rh", "tp"]
    return df[[c for c in keep if c in df.columns]]


def aggregate_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Agrege spatialement (moyenne sur la zone) et exporte un dataframe
    quotidien avec colonnes : date, zone, t2m, d2m, rh, tp.
    """
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"

    # Zone 1 : Mediterranee complete (toute la grille)
    med = ds.mean(dim=["latitude", "longitude"])
    df_med = med.to_dataframe().reset_index()
    df_med["zone"] = "mediterranee"

    # Zone 2 : Tunisie (rive sud)
    tun = ds.sel(**TUNISIA_BOX).mean(dim=["latitude", "longitude"])
    df_tun = tun.to_dataframe().reset_index()
    df_tun["zone"] = "tunisie"

    # Zone 3 : Europe meridionale (rive nord)
    eur = ds.sel(**EUROPE_BOX).mean(dim=["latitude", "longitude"])
    df_eur = eur.to_dataframe().reset_index()
    df_eur["zone"] = "europe"

    df = pd.concat([df_med, df_tun, df_eur], ignore_index=True)
    df = df.rename(columns={time_coord: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map({
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    })

    keep = ["date", "year", "month", "season", "zone",
            "t2m", "d2m", "rh", "tp"]
    return df[[c for c in keep if c in df.columns]]


def main() -> None:
    print("=" * 60)
    print("Preprocessing ERA5 1980-2024")
    print("=" * 60)

    ds = load_all_years()
    print("\n--- Conversion d'unites ---")
    ds = convert_units(ds)
    print("\n--- Interpolation des NA ---")
    ds = interpolate_missing(ds)
    print("\n--- Calcul humidite relative ---")
    ds = add_relative_humidity(ds)

    print("\n--- Rapport qualite ---")
    qrep = quality_report(ds)
    print(qrep.to_string(index=False))
    qrep.to_csv(PROCESSED_DIR / "quality_report.csv", index=False)

    print("\n--- Agregation spatiale (Mediterranee + Tunisie) ---")
    df = aggregate_to_dataframe(ds)
    print(f"Lignes finales : {len(df):,}")
    print(df.head())

    out = PROCESSED_DIR / "era5_daily.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSauvegarde : {out}")
    print(f"Taille     : {out.stat().st_size / 1e6:.1f} MB")

    print("\n--- Agregation spatiale annuelle (pour cartes) ---")
    df_sp = aggregate_spatial_yearly(ds)
    out_sp = PROCESSED_DIR / "era5_spatial_yearly.parquet"
    df_sp.to_parquet(out_sp, index=False)
    print(f"Lignes : {len(df_sp):,}")
    print(f"Sauvegarde : {out_sp}")
    print(f"Taille     : {out_sp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
