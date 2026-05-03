"""
02_inspect_era5_test.py
=======================

Charge les NetCDF ERA5 (instantanes + accumules), les fusionne, inspecte la
structure (dimensions, variables, unites), affiche un resume statistique
et trace un apercu rapide.

A executer apres 01_download_era5_test.py.
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "era5_test_jan2024"
INSTANT_NC = EXTRACT_DIR / "instant.nc"
ACCUM_NC = EXTRACT_DIR / "accum.nc"


def load_dataset() -> xr.Dataset:
    """Charge instant + accum et les fusionne sur un meme axe temporel."""
    if not INSTANT_NC.exists() or not ACCUM_NC.exists():
        raise FileNotFoundError(
            f"NetCDF manquants dans {EXTRACT_DIR}.\n"
            "Lancez d'abord scripts/01_download_era5_test.py"
        )

    print(f"Lecture instant : {INSTANT_NC.name}")
    ds_inst = xr.open_dataset(INSTANT_NC)
    print(f"  Variables : {list(ds_inst.data_vars)}")

    print(f"Lecture accum   : {ACCUM_NC.name}")
    ds_acc = xr.open_dataset(ACCUM_NC)
    print(f"  Variables : {list(ds_acc.data_vars)}")

    # Fusion : memes coordonnees lat/lon/time, on merge les data_vars
    ds = xr.merge([ds_inst, ds_acc], compat="override")
    return ds


def main() -> None:
    ds = load_dataset()

    print("\n=== Structure du dataset fusionne ===")
    print(ds)

    print("\n=== Variables ===")
    for name, var in ds.data_vars.items():
        units = var.attrs.get("units", "?")
        long_name = var.attrs.get("long_name", name)
        print(f"  {name:30s} | {long_name:40s} | unites = {units}")

    print("\n=== Dimensions ===")
    for dim, size in ds.sizes.items():
        print(f"  {dim:20s} = {size}")

    # Detection automatique du nom de l'axe temporel
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    print(f"\n=== Etendue temporelle (coord='{time_coord}') ===")
    t = ds[time_coord]
    print(f"  De  : {t.min().values}")
    print(f"  A   : {t.max().values}")
    print(f"  Pas : {len(t)} pas de temps")

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    print(f"\n=== Etendue spatiale ===")
    print(f"  Latitudes  : {ds[lat_name].min().values:.2f} -> "
          f"{ds[lat_name].max().values:.2f}  ({len(ds[lat_name])} points)")
    print(f"  Longitudes : {ds[lon_name].min().values:.2f} -> "
          f"{ds[lon_name].max().values:.2f}  ({len(ds[lon_name])} points)")

    # --- Statistiques rapides par variable ---------------------------------
    print("\n=== Statistiques (sur l'ensemble du mois et de la zone) ===")
    for vname, var in ds.data_vars.items():
        units = var.attrs.get("units", "")
        print(f"  {vname:30s} | min={float(var.min()):>9.3f} "
              f"| mean={float(var.mean()):>9.3f} "
              f"| max={float(var.max()):>9.3f}  ({units})")

    # --- Apercu visuel : moyenne du mois pour chaque variable -------------
    print("\nGeneration de l'apercu visuel...")
    var_names = list(ds.data_vars.keys())
    fig, axes = plt.subplots(1, len(var_names), figsize=(5 * len(var_names), 4))
    if len(var_names) == 1:
        axes = [axes]

    for ax, vname in zip(axes, var_names):
        mean_field = ds[vname].mean(dim=time_coord)
        # Conversion T en degres Celsius pour l'affichage si en Kelvin
        units = ds[vname].attrs.get("units", "")
        if units == "K":
            mean_field = mean_field - 273.15
            cb_label = "deg C"
        else:
            cb_label = units

        mean_field.plot(ax=ax, cbar_kwargs={"label": cb_label})
        ax.set_title(f"{vname}\nmoyenne janvier 2024")

    fig.suptitle("Apercu ERA5 - Tunisie & Mediterranee (test)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_png = PROJECT_ROOT / "data" / "raw" / "era5_test_preview.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Apercu sauvegarde : {out_png}")


if __name__ == "__main__":
    main()
