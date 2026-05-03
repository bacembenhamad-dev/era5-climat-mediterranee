"""
03_download_era5_full.py
========================

Telechargement COMPLET ERA5 pour la zone Tunisie + Mediterranee, 1980-2024.

Strategie : 1 fichier ZIP par annee (45 fichiers) pour etre robuste aux
interruptions reseau et permettre une reprise partielle sans tout
re-telecharger. CDS zippe lui-meme la reponse en 2 NetCDF (instant + accum).

Granularite : 1 valeur par jour (12:00 UTC) -> ~13 MB par annee, ~600 MB total.

Variables (3) :
    - 2m_temperature           (T en K)
    - 2m_dewpoint_temperature  (dewpoint en K -> permet humidite relative)
    - total_precipitation      (precipitations en m, accumulees sur l'heure)

Lancement :
    python scripts/03_download_era5_full.py

Reprise apres interruption : relancer simplement, les annees deja
telechargees seront sautees.
"""

from pathlib import Path
import time
import zipfile
import cdsapi

# --- Configuration ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5_full"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"

YEARS = list(range(1980, 2025))   # 1980 inclus, 2024 inclus
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = ["12:00"]                 # 1 valeur / jour, midi UTC
AREA = [46, -10, 28, 40]          # N, W, S, E

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
]


def request_for_year(year: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrait le zip CDS en instant.nc / accum.nc."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    for nc in dest_dir.glob("*instant*.nc"):
        nc.replace(dest_dir / "instant.nc")
    for nc in dest_dir.glob("*accum*.nc"):
        nc.replace(dest_dir / "accum.nc")


def download_year(client: cdsapi.Client, year: int) -> bool:
    """Telecharge 1 annee. Renvoie True si nouveau telechargement, False si cache."""
    year_dir = RAW_DIR / str(year)
    zip_file = RAW_DIR / f"era5_{year}.zip"
    instant_nc = year_dir / "instant.nc"
    accum_nc = year_dir / "accum.nc"

    if instant_nc.exists() and accum_nc.exists():
        print(f"  [{year}] deja extrait -> skip")
        return False

    if not zip_file.exists():
        print(f"  [{year}] telechargement...")
        client.retrieve(DATASET, request_for_year(year), str(zip_file))
        size_mb = zip_file.stat().st_size / 1e6
        print(f"  [{year}] OK ({size_mb:.1f} MB)")
    else:
        size_mb = zip_file.stat().st_size / 1e6
        print(f"  [{year}] zip deja la ({size_mb:.1f} MB)")

    print(f"  [{year}] extraction...")
    extract_zip(zip_file, year_dir)
    return True


def main() -> None:
    print("=" * 60)
    print("ERA5 1980-2024 - Tunisie & Mediterranee")
    print(f"Variables : {', '.join(VARIABLES)}")
    print(f"Annees    : {YEARS[0]} -> {YEARS[-1]} ({len(YEARS)} annees)")
    print(f"Heure     : {TIMES[0]} (1 valeur / jour)")
    print(f"Zone      : N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")
    print(f"Sortie    : {RAW_DIR}")
    print("=" * 60)

    client = cdsapi.Client()

    n_new, n_skip, n_fail = 0, 0, 0
    start = time.time()

    for year in YEARS:
        try:
            if download_year(client, year):
                n_new += 1
            else:
                n_skip += 1
        except Exception as e:
            n_fail += 1
            print(f"  [{year}] ERREUR : {type(e).__name__} - {e}")

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Termine en {elapsed/60:.1f} min")
    print(f"  Nouveaux : {n_new}")
    print(f"  Sautes   : {n_skip}")
    print(f"  Echecs   : {n_fail}")
    if n_fail:
        print("Relancez le script pour reprendre les annees en echec.")


if __name__ == "__main__":
    main()
