"""
01_download_era5_test.py
========================

Script de TEST : télécharge 1 seul mois (janvier 2024) de données ERA5
pour la Tunisie + Méditerranée afin de valider la chaîne CDS -> NetCDF -> xarray
avant de lancer le téléchargement complet 1980-2024.

Variables (3, comme exigé par le cahier des charges) :
    - 2m_temperature           (température à 2 m)
    - total_precipitation      (précipitations totales)
    - 2m_dewpoint_temperature  (point de rosée -> humidité relative dérivée)

Zone : 28 N – 46 N  /  -10 E – 40 E  (Tunisie centrée, Méditerranée incluse)

Note : CDS retourne un ZIP contenant deux NetCDF (variables instantanées
       et accumulées). On le télécharge tel quel puis on l'extrait.

Sorties :
    data/raw/era5_test_jan2024.zip                     (zip brut renvoyé par CDS)
    data/raw/era5_test_jan2024/instant.nc              (T2m, dewpoint)
    data/raw/era5_test_jan2024/accum.nc                (precipitation)
"""

from pathlib import Path
import zipfile
import cdsapi

# --- Configuration ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_FILE = OUTPUT_DIR / "era5_test_jan2024.zip"
EXTRACT_DIR = OUTPUT_DIR / "era5_test_jan2024"

DATASET = "reanalysis-era5-single-levels"

REQUEST = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_temperature",
        "total_precipitation",
        "2m_dewpoint_temperature",
    ],
    "year": ["2024"],
    "month": ["01"],
    "day": [f"{d:02d}" for d in range(1, 32)],
    # Échantillonnage 4× par jour suffit pour un test (00, 06, 12, 18 UTC)
    "time": ["00:00", "06:00", "12:00", "18:00"],
    # Zone : Tunisie + Méditerranée occidentale et centrale
    # Format CDS : [North, West, South, East]
    "area": [46, -10, 28, 40],
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrait le ZIP CDS et renomme les NetCDF en instant.nc / accum.nc."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            print(f"  -> {info.filename} ({info.file_size/1e6:.2f} MB)")
            z.extract(info, dest_dir)

    # Renommage standardisé pour faciliter la suite
    for nc in dest_dir.glob("*instant*.nc"):
        target = dest_dir / "instant.nc"
        if nc != target:
            nc.replace(target)
    for nc in dest_dir.glob("*accum*.nc"):
        target = dest_dir / "accum.nc"
        if nc != target:
            nc.replace(target)


def main() -> None:
    if ZIP_FILE.exists():
        size_mb = ZIP_FILE.stat().st_size / 1e6
        print(f"Zip deja present : {ZIP_FILE} ({size_mb:.1f} MB)")
        print("Extraction...")
    else:
        print(f"Telechargement : {DATASET}")
        print(f"Variables      : {', '.join(REQUEST['variable'])}")
        print(f"Periode        : {REQUEST['year'][0]}-{REQUEST['month'][0]}")
        print(f"Zone           : N={REQUEST['area'][0]} W={REQUEST['area'][1]} "
              f"S={REQUEST['area'][2]} E={REQUEST['area'][3]}")
        print(f"Sortie         : {ZIP_FILE}")
        print("-" * 60)
        print("Requete mise en file d'attente CDS (peut prendre 1-15 min)...")
        print("-" * 60)

        client = cdsapi.Client()
        client.retrieve(DATASET, REQUEST, str(ZIP_FILE))

        size_mb = ZIP_FILE.stat().st_size / 1e6
        print(f"\nOK. Zip telecharge : {ZIP_FILE} ({size_mb:.1f} MB)")
        print("Extraction du contenu...")

    extract_zip(ZIP_FILE, EXTRACT_DIR)
    print(f"\nFichiers extraits dans : {EXTRACT_DIR}")
    for nc in sorted(EXTRACT_DIR.glob("*.nc")):
        size_mb = nc.stat().st_size / 1e6
        print(f"  {nc.name:20s} {size_mb:>6.2f} MB")


if __name__ == "__main__":
    main()
