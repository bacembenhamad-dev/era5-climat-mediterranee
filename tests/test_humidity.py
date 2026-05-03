"""
Tests unitaires pour les fonctions de preprocessing.

Verifie :
- RH(T = Td) ~= 100 % (saturation)
- RH bornee dans [0, 100]
- detect_heatwaves : episodes consecutifs corrects

Lancement :
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from preprocessing import add_relative_humidity  # noqa: E402
from dashboard import detect_heatwaves           # noqa: E402


def _make_ds(t_celsius, td_celsius):
    """Cree un mini-Dataset xarray avec t2m et d2m en degC."""
    t = np.asarray(t_celsius, dtype=float)
    td = np.asarray(td_celsius, dtype=float)
    coords = {"time": np.arange(len(t))}
    return xr.Dataset(
        {"t2m": ("time", t), "d2m": ("time", td)},
        coords=coords,
    )


def test_rh_saturation():
    """A T = Td, l'air est sature -> RH ~= 100 %."""
    ds = _make_ds([20.0, 0.0, 35.0], [20.0, 0.0, 35.0])
    ds = add_relative_humidity(ds)
    np.testing.assert_allclose(ds["rh"].values, 100.0, atol=1e-6)


def test_rh_in_bounds():
    """RH doit toujours etre dans [0, 100], meme avec des entrees extremes."""
    ds = _make_ds([40.0, 30.0, 10.0, -5.0],
                  [-10.0, 30.0, 5.0, -5.0])
    ds = add_relative_humidity(ds)
    rh = ds["rh"].values
    assert (rh >= 0).all() and (rh <= 100).all()


def test_rh_dry_air():
    """Td << T -> RH faible."""
    ds = _make_ds([30.0], [0.0])
    ds = add_relative_humidity(ds)
    assert ds["rh"].values[0] < 30.0


def test_heatwave_detection_single_episode():
    """Detecte 1 episode de 4 jours au-dessus du seuil."""
    daily = np.array([20, 25, 31, 32, 33, 34, 28, 25])
    events = detect_heatwaves(daily, threshold=30, min_duration=3)
    assert len(events) == 1
    assert events[0] == (2, 4)


def test_heatwave_below_min_duration():
    """Episode de 2 jours -> ignore (min_duration = 3)."""
    daily = np.array([20, 31, 32, 25, 20])
    events = detect_heatwaves(daily, threshold=30, min_duration=3)
    assert events == []


def test_heatwave_two_episodes():
    """Deux episodes distincts."""
    daily = np.array([31, 32, 33, 25, 20, 31, 32, 35, 36])
    events = detect_heatwaves(daily, threshold=30, min_duration=3)
    assert len(events) == 2
    assert events[0] == (0, 3)
    assert events[1] == (5, 4)
