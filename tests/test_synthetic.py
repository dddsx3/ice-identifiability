"""Regression tests pinned to the redistributable fixtures and the two defects.

1. A product without a declared thickness uncertainty must end in
   ``not_assessable`` and must still write a parseable ``gates.json``.
   Before the fix the run died with ``TypeError: Object of type bool is not
   JSON serializable`` (a ``numpy.bool_`` in the G3 threshold payload) and,
   once that was fixed, G3 reported a smooth sigma-less field as
   ``unidentifiable`` - a structure verdict that requires sigma.
2. Since the v4 contract G3 uses only sigma-free quantities, so a sigma-less
   field now gets a G3 *verdict* (that is the point of the change); the run
   still ends ``not_assessable`` because G4/G5 cannot be evaluated without a
   declared uncertainty.  The sigma-normalised block excess stays ``None``.
3. The white-noise fixture must stay ``unidentifiable``.
4. The synthetic alpha must come back from the tool (the error is reported by
   the assertion message, not hidden).
5. The G3 block is parameterised in kilometres: 200 km is 8 cells on the
   25 km fixture (the original calibration), and a grid that cannot realise
   200 km must refuse the G3 verdict instead of applying a constant that was
   calibrated at another scale.

The fixtures are shipped in ``examples/data``.  If they are absent the same
generator is run into a temporary directory, so the tests never depend on a
restricted product.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset, date2num

from ice_identifiability.structure import block_cells_for, cell_extent_km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = os.path.join(ROOT, "examples", "data")
ALPHA_TOL = 0.01


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory) -> str:
    truth = os.path.join(SHIPPED, "synth_truth.json")
    if os.path.exists(truth):
        return SHIPPED
    out = str(tmp_path_factory.mktemp("fixtures"))
    subprocess.run([sys.executable, os.path.join(ROOT, "examples",
                                                 "make_synthetic_fixtures.py"),
                    "--out", out], check=True, env=_env())
    return out


def _audit(fixtures: str, tmp_path, name: str, extra=()) -> tuple[int, dict]:
    out = os.path.join(str(tmp_path), name)
    cmd = [sys.executable, "-m", "ice_identifiability", "run",
           "--thickness", os.path.join(fixtures, name + ".nc"),
           "--out", out, "--label", name, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    with open(os.path.join(out, "gates.json"), encoding="utf-8") as f:
        return proc.returncode, json.load(f)


def _gate(result: dict, gid: str) -> dict:
    return next(g for g in result["gates"] if g["id"] == gid)


def test_no_sigma_product_is_not_assessable_and_writes_gates(fixtures, tmp_path):
    code, res = _audit(fixtures, tmp_path, "synth_nosigma_grid")
    assert code == 2, res["verdict"]
    assert res["verdict"] == "not_assessable"
    # G3 is sigma-free since the v4 contract: a sigma-less field can be screened
    g3 = _gate(res, "G3")
    assert g3["status"] == "pass", g3["reason"]
    assert np.isfinite(res["structure"]["block_excess_free_primary"])
    assert np.isfinite(g3["inputs"]["block_excess_free_primary"])
    # ... but the sigma-normalised layer is withheld, including the G5 sub-check
    assert res["structure"]["block_excess_primary"] is None
    assert _gate(res, "G5")["inputs"]["sigma_as_error_scale"] == "not_assessable"
    # the structure that *is* measurable must survive into the record
    assert np.isfinite(res["structure"]["nu_lag1"])
    assert np.isfinite(res["structure"]["nu_lag8"])
    assert res["bands"] == {}
    assert res["n_rows_alpha"] == 0
    assert len(res["gates"]) == 9
    # gates.json must be strict JSON: no NaN / Infinity tokens, only null
    with open(os.path.join(str(tmp_path), "synth_nosigma_grid", "gates.json"),
              encoding="utf-8") as f:
        json.loads(f.read(),
                   parse_constant=lambda tok: pytest.fail(f"non-strict token {tok}"))


def test_white_noise_fixture_stays_unidentifiable(fixtures, tmp_path):
    code, res = _audit(fixtures, tmp_path, "synth_white_noise",
                       ("--raw-channel", "posterior"))
    assert code == 3
    assert res["verdict"] == "unidentifiable"
    assert _gate(res, "G3")["status"] == "fail"
    assert res["structure"]["nu_lag1"] > 0.9
    assert res["structure"]["block_excess_primary"] < 1.3
    assert res["bands"] == {}


def test_synthetic_alpha_is_recovered(fixtures, tmp_path):
    with open(os.path.join(fixtures, "synth_truth.json"), encoding="utf-8") as f:
        truth = json.load(f)

    code, res = _audit(fixtures, tmp_path, "synth_sigma_grid",
                       ("--raw-channel", "posterior"))
    assert code == 0, res["verdict_reason"]
    assert res["verdict"] == "assessable"
    errs = {}
    for band, want in truth["alpha_true_band"].items():
        got = res["bands"][band]["alpha_median"]
        errs[band] = abs(got - want)
        assert errs[band] < ALPHA_TOL, (band, got, want)

    code, res_fb = _audit(fixtures, tmp_path, "synth_sigma_grid",
                          ("--raw-channel", "freeboard", "--label", "freeboard"))
    assert code == 0
    assert res_fb["inputs"]["alpha_object"] == "nuisance_share_of_freeboard_channel"
    for band, want in truth["alpha_coarse_analytic"]["freeboard"].items():
        got = res_fb["bands"][band]["alpha_median"]
        errs[band + "/freeboard"] = abs(got - want)
        assert errs[band + "/freeboard"] < ALPHA_TOL, (band, got, want)

    # the recovery is algebraic (the generator inverts the same budget), so it
    # is a wiring test, not a validation of the priors; say so in the message
    print("alpha recovery error (tool vs truth):",
          {k: f"{v:.2e}" for k, v in errs.items()})


def test_g3_block_is_parameterised_in_km(fixtures, tmp_path):
    with Dataset(os.path.join(fixtures, "synth_sigma_grid.nc")) as ds:
        lat = np.asarray(ds.variables["lat"][:])
        lon = np.asarray(ds.variables["lon"][:])
    n, dx, dy = block_cells_for(lat, lon, None, 200.0)
    assert n == 8                       # the published 8 x 8 calibration
    assert 0.75 <= (dx * n) / 200.0 <= 1.25
    assert 0.75 <= (dy * n) / 200.0 <= 1.25
    assert (dx, dy) == pytest.approx((24.5, 25.0), abs=0.6)

    # a coarse lat/lon grid cannot realise 200 km with >= 2 cells
    glat = np.linspace(55.0, 85.0, 13)
    glon = np.arange(0.0, 360.0, 2.5)
    blat, blon = np.meshgrid(glat, glon, indexing="ij")
    n2, dx2, dy2 = block_cells_for(blat, blon, None, 200.0)
    assert n2 == 2
    assert (dy2 * n2) / 200.0 > 1.25

    # end to end: the same geometry must refuse the G3 verdict, not fail it
    path = os.path.join(str(tmp_path), "coarse.nc")
    shape = (1,) + blat.shape
    # a smooth latitude-dependent field: nu is defined, the block is not
    hcoarse = 1.5 + 0.6 * np.cos(np.radians(blat)) ** 2
    with Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("yc", blat.shape[0])
        ds.createDimension("xc", blat.shape[1])
        v = ds.createVariable("lat", "f8", ("yc", "xc")); v[:] = blat
        v = ds.createVariable("lon", "f8", ("yc", "xc")); v[:] = blon
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2018-01-01 00:00:00"
        t[:] = date2num([__import__("datetime").datetime(2018, 4, 15)], t.units)
        for name, val, std in (("sea_ice_thickness", hcoarse, "sea_ice_thickness"),
                               ("sea_ice_thickness_uncertainty", 0.5,
                                "sea_ice_thickness standard_error"),
                               ("snow_depth", 0.2,
                                "surface_snow_thickness_where_sea_ice"),
                               ("snow_depth_uncertainty", 0.05,
                                "surface_snow_thickness_where_sea_ice standard_error")):
            v = ds.createVariable(name, "f4", ("time", "yc", "xc"))
            v.standard_name = std
            v.units = "m"
            v[:] = np.broadcast_to(val, shape).astype("f4")

    out = os.path.join(str(tmp_path), "coarse_run")
    proc = subprocess.run([sys.executable, "-m", "ice_identifiability", "run",
                           "--thickness", path, "--out", out,
                           "--raw-channel", "posterior"],
                          capture_output=True, text=True, env=_env())
    with open(os.path.join(out, "gates.json"), encoding="utf-8") as f:
        res = json.load(f)
    assert proc.returncode == 2
    assert res["verdict"] == "not_assessable"
    g3 = _gate(res, "G3")
    assert g3["status"] == "not_assessable"
    assert "cannot realise" in g3["reason"]
    assert g3["inputs"]["block_scale_realised"] is False
