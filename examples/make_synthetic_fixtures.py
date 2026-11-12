"""Generate the redistributable synthetic fixtures used by the paper.

Creates four small NetCDF products on a 60 x 60 / 25 km polar grid, each with
12 monthly steps, plus ``synth_truth.json``.  The products cover the three
input paths of the audit:

  synth_sigma_grid.nc     thickness + declared sigma + snow + snow sigma
                          + freeboard + freeboard sigma  (fully assessable)
  synth_nosigma_grid.nc   thickness + snow, no declared sigma (structure only)
  synth_nosnow_grid.nc    thickness + declared sigma, no snow (alpha withheld)
  synth_white_noise.nc    white-noise thickness (must be unidentifiable)

Ground truth is analytic: the declared sigma is chosen so that the variance
budget of SPEC.md has a prescribed nuisance share ``alpha_true`` in each
latitude band.  The fields are generated at 6.25 km and block-averaged to
25 km, so the coarse product carries a measurable aggregation effect; both the
band truth and the coarse-grained analytic alpha are recorded.

Nothing here reads a restricted product.  Everything is reproducible from the
fixed seed.  The projection is a spherical azimuthal equidistant grid centred
on the pole with 25.000 km spacing (EASE2-like geometry, not the official
EPSG:6931 grid).

Usage:
    python make_synthetic_fixtures.py --out examples/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
from netCDF4 import Dataset, date2num
from scipy.ndimage import gaussian_filter

SEED = 20261108
FINE = 4                     # 4 x 4 fine cells -> 1 product cell
CELL_KM = 25.0
NX = NY = 60
Y0_KM = 2223.0               # window centre north of the pole offset
SIGMA_DECLARED = 0.9         # m, constant declared 1-sigma thickness error
# Freeboard uncertainty of the fixture.  It is set to SIGMA_DECLARED / KAPPA so
# that the two declared channels describe the *same* 0.9 m thickness error: the
# freeboard channel must then return the same alpha as the posterior channel,
# which makes the fixture a test of object invariance rather than of a
# difference.  (The CCI product does not do this -- its 0.01 m freeboard
# uncertainty implies 0.1 m of thickness error against a declared 0.38 m -- and
# that inconsistency is what G8 refuses; see Section 5.)
SIGMA_FREEBOARD = SIGMA_DECLARED / (1027.0 / (1027.0 - 910.0))
ALPHA_TRUE = {"60-70": 0.45, "70-80": 0.68}
# Spatial structure of the thickness field, in *fine* cells (6.25 km).  The
# long component sets the correlation length that the block-variance test
# sees; the short one is the grid-scale texture that G6 compares with sigma.
H_SD_LONG, H_LEN_FINE = 0.36, 10.0
H_SD_TEXTURE, H_LEN_TEXTURE_FINE = 0.05, 1.5
HS_LEN_FINE = 7.0

# priors, copied from constants.py and re-implemented here on purpose: the
# generator must not import the code it is supposed to cross-check.
RHO_W, RHO_I, RHO_S = 1027.0, 910.0, 320.0
SIG_RHO_I, SIG_RHO_S = 30.0, 60.0
N0 = np.sqrt(1.9)
EPS_SD = 0.5 / np.sqrt(3.0)
KAPPA = RHO_W / (RHO_W - RHO_I)
DEFF = RHO_W - RHO_I
SNOW_SENS = KAPPA * (N0 - 1.0) + RHO_S / DEFF


def make_grid():
    jj, ii = np.meshgrid(np.arange(NX), np.arange(NY))
    x = (jj - (NX - 1) / 2.0) * CELL_KM
    y = Y0_KM + (ii - (NY - 1) / 2.0) * CELL_KM
    r = np.hypot(x, y)
    lat = 90.0 - np.degrees(r / 6371.0088)
    lon = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return lat, lon


def band_of(lat):
    out = np.full(lat.shape, "", dtype=object)
    out[(lat >= 60) & (lat < 70)] = "60-70"
    out[(lat >= 70) & (lat < 80)] = "70-80"
    out[lat >= 80] = "80+"
    return out


def grf(rng, shape, sigma_cells, sd):
    z = rng.normal(size=shape)
    z = gaussian_filter(z, sigma_cells, mode="reflect")
    z = z / z.std()
    return sd * z


def alpha_to_sigma_snow(h, hs, alpha, a):
    """Invert the budget: choose sigma_hs so that alpha is exactly ``alpha``."""
    post2 = a ** 2 / (1.0 - alpha)
    floor2 = post2 - a ** 2
    c2 = (h / DEFF * SIG_RHO_I) ** 2 + (hs / DEFF * SIG_RHO_S) ** 2
    d2 = (EPS_SD * KAPPA * (N0 - 1.0) * hs) ** 2
    b2 = floor2 - c2 - d2
    if np.any(b2 <= 0):
        raise SystemExit("alpha infeasible with the chosen a/h/hs envelope")
    return np.sqrt(b2) / SNOW_SENS


def alpha_of(h, hs, sig_hs, a):
    b2 = (SNOW_SENS * sig_hs) ** 2
    c2 = (h / DEFF * SIG_RHO_I) ** 2 + (hs / DEFF * SIG_RHO_S) ** 2
    d2 = (EPS_SD * KAPPA * (N0 - 1.0) * hs) ** 2
    return 1.0 - a ** 2 / (a ** 2 + b2 + c2 + d2)


def block_mean(fine):
    return fine.reshape(NY, FINE, NX, FINE).mean(axis=(1, 3))


def month_fields(rng, lat2d, month):
    """One month of fine-scale thickness and snow, aggregated to 25 km.

    The thickness field is the sum of a coherent component with a ~90 km
    correlation length (the scale the G3 block test is calibrated on) and a
    small grid-scale texture; the seasonal cycle moves the band mean by 6 cm.
    """
    z24 = (240, 240)
    phase = 2.0 * np.pi * (month - 1) / 12.0
    h_fine = (1.52 + 0.06 * np.cos(phase)
              + grf(rng, z24, H_LEN_FINE, H_SD_LONG)
              + grf(rng, z24, H_LEN_TEXTURE_FINE, H_SD_TEXTURE))
    hs_fine = 0.24 + grf(rng, z24, HS_LEN_FINE, 0.035)
    h_fine = np.clip(h_fine, 1.00, 2.10)
    hs_fine = np.clip(hs_fine, 0.10, 0.38)

    band_coarse = band_of(lat2d)
    band_fine = np.repeat(np.repeat(band_coarse, FINE, axis=0), FINE, axis=1)
    a = SIGMA_DECLARED
    sig_hs_fine = np.zeros_like(h_fine)
    for name, alpha in ALPHA_TRUE.items():
        sel = band_fine == name
        if sel.any():
            sig_hs_fine[sel] = alpha_to_sigma_snow(h_fine[sel], hs_fine[sel], alpha, a)

    h = block_mean(h_fine)
    hs = block_mean(hs_fine)
    sig_hs = block_mean(sig_hs_fine)
    alpha_fine_truth = {k: v for k, v in ALPHA_TRUE.items()}
    return h, hs, sig_hs, alpha_fine_truth


def write_product(path, lat, lon, months, fields, title, summary):
    ny, nx = lat.shape
    with Dataset(path, "w") as ds:
        ds.createDimension("time", len(months))
        ds.createDimension("yc", ny)
        ds.createDimension("xc", nx)
        ds.title = title
        ds.summary = summary
        ds.license = "CC-BY-4.0"
        ds.Conventions = "CF-1.8"
        ds.source = "synthetic: make_synthetic_fixtures.py (seed %d)" % SEED
        ds.comment = ("Synthetic product. Values are not observations of any real "
                      "sea-ice field; the declared sigma encodes a known nuisance "
                      "share (see synth_truth.json).")

        v = ds.createVariable("lat", "f8", ("yc", "xc"))
        v.units = "degrees_north"
        v.standard_name = "latitude"
        v[:] = lat
        v = ds.createVariable("lon", "f8", ("yc", "xc"))
        v.units = "degrees_east"
        v.standard_name = "longitude"
        v[:] = lon

        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2018-01-01 00:00:00"
        t.standard_name = "time"
        t.calendar = "standard"
        t[:] = date2num([__import__("datetime").datetime(2018, m, 15) for m in months],
                        t.units, t.calendar)

        for name, (data, units, std, long) in fields.items():
            v = ds.createVariable(name, "f4", ("time", "yc", "xc"))
            v.units = units
            v.standard_name = std
            v.long_name = long
            v[:] = data.astype("float32")
        # sigma semantics are declared machine-readably (G5)
        if "sea_ice_thickness_uncertainty" in fields:
            ds.variables["sea_ice_thickness_uncertainty"].comment = (
                "declared 1-sigma thickness uncertainty (synthetic, standard error)")
        if "snow_depth_uncertainty" in fields:
            ds.variables["snow_depth_uncertainty"].comment = (
                "1-sigma snow-depth uncertainty (synthetic, standard error)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(SEED)
    lat, lon = make_grid()
    bands = band_of(lat)
    months = list(range(1, 13))

    H, HS, SHS, FB, FBU, CONC = [], [], [], [], [], []
    for m in months:
        h, hs, sig_hs, _ = month_fields(rng, lat, m)
        fb = 0.114 * h + grf(rng, h.shape, 1.5, 0.02)
        conc = np.clip(90.0 + grf(rng, h.shape, 2.0, 4.0), 40.0, 100.0)
        H.append(h); HS.append(hs); SHS.append(sig_hs); FB.append(fb)
        FBU.append(np.full_like(h, SIGMA_FREEBOARD)); CONC.append(conc)

    H, HS, SHS = map(lambda x: np.stack(x), (H, HS, SHS))
    FB, FBU, CONC = map(lambda x: np.stack(x), (FB, FBU, CONC))

    # white-noise control: same grid, same declared sigma, no spatial structure
    rng_noise = np.random.default_rng(SEED + 1)
    WN = np.stack([1.25 + rng_noise.normal(0.0, 0.25, lat.shape) for _ in months])
    WNS = np.stack([np.full(lat.shape, 0.25) for _ in months])
    WNHS = np.stack([np.full(lat.shape, 0.24) for _ in months])
    WNSS = np.stack([np.full(lat.shape, 0.03) for _ in months])

    snow = {
        "snow_depth": (HS, "m", "surface_snow_thickness_where_sea_ice",
                       "synthetic snow depth on sea ice"),
        "snow_depth_uncertainty": (SHS, "m",
                                   "surface_snow_thickness_where_sea_ice standard_error",
                                   "sigma derived so that the budget has a known alpha"),
    }
    full = {
        "sea_ice_thickness": (H, "m", "sea_ice_thickness", "synthetic sea-ice thickness"),
        "sea_ice_thickness_uncertainty": (
            np.full_like(H, SIGMA_DECLARED), "m", "sea_ice_thickness standard_error",
            "declared 1-sigma thickness uncertainty (synthetic)"),
        **snow,
        "radar_freeboard": (FB, "m", "sea_ice_freeboard", "synthetic radar freeboard"),
        "radar_freeboard_uncertainty": (FBU, "m", "sea_ice_freeboard standard_error",
                                        "declared freeboard uncertainty (synthetic)"),
        "sea_ice_concentration": (CONC, "%", "sea_ice_concentration",
                                  "synthetic ice concentration"),
    }
    nosigma = {
        "sea_ice_thickness": (H, "m", "sea_ice_thickness", "synthetic sea-ice thickness"),
        **snow,
        "sea_ice_concentration": (CONC, "%", "sea_ice_concentration",
                                  "synthetic ice concentration"),
    }
    nosnow = {
        "sea_ice_thickness": (H, "m", "sea_ice_thickness", "synthetic sea-ice thickness"),
        "sea_ice_thickness_uncertainty": (
            np.full_like(H, SIGMA_DECLARED), "m", "sea_ice_thickness standard_error",
            "declared 1-sigma thickness uncertainty (synthetic)"),
        "sea_ice_concentration": (CONC, "%", "sea_ice_concentration",
                                  "synthetic ice concentration"),
    }
    noise = {
        "sea_ice_thickness": (WN, "m", "sea_ice_thickness", "white-noise control field"),
        "sea_ice_thickness_uncertainty": (
            WNS, "m", "sea_ice_thickness standard_error", "declared sigma (control)"),
        "snow_depth": (WNHS, "m", "surface_snow_thickness_where_sea_ice",
                       "constant snow depth (control)"),
        "snow_depth_uncertainty": (WNSS, "m",
                                   "surface_snow_thickness_where_sea_ice standard_error",
                                   "constant snow sigma (control)"),
        "sea_ice_concentration": (np.full_like(WN, 90.0), "%", "sea_ice_concentration",
                                  "constant concentration (control)"),
    }

    files = {
        "synth_sigma_grid.nc": (full, "synthetic product, fully assessable path"),
        "synth_nosigma_grid.nc": (nosigma, "synthetic product, no declared sigma"),
        "synth_nosnow_grid.nc": (nosnow, "synthetic product, no snow field"),
        "synth_white_noise.nc": (noise, "white-noise control"),
    }
    for name, (fields, title) in files.items():
        path = os.path.join(args.out, name)
        write_product(path, lat, lon, months, fields, title,
                      "ice-identifiability synthetic fixture")

    # ---------------------------------------------------------------- truth
    a_post = SIGMA_DECLARED
    a_fb = KAPPA * SIGMA_FREEBOARD
    alpha_coarse = {ch: {} for ch in ("posterior", "freeboard")}
    for name in ("60-70", "70-80"):
        sel = (bands == name)
        if not sel.any():
            continue
        alpha_coarse["posterior"][name] = float(
            np.median(alpha_of(H[:, sel], HS[:, sel], SHS[:, sel], a_post)))
        alpha_coarse["freeboard"][name] = float(
            np.median(alpha_of(H[:, sel], HS[:, sel], SHS[:, sel], a_fb)))

    # realised structure of the shipped field (cheap, same definition as the tool)
    h0 = H[0]
    valid = np.isfinite(h0)
    d = (h0[:, 1:] - h0[:, :-1])[valid[:, 1:] & valid[:, :-1]]
    nu = float(np.median(d ** 2) / 2.0 / np.nanvar(h0[valid]))

    truth = {
        "seed": SEED,
        "grid": {"shape": [NY, NX], "cell_km": CELL_KM,
                 "projection": "spherical azimuthal equidistant, pole-centred",
                 "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))]},
        "alpha_true_band": ALPHA_TRUE,
        "alpha_coarse_analytic": alpha_coarse,
        "sigma_declared_m": SIGMA_DECLARED,
        "sigma_freeboard_m": SIGMA_FREEBOARD,
        "generation": {"fine_cell_km": CELL_KM / FINE,
                       "h_correlation_sigma_fine_cells": H_LEN_FINE,
                       "h_correlation_length_km_approx":
                           float(np.sqrt(2.0) * H_LEN_FINE * CELL_KM / FINE),
                       "h_sd_coherent_m": H_SD_LONG,
                       "h_sd_texture_m": H_SD_TEXTURE,
                       "seasonal_amplitude_m": 0.06},
        "nu_lag1_first_month": nu,
        "h_field": {"mean_m": float(np.mean(H)), "sd_m": float(np.std(H)),
                    "min_m": float(np.min(H)), "max_m": float(np.max(H))},
        "note": ("alpha_true_band is exact by construction at the 6.25 km generation "
                 "scale; alpha_coarse_analytic is the same identity applied to the "
                 "shipped 25 km fields, so tool-vs-coarse differences measure "
                 "aggregation, not algebra."),
        "files": {name: sha256(os.path.join(args.out, name)) for name in files},
    }
    with open(os.path.join(args.out, "synth_truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, sort_keys=True)

    total = sum(os.path.getsize(os.path.join(args.out, n)) for n in files)
    print(json.dumps({k: truth[k] for k in
                      ("alpha_true_band", "alpha_coarse_analytic",
                       "nu_lag1_first_month", "h_field")}, indent=2))
    print(f"wrote {len(files)} fixtures, {total / 1e6:.2f} MB total, to {args.out}")


if __name__ == "__main__":
    main()
