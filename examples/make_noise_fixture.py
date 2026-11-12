"""Build a *synthetic white-noise* product for the discrimination test.

This file is a TEST FIXTURE.  It exists only to prove that the auditing tool
returns `unidentifiable` for a field with no spatial structure.  It is
never used to produce a scientific result, and the tool itself never
synthesises thickness.

The fixture reuses a real product's grid and ocean mask so that every
downstream mask/gate behaves exactly as on the real data; only the
thickness values are replaced with white noise.
"""

from __future__ import annotations

import argparse
import datetime
import os

import netCDF4 as nc
import numpy as np

_FILL = np.float32(9.969209968386869e36)


def build(template: str, out: str, seed: int, mean: float,
          sigma: float, snow: float, snow_sigma: float) -> str:
    rng = np.random.default_rng(seed)
    with nc.Dataset(template) as ds:
        lat = np.asarray(ds["lat"][:], dtype="float64")
        lon = np.asarray(ds["lon"][:], dtype="float64")
        xc = np.asarray(ds["xc"][:], dtype="float64")
        yc = np.asarray(ds["yc"][:], dtype="float64")
        conc = np.ma.filled(np.ma.asarray(ds["sea_ice_concentration"][:], dtype="float64"),
                            np.nan).squeeze()
        h0 = np.ma.filled(np.ma.asarray(ds["sea_ice_thickness"][:], dtype="float64"),
                          np.nan).squeeze()
        label = os.path.basename(template)

    valid = np.isfinite(h0) & np.isfinite(conc) & (conc > 15)
    z = rng.standard_normal(lat.shape)
    h = np.clip(mean + sigma * z, 0.05, 6.0)
    h = np.where(valid, h, np.nan)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with nc.Dataset(out, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("yc", len(yc))
        ds.createDimension("xc", len(xc))
        v = ds.createVariable("time", "f8", ("time",))
        v.units = "days since 1970-01-01"
        v[:] = [nc.date2num(datetime.datetime(2019, 4, 15), v.units)]
        for name, data, units in (("xc", xc, "km"), ("yc", yc, "km")):
            vv = ds.createVariable(name, "f8", (name,))
            vv.units = units
            vv[:] = data
        ds.createVariable("lon", "f8", ("yc", "xc"))[:] = lon
        ds["lon"].units = "degrees_east"
        ds.createVariable("lat", "f8", ("yc", "xc"))[:] = lat
        ds["lat"].units = "degrees_north"

        def put(name, arr, units, standard_name=""):
            vv = ds.createVariable(name, "f4", ("time", "yc", "xc"),
                                   fill_value=_FILL, zlib=True)
            vv.units = units
            if standard_name:
                vv.standard_name = standard_name
            vv[:] = arr[None, ...]

        put("sea_ice_thickness", h, "m", "sea_ice_thickness")
        put("sea_ice_thickness_uncertainty", np.where(valid, sigma, np.nan), "m",
            "sea_ice_thickness standard_error")
        put("sea_ice_concentration", np.where(valid, conc, np.nan), "%",
            "sea_ice_area_fraction")
        # a complete nuisance set, so the budget *can* be built: the tool
        # must still refuse to publish alpha.
        put("snow_depth", np.where(valid, snow, np.nan), "m",
            "surface_snow_thickness_where_sea_ice")
        put("snow_depth_uncertainty", np.where(valid, snow_sigma, np.nan), "m",
            "surface_snow_thickness_where_sea_ice standard_error")

        ds.title = "SYNTHETIC white-noise fixture for ice-identifiability"
        ds.source = f"template={label}"
        ds.comment = ("random field with no spatial structure; used only to test "
                      "that the tool emits 'unidentifiable'")
        ds.history = (f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} "
                      f"examples/make_noise_fixture.py seed={seed}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--mean", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--snow", type=float, default=0.2)
    p.add_argument("--snow-sigma", type=float, default=0.05)
    a = p.parse_args()
    print(build(a.template, a.out, a.seed, a.mean, a.sigma, a.snow, a.snow_sigma))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
