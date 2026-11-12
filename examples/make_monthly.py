"""Build a monthly composite from weekly CS2SMOS files.

This is a *temporal reduction* of observed values (mean of the weekly
analyses).  It does not interpolate, does not fill gaps, does not create
thickness where none was observed.  The provenance is written into the
output file's global attributes.

Usage:
  python examples/make_monthly.py --dir <weekly_dir> --month 2019-04 --out <file.nc>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

VARS = {
    "analysis_sea_ice_thickness": ("m", "sea_ice_thickness",
                                   "CryoSat-2/SMOS merged sea-ice thickness"),
    "analysis_sea_ice_thickness_unc": ("m", "", "analysis thickness uncertainty"),
    "sea_ice_concentration": ("%", "sea_ice_area_fraction", "sea-ice concentration"),
}
_FILL = np.float32(9.969209968386869e36)


def build(directory: str, month: str, out: str) -> str:
    yyyymm = month.replace("-", "")
    files = sorted(glob.glob(os.path.join(directory, f"*{yyyymm}*.nc")))
    if not files:
        raise SystemExit(f"no weekly files for {month} in {directory}")
    ds0 = nc.Dataset(files[0])
    lat = np.asarray(ds0["lat"][:], dtype="float64")
    lon = np.asarray(ds0["lon"][:], dtype="float64")
    xc = np.asarray(ds0["xc"][:], dtype="float64")
    yc = np.asarray(ds0["yc"][:], dtype="float64")
    ds0.close()

    acc = {k: np.zeros((1,) + lat.shape) for k in VARS}
    cnt = {k: np.zeros((1,) + lat.shape) for k in VARS}
    for f in files:
        with nc.Dataset(f) as ds:
            for k in VARS:
                if k not in ds.variables:
                    continue
                a = np.ma.filled(np.ma.asarray(ds[k][:], dtype="float64"), np.nan)
                ok = np.isfinite(a)
                acc[k][ok] += a[ok]
                cnt[k][ok] += 1.0

    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    with nc.Dataset(out, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("yc", len(yc))
        ds.createDimension("xc", len(xc))
        ds.createDimension("nv", 2)
        v = ds.createVariable("time", "f8", ("time",))
        v.units = "seconds since 1978-01-01 00:00:00"
        v.long_name = "time"
        v.standard_name = "time"
        v[:] = [nc.date2num(__import__("datetime").datetime(int(month[:4]),
                                                            int(month[5:]), 15),
                            v.units)]
        v = ds.createVariable("xc", "f8", ("xc",))
        v.units = "km"; v.standard_name = "projection_x_coordinate"; v[:] = xc
        v = ds.createVariable("yc", "f8", ("yc",))
        v.units = "km"; v.standard_name = "projection_y_coordinate"; v[:] = yc
        v = ds.createVariable("lon", "f8", ("yc", "xc"))
        v.units = "degrees_east"; v.standard_name = "longitude"; v[:] = lon
        v = ds.createVariable("lat", "f8", ("yc", "xc"))
        v.units = "degrees_north"; v.standard_name = "latitude"; v[:] = lat
        for k, (units, sname, lname) in VARS.items():
            data = np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), np.nan)
            data[:, lat < 0] = np.nan
            var = ds.createVariable(k, "f4", ("time", "yc", "xc"),
                                    fill_value=_FILL, zlib=True, complevel=2)
            var.units = units
            if sname:
                var.standard_name = sname
            var.long_name = lname
            var[:] = data
        ds.title = "Monthly composite of weekly CryoSat-2/SMOS sea-ice thickness"
        ds.source = "wisconsin L4 v206 weekly files"
        ds.comment = ("monthly mean of weekly analysis fields over the calendar month; "
                      "no interpolation, no gap filling, no synthetic thickness")
        ds.history = f"{__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())} ice-identifiability examples/make_monthly.py"
        ds.n_weeks = len(files)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--month", required=True, help="YYYY-MM")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = build(a.dir, a.month, a.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
