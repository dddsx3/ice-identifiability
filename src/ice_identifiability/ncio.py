"""CF-aware NetCDF reading for the identifiability audit.

Design rules (see SPEC.md):
  * never interpolate, never regrid;
  * never fill missing thickness;
  * unit conversion is explicit and recorded; unknown units -> NotAssessable;
  * a declared uncertainty of 0 is *invalid*, not "zero error".
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field

import netCDF4 as nc
import numpy as np


class NotAssessable(Exception):
    """Raised when a required, checkable input is missing or unverifiable."""

    def __init__(self, reason: str, gate: str = "G0"):
        super().__init__(reason)
        self.reason = reason
        self.gate = gate


_UNIT_SCALE = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
}

_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")

_KIND_CANDIDATES = {
    "thickness": ("sea_ice_thickness", "analysis_sea_ice_thickness",
                  "ice_thickness", "thickness"),
    "thickness_unc": ("sea_ice_thickness_uncertainty", "analysis_sea_ice_thickness_unc",
                      "sea_ice_thickness_unc", "ice_thickness_uncertainty",
                      "thickness_uncertainty"),
    "snow": ("snow_depth", "surface_snow_thickness_where_sea_ice", "snow_thickness"),
    "snow_unc": ("snow_depth_uncertainty", "snow_depth_unc",
                 "surface_snow_thickness_where_sea_ice_uncertainty"),
    "freeboard": ("radar_freeboard", "sea_ice_freeboard", "freeboard"),
    "freeboard_unc": ("radar_freeboard_uncertainty", "sea_ice_freeboard_uncertainty",
                      "freeboard_uncertainty", "freeboard_unc"),
    "concentration": ("sea_ice_concentration", "ice_concentration", "concentration"),
    "quality": ("quality_flag", "status_flag"),
}

_KIND_STDNAME = {
    "thickness": ("sea_ice_thickness",),
    "thickness_unc": ("sea_ice_thickness standard_error",),
    "snow": ("surface_snow_thickness_where_sea_ice",),
    "snow_unc": ("surface_snow_thickness_where_sea_ice standard_error",),
}


@dataclass
class Field:
    """A single variable on a (T, Y, X) grid with provenance."""
    name: str
    units: str
    scale: float
    data: np.ndarray          # (T, Y, X) float64, NaN = missing
    attrs: dict = field(default_factory=dict)


@dataclass
class Grid:
    lat: np.ndarray           # (Y, X)
    lon: np.ndarray
    y: np.ndarray | None = None
    x: np.ndarray | None = None
    crs: str = ""
    shape: tuple = (0, 0)


@dataclass
class Product:
    label: str
    path: str
    time_labels: list[str]
    grid: Grid
    h: Field
    sigma: Field | None
    snow: Field | None
    snow_unc: Field | None
    freeboard: Field | None
    freeboard_unc: Field | None
    concentration: Field | None
    quality: Field | None
    var_names: dict
    sha256: str
    sources: dict = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return self.h.data.shape[0] if self.h is not None else len(self.time_labels)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expand_paths(path: str, months: list[str] | None = None) -> list[str]:
    """A file, a directory of *.nc, or a glob -> sorted file list."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.nc")))
    elif any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    if not files:
        raise NotAssessable(f"no NetCDF files matched {path!r}")
    if months:
        keep = []
        for f in files:
            lab = _month_from_filename(f) or ""
            if lab in months:
                keep.append(f)
        if not keep:
            raise NotAssessable(
                f"none of the files in {path!r} match --months {months}")
        files = keep
    return files


def _month_from_filename(path: str) -> str | None:
    base = os.path.basename(path)
    m = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])(?!\d)", base)
    return f"{m.group(1)}-{m.group(2)}" if m else None


_CONC_SCALE = {
    "%": 1.0, "percent": 1.0, "percentage": 1.0,
    "1": 100.0, "fraction": 100.0, "dimensionless": 100.0,
}


def _unit_scale(var, kind: str = "") -> tuple[str, float]:
    raw = getattr(var, "units", None)
    if raw is None:
        raise NotAssessable(
            f"variable {var.name!r} has no units attribute; refusing to guess", "G0")
    key = str(raw).strip().lower().replace(" ", "")
    if kind in ("quality",):
        return str(raw), 1.0
    if kind == "concentration":
        if key in _CONC_SCALE:
            return str(raw), _CONC_SCALE[key]
        raise NotAssessable(
            f"concentration {var.name!r} has unrecognised units {raw!r}", "G0")
    if key not in _UNIT_SCALE:
        raise NotAssessable(
            f"variable {var.name!r} has unrecognised units {raw!r}", "G0")
    return str(raw), _UNIT_SCALE[key]


def _to_float(var, name: str) -> np.ndarray:
    """Read a variable into float64 with NaN for every masked/fill value."""
    raw = var[:]
    a = np.ma.filled(np.ma.asarray(raw, dtype="float64"), np.nan)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise NotAssessable(f"variable {name!r} has ndim={a.ndim}, expected 2 or 3", "G0")
    a[np.abs(a) > 1e20] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def _detect(ds, kind: str) -> str | None:
    names = list(_KIND_CANDIDATES.get(kind, ()))
    for cand in names:
        if cand in ds.variables:
            return cand
    # fall back to standard_name
    for pat in _KIND_STDNAME.get(kind, ()):
        for vname, var in ds.variables.items():
            sn = str(getattr(var, "standard_name", "")).strip().lower()
            if sn == pat:
                return vname
    return None


def _read_var(ds, kind: str, required: bool = False) -> Field | None:
    name = _detect(ds, kind)
    if name is None:
        if required:
            raise NotAssessable(
                f"no variable matching kind={kind!r} (tried {_KIND_CANDIDATES.get(kind)})", "G4")
        return None
    var = ds.variables[name]
    units, scale = _unit_scale(var, kind)
    attrs = {k: str(getattr(var, k)) for k in
             ("standard_name", "long_name", "comment", "description",
              "source") if hasattr(var, k)}
    return Field(name=name, units=units, scale=scale,
                 data=_to_float(var, name), attrs=attrs)


def _read_grid(ds) -> Grid:
    lat_name = next((n for n in _LAT_NAMES if n in ds.variables), None)
    lon_name = next((n for n in _LON_NAMES if n in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise NotAssessable("no lat/lon coordinate variables; cannot verify the grid", "G0")
    lat = np.ma.filled(np.ma.asarray(ds.variables[lat_name][:], dtype="float64"), np.nan)
    lon = np.ma.filled(np.ma.asarray(ds.variables[lon_name][:], dtype="float64"), np.nan)
    if lat.ndim != 2 or lon.ndim != 2:
        raise NotAssessable("only 2-D curvilinear lat/lon are supported (EASE2-like)", "G0")
    y = ds.variables.get("yc")
    x = ds.variables.get("xc")
    yv = np.asarray(y[:], dtype="float64") if y is not None else None
    xv = np.asarray(x[:], dtype="float64") if x is not None else None
    crs = str(getattr(ds.variables[lat_name], "grid_mapping", "") or "")
    return Grid(lat=lat, lon=lon, y=yv, x=xv, crs=crs, shape=lat.shape)


def _time_labels(ds, n_time: int, path: str) -> list[str]:
    labels = []
    if "time" in ds.variables:
        try:
            t = ds.variables["time"]
            dates = nc.num2date(t[:], t.units, only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
            for d in np.atleast_1d(dates):
                labels.append(f"{d.year:04d}-{d.month:02d}")
        except Exception:
            labels = []
    if len(labels) != n_time:
        fb = _month_from_filename(path)
        labels = [fb or f"t{i:03d}" for i in range(n_time)]
    return labels


def _check_same_grid(a: Grid, b: Grid, a_path: str, b_path: str) -> None:
    if a.shape != b.shape:
        raise NotAssessable(
            f"grid shapes differ: {a.shape} ({a_path}) vs {b.shape} ({b_path}); "
            "the tool never regrids", "G0")
    lat_ok = np.allclose(a.lat, b.lat, atol=1e-3, equal_nan=True)
    lon_ok = np.allclose(a.lon, b.lon, atol=1e-3, equal_nan=True)
    if not (lat_ok and lon_ok):
        lon_ok = np.allclose(np.mod(a.lon, 360.0), np.mod(b.lon, 360.0),
                             atol=1e-3, equal_nan=True)
        lat_ok = np.allclose(a.lat, b.lat, atol=1e-3, equal_nan=True)
    if not (lat_ok and lon_ok):
        raise NotAssessable(
            f"lat/lon differ between {a_path} and {b_path}; the tool never regrids", "G0")


def _open_series(paths: list[str], label: str, kinds: dict[str, bool],
                 require_thickness: bool = True) -> Product:
    """Open one or more files of the same product and stack them along time."""
    first = paths[0]
    with nc.Dataset(first) as ds:
        grid = _read_grid(ds)

    chunks: dict[str, list[np.ndarray]] = {k: [] for k in kinds}
    scales: dict[str, float] = {}
    units_d: dict[str, str] = {}
    attrs_d: dict[str, dict] = {}
    names_d: dict[str, str] = {}
    labels: list[str] = []

    for path in paths:
        with nc.Dataset(path) as ds:
            g = _read_grid(ds)
            _check_same_grid(grid, g, first, path)
            lab = _time_labels(ds, 1, path)
            n = 1
            for kind, required in kinds.items():
                fld = _read_var(ds, kind, required=required)
                if fld is None:
                    chunks[kind].append(None)  # type: ignore[arg-type]
                    continue
                n = fld.data.shape[0]
                if kind not in scales:
                    scales[kind], units_d[kind], attrs_d[kind] = fld.scale, fld.units, fld.attrs
                    names_d[kind] = fld.name
                elif names_d[kind] != fld.name:
                    raise NotAssessable(
                        f"variable name for {kind!r} changes across files "
                        f"({names_d[kind]!r} -> {fld.name!r})", "G0")
                chunks[kind].append(fld.data)
            labels.extend(lab if len(lab) == n else [f"t{i}" for i in range(n)])

    def stack(kind: str) -> Field | None:
        parts = chunks.get(kind)
        if not parts or parts[0] is None:
            return None
        if any(p is None for p in parts):
            raise NotAssessable(f"{kind!r} is missing in some files of the series", "G4")
        data = np.concatenate(parts, axis=0)
        return Field(name=names_d[kind], units=units_d[kind], scale=scales[kind],
                     data=data, attrs=attrs_d.get(kind, {}))

    h = stack("thickness")
    if require_thickness and h is None:
        raise NotAssessable("thickness variable not found", "G0")
    return Product(
        label=label,
        path=first if len(paths) == 1 else os.path.dirname(first),
        time_labels=labels,
        grid=grid,
        h=h,  # type: ignore[arg-type]
        sigma=stack("thickness_unc"),
        snow=stack("snow"),
        snow_unc=stack("snow_unc"),
        freeboard=stack("freeboard"),
        freeboard_unc=stack("freeboard_unc"),
        concentration=stack("concentration"),
        quality=stack("quality"),
        var_names=names_d,
        sha256=sha256_of(first) if len(paths) == 1 else "",
    )


def load_product(path: str, *, label: str | None = None,
                 months: list[str] | None = None,
                 need_snow: bool = False) -> Product:
    """Load a thickness product (file / directory / glob)."""
    if not os.path.exists(path) and not any(ch in path for ch in "*?["):
        raise NotAssessable(f"input path does not exist: {path}", "G0")
    paths = expand_paths(path, months)
    kinds = {
        "thickness": True,
        "thickness_unc": False,
        "snow": need_snow,
        "snow_unc": False,
        "freeboard": False,
        "freeboard_unc": False,
        "concentration": False,
        "quality": False,
    }
    prod = _open_series(paths, label or os.path.basename(path), kinds)
    if prod.h is None:
        raise NotAssessable("thickness variable not found", "G0")
    prod.sources["thickness"] = os.path.abspath(path)
    for k in kinds:
        if k != "thickness" and getattr(prod, k, None) is not None:
            prod.sources[k] = os.path.abspath(path)
    return prod


_OVERLAY_KINDS = ("thickness_unc", "snow", "snow_unc", "freeboard",
                  "freeboard_unc", "concentration")


def overlay_fields(product: Product, path: str,
                   kinds: tuple[str, ...] = _OVERLAY_KINDS,
                   months: list[str] | None = None) -> Product:
    """Attach fields read from another file (a user-declared assumption).

    Months are intersected, never interpolated.  Nothing is inferred: a kind
    that is absent in the supplier stays absent.
    """
    paths = expand_paths(path, months)
    sup = _open_series(paths, "supplier", {k: False for k in kinds},
                       require_thickness=False)
    _check_same_grid(product.grid, sup.grid, product.path, path)
    if sup.time_labels != product.time_labels:
        common = [m for m in product.time_labels if m in sup.time_labels]
        if not common:
            raise NotAssessable("supplier shares no months with the product", "G4")
        ih = [product.time_labels.index(m) for m in common]
        isup = [sup.time_labels.index(m) for m in common]
        product.time_labels = common
        for attr in ("h", "sigma", "snow", "snow_unc", "freeboard",
                     "freeboard_unc", "concentration", "quality"):
            fld = getattr(product, attr)
            if fld is not None:
                fld.data = fld.data[ih]
        for k in kinds:
            fld = getattr(sup, k)
            if fld is not None:
                setattr(product, k, Field(fld.name, fld.units, fld.scale,
                                          fld.data[isup], fld.attrs))
                product.var_names[k] = fld.name
                product.sources[k] = os.path.abspath(path)
    else:
        if sup.n_time != product.n_time:
            raise NotAssessable("supplier time length differs from the product", "G4")
        for k in kinds:
            fld = getattr(sup, k)
            if fld is not None:
                setattr(product, k, fld)
                product.var_names[k] = fld.name
                product.sources[k] = os.path.abspath(path)
    return product
