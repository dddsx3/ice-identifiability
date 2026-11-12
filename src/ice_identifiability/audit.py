"""Orchestration: compute -> write CSV/JSON -> write report. No plotting."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import __version__
from .budget import BudgetPolicy, BudgetResult, band_of, compute_budget
from .constants import PRIOR_RANGES, Priors, outside_envelope
from .gates import evaluate_gates
from .ncio import NotAssessable, Product, _check_same_grid, sha256_of
from .report import render
from .structure import block_cells_for, reference_report, structure_report

_BANDS = ("60-70", "70-80", "80+")

# The table schema is part of the output contract (SPEC.md 6.1/6.2).  Every
# path that writes these two files must use the same header, including the
# "nothing was computed" path: two schemas under one filename fail silently
# for a consumer that reads by column position, which is worse than crashing.
ALPHA_COLUMNS = ["month", "iy", "ix", "lat", "lon", "band", "h_m",
                 "sigma_declared_m", "sigma_raw_m", "alpha_object", "alpha",
                 "alpha_status", "sigma_post_m", "sigma_floor_m",
                 "a_m", "b_m", "c_m", "d_m"]
BUDGET_COLUMNS = ["month", "iy", "ix", "band", "alpha_object", "var_a", "var_b",
                  "var_c", "var_d", "var_total", "share_a", "share_b",
                  "share_c", "share_d"]


def write_empty_tables(outdir: str) -> None:
    """Write header-only alpha.csv / budget.csv with the contract schema."""
    pd.DataFrame(columns=ALPHA_COLUMNS).to_csv(
        os.path.join(outdir, "alpha.csv"), index=False)
    pd.DataFrame(columns=BUDGET_COLUMNS).to_csv(
        os.path.join(outdir, "budget.csv"), index=False)


def _jsonable(obj):
    """Replace non-finite floats with null so gates.json is strict JSON.

    numpy scalars used to reach json.dump unchanged (a numpy.bool_ raised
    ``TypeError: Object of type bool is not JSON serializable``), and NaN was
    emitted as the bare token ``NaN``, which strict parsers reject.  A missing
    number is now ``null``: an absent value, never a zero.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _obs_mask(product: Product, policy: BudgetPolicy) -> tuple[np.ndarray, np.ndarray]:
    h = product.h.data * product.h.scale
    with np.errstate(invalid="ignore"):
        m = np.isfinite(h) & (h >= policy.min_thickness) & (h <= policy.max_thickness)
        if product.concentration is not None:
            c = product.concentration.data
            m &= np.isfinite(c) & (c > policy.min_concentration)
    return h, m


def _bands_summary(budget: BudgetResult, h: np.ndarray, lat: np.ndarray,
                   band: np.ndarray) -> dict:
    out = {}
    for b in _BANDS:
        sel = budget.mask & (band == b)
        if sel.sum() < 50:
            continue
        alpha = budget.alpha[sel]
        sp = budget.sigma_post[sel]
        sf = budget.sigma_floor[sel]
        with np.errstate(invalid="ignore", divide="ignore"):
            g8_ratio = budget.comps["a"][sel] / sf
        out[b] = {
            "n": int(sel.sum()),
            "alpha_median": float(np.nanmedian(alpha)),
            "alpha_p10": float(np.nanpercentile(alpha, 10)),
            "alpha_p90": float(np.nanpercentile(alpha, 90)),
            "sigma_post_median": float(np.nanmedian(sp)),
            "sigma_floor_median": float(np.nanmedian(sf)),
            "h_median": float(np.nanmedian(h[sel])),
            # The quantity G8 decides on, recorded per band so the decision can
            # be re-derived from the record without the fields: median(f(x)) =
            # f(median(x)) for the monotone f(s) = sqrt(s/(1-s)), so this equals
            # the ratio implied by share_a to the last bit.
            "g8_ratio_median": float(np.nanmedian(g8_ratio)),
            **{f"share_{k}": float(np.nanmedian(budget.shares[k][sel]))
               for k in ("a", "b", "c", "d")},
        }
    return out


def _write_alpha_csv(path, product, budget, band, alpha_status):
    rows = []
    h = product.h.data * product.h.scale
    sig = product.sigma.data * product.sigma.scale if product.sigma is not None else None
    lat, lon = product.grid.lat, product.grid.lon
    for t, month in enumerate(product.time_labels):
        m = budget.mask[t]
        if not m.any():
            continue
        iy, ix = np.where(m)
        rec = pd.DataFrame({
            "month": month, "iy": iy, "ix": ix,
            "lat": lat[iy, ix], "lon": lon[iy, ix], "band": band[iy, ix],
            "h_m": h[t][iy, ix],
            "sigma_declared_m": (sig[t][iy, ix] if sig is not None else np.nan),
            "sigma_raw_m": budget.sigma_raw[t][iy, ix],
            "alpha_object": budget.alpha_object,
            "alpha": (budget.alpha[t][iy, ix] if alpha_status == "assessable" else np.nan),
            "alpha_status": alpha_status,
            "sigma_post_m": budget.sigma_post[t][iy, ix],
            "sigma_floor_m": budget.sigma_floor[t][iy, ix],
            **{f"{k}_m": budget.comps[k][t][iy, ix] for k in ("a", "b", "c", "d")},
        })
        rows.append(rec)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=ALPHA_COLUMNS)
    df.to_csv(path, index=False, float_format="%.6g")
    return int(len(df))


def _write_budget_csv(path, product, budget, band):
    rows = []
    for t, month in enumerate(product.time_labels):
        m = budget.mask[t]
        if not m.any():
            continue
        iy, ix = np.where(m)
        comps = {k: budget.comps[k][t][iy, ix] for k in ("a", "b", "c", "d")}
        rec = pd.DataFrame({
            "month": month, "iy": iy, "ix": ix, "band": band[iy, ix],
            "alpha_object": budget.alpha_object,
            **{f"var_{k}": comps[k] ** 2 for k in comps},
            "var_total": budget.sigma_post[t][iy, ix] ** 2,
            **{f"share_{k}": budget.shares[k][t][iy, ix] for k in comps},
        })
        rows.append(rec)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=BUDGET_COLUMNS)
    df.to_csv(path, index=False, float_format="%.6g")
    return int(len(df))


def run_audit(product: Product, priors: Priors, policy: BudgetPolicy,
              outdir: str, *, command: str = "",
              reference: Product | None = None) -> dict:
    # A prior outside the documented envelope cannot be checked against the
    # literature, so no identifiability number is issued (SPEC.md 7).  The
    # refusal names the offending quantity: "a prior is out of range" is not
    # actionable.
    outside = outside_envelope(priors)
    if outside:
        detail = "; ".join(
            f"{n}={getattr(priors, n):g} not in "
            f"[{PRIOR_RANGES[n][0]:g}, {PRIOR_RANGES[n][1]:g}]" for n in outside)
        raise NotAssessable(f"priors outside PRIOR_RANGES: {detail}", "G0")

    os.makedirs(outdir, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    h, obs_mask = _obs_mask(product, policy)
    sigma_declared = (product.sigma.data * product.sigma.scale
                      if product.sigma is not None else None)
    # G3's block test is defined in kilometres (200 km calibration), not in a
    # fixed number of cells; convert the target to cells for this grid.
    block_cells, dx_km, dy_km = block_cells_for(
        product.grid.lat, product.grid.lon, obs_mask, policy.block_km)
    stats = structure_report(h, obs_mask, sigma_declared, block_cells=block_cells,
                             lat=product.grid.lat, lon=product.grid.lon)
    stats["block_km_target"] = float(policy.block_km)
    band = band_of(product.grid.lat)

    budget = None
    budget_error = None
    try:
        budget = compute_budget(product, priors, policy)
    except NotAssessable as exc:
        budget_error = exc

    ref_stats = None
    if reference is not None:
        try:
            _check_same_grid(product.grid, reference.grid,
                             product.path, reference.path)
            rmask = np.isfinite(reference.h.data) & (reference.h.data > 0)
            n = min(reference.h.data.shape[0], h.shape[0])
            if reference.h.data.shape[0] != h.shape[0]:
                raise NotAssessable(
                    "reference has a different number of time steps; align months first", "G0")
            ref_stats = reference_report(h, reference.h.data * reference.h.scale,
                                         obs_mask, rmask)
            ref_stats["label"] = reference.label
            ref_stats["var"] = reference.h.name
        except NotAssessable as exc:
            ref_stats = {"status": "not_assessable", "reason": exc.reason}

    gate_res = evaluate_gates(product, budget, stats, ref_stats, priors, policy,
                             budget_error_reason=(budget_error.reason if budget_error else None))

    alpha_status = "assessable" if gate_res["verdict"] == "assessable" else gate_res["verdict"]
    n_alpha = 0
    n_budget = 0
    bands = {}
    if budget is not None:
        n_alpha = _write_alpha_csv(os.path.join(outdir, "alpha.csv"),
                                   product, budget, band, alpha_status)
        n_budget = _write_budget_csv(os.path.join(outdir, "budget.csv"),
                                     product, budget, band)
        if alpha_status == "assessable":
            bands = _bands_summary(budget, h, product.grid.lat, band)

    if budget is None:
        # keep the required output files present, explicitly empty, and with
        # the same header a populated run would produce
        write_empty_tables(outdir)

    inputs = {
        "product_path": os.path.abspath(product.path),
        "thickness_var": product.h.name,
        "thickness_units": product.h.units,
        "sigma_var": (product.sigma.name if product.sigma is not None else None),
        "sigma_units": (product.sigma.units if product.sigma is not None else None),
        "snow_var": (product.snow.name if product.snow is not None else None),
        "snow_sigma_source": (budget.sigma_snow_source if budget is not None else "not_assessable"),
        "raw_channel": (budget.raw_channel_used if budget is not None else "not_assessable"),
        "alpha_object": (budget.alpha_object if budget is not None else "not_assessable"),
        "months": (product.time_labels if len(product.time_labels) <= 12
                   else [product.time_labels[0], "...", product.time_labels[-1]]),
        "sha256": product.sha256,
        "sources": product.sources,
        "snow_source_path": product.sources.get("snow"),
        "unc_source_path": product.sources.get("thickness_unc"),
        "freeboard_source_path": product.sources.get("freeboard_unc"),
    }
    result = {
        "tool": "ice-identifiability", "version": __version__,
        "generated_utc": started,
        "label": product.label,
        "inputs": inputs,
        "policy": policy.__dict__,
        "priors": priors.to_dict(),
        "structure": stats,
        "reference": ref_stats,
        "gates": gate_res["gates"],
        "verdict": gate_res["verdict"],
        "verdict_reason": gate_res["verdict_reason"],
        "alpha_reason_code": gate_res.get("alpha_reason_code"),
        "claims_not_supported": gate_res["claims_not_supported"],
        "bands": bands,
        "n_rows_alpha": n_alpha, "n_rows_budget": n_budget,
        "budget_error": (budget_error.reason if budget_error else None),
        "outputs": {"alpha_csv": os.path.abspath(os.path.join(outdir, "alpha.csv")),
                    "budget_csv": os.path.abspath(os.path.join(outdir, "budget.csv")),
                    "gates_json": os.path.abspath(os.path.join(outdir, "gates.json")),
                    "report_md": os.path.abspath(os.path.join(outdir, "report.md"))},
    }

    with open(os.path.join(outdir, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=1, ensure_ascii=False,
                  allow_nan=False)

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render(product.label, result, command, outdir))

    manifest = {
        "generated_utc": started,
        "inputs": {k: v for k, v in inputs.items()},
        "outputs": {
            "alpha.csv": sha256_of(os.path.join(outdir, "alpha.csv")),
            "budget.csv": sha256_of(os.path.join(outdir, "budget.csv")),
            "gates.json": sha256_of(os.path.join(outdir, "gates.json")),
            "report.md": sha256_of(os.path.join(outdir, "report.md")),
        },
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return result
