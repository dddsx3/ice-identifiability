"""Does G3 transfer from one thickness product to another?

Implements the calibration behind A2-3 section 5.2 *inside* the toolkit, so the
claim can be reproduced with one command instead of a script that lives
outside the repository:

    python examples/g3_transfer.py --cci <CCI L3C dir> --cs2 <CS2SMOS dir> --out <dir>

The fields are restricted (CCI CryoSat-2 SIT v4.0 L3C, CS2SMOS v206) and are
not redistributed; the command fails with exit 2 and a clear message when the
directories are absent.  Nothing is downloaded and nothing is written outside
``--out``.

What it computes, per product x month x latitude band:

* ``nu1``     lag-1 nugget (sigma-free)
* ``be8``     the sigma-normalised block excess, kept for comparison only
* ``be8_free`` the sigma-free block excess that G3 now uses
* ``phi = be8_free/be8`` = mean(sigma^2)/Var(h), the G5 sub-check on real fields
* ``snr``     median(|h|/sigma)
* a shuffled null (spatial coherence destroyed, marginal preserved), median of
  ``--shuffles`` draws; the null locates the constants, it is not a criterion

and then counts the verdicts of the old three-clause G3 and of the sigma-free
two-clause G3.  Anything a constant cannot decide is reported as *not
assessable*, never as a failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from ice_identifiability import BudgetPolicy, band_of, load_product      # noqa: E402
from ice_identifiability.constants import (G3_BLOCK_EXCESS_MIN,         # noqa: E402
                                           G3_NU_MAX, G3_SNR_MIN)
from ice_identifiability.structure import (block_excess, block_excess_free,  # noqa: E402
                                           nugget_ratio, q_sigma, snr_median)

# The historical constant, defined on the *sigma-normalised* block excess be8.
# It is kept only so the old three-clause G3 can be counted side by side.  Never
# compare a be_free reading against it: on the calibration fields be_free = be8*w
# with w the white-noise fraction, so the two constants live on different scales
# (SPEC.md section 4.1).
BE8_MIN_OLD = 3.0

BANDS = ("60-70", "70-80", "80+")
BLOCK = 8           # 200 km on the 25 km real grids; unchanged in this round
MIN_CELLS = 2000    # below this a latitude band is not entered at all
MONTHS = ("2019-04", "2019-10", "2020-01", "2020-04")
# CS2SMOS is a weekly product: A2-3 used the first complete week of each month,
# which is the 15th for October 2019 (the 1st-7th is incomplete in that archive).
CS2_DAY = {"2019-04": "20190401", "2019-10": "20191015",
           "2020-01": "20200101", "2020-04": "20200401"}


def _first_week(cs2_dir: str, month: str):
    for datestr in (CS2_DAY.get(month), month.replace("-", "") + "01"):
        if not datestr:
            continue
        hits = sorted(glob.glob(os.path.join(cs2_dir, f"*{datestr}*.nc")))
        if hits:
            return hits[0]
    return None


def _load(cci_dir: str, cs2_dir: str, product: str, month: str):
    if product == "CCI":
        return load_product(cci_dir, months=[month], label="CCI")
    path = _first_week(cs2_dir, month)
    if path is None:
        raise FileNotFoundError(f"no CS2SMOS file for {month} in {cs2_dir}")
    return load_product(path, label="CS2SMOS")


def _slices(product):
    h = product.h.data * product.h.scale
    sig = (product.sigma.data * product.sigma.scale
           if product.sigma is not None else None)
    conc = product.concentration.data if product.concentration is not None else None
    policy = BudgetPolicy()
    base = np.isfinite(h)
    base &= (h >= policy.min_thickness) & (h <= policy.max_thickness)
    if conc is not None:
        base &= np.isfinite(conc) & (conc > policy.min_concentration)
    if sig is not None:
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    for t in range(h.shape[0]):
        yield h[t], base[t], (sig[t] if sig is not None else None)


def _med(values):
    """Median of the finite entries; NaN when nothing is finite."""
    arr = np.asarray(list(values), dtype="float64")
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _stats(h, mask, sig, rng, shuffles):
    if mask.sum() < MIN_CELLS:
        return None
    one_h, one_m, one_s = h[None], mask[None], (sig[None] if sig is not None else None)
    out = dict(n_valid=int(mask.sum()),
               nu1=nugget_ratio(one_h, one_m, 1),
               be8=block_excess(one_h, one_m, one_s, BLOCK),
               be8_free=block_excess_free(one_h, one_m, BLOCK),
               snr=snr_median(one_h, one_m, one_s),
               q_sigma=q_sigma(one_h, one_m, one_s))
    out["phi"] = (float(out["be8_free"] / out["be8"])
                  if np.isfinite(out["be8"]) and np.isfinite(out["be8_free"])
                  and out["be8"] > 0 else float("nan"))
    nu_null, be_null, be_free_null = [], [], []
    for _ in range(shuffles):
        hs = h.copy()
        hs[mask] = rng.permutation(h[mask])
        hs1 = hs[None]
        nu_null.append(nugget_ratio(hs1, one_m, 1))
        be_null.append(block_excess(hs1, one_m, one_s, BLOCK))
        be_free_null.append(block_excess_free(hs1, one_m, BLOCK))
    out["nu1_null"] = _med(nu_null)
    out["be8_null"] = _med(be_null)
    out["be8_free_null"] = _med(be_free_null)
    return out


def _pass(value, op, threshold):
    """True / False, or None when the statistic is not assessable."""
    if value is None or not np.isfinite(value):
        return None
    return bool(value <= threshold) if op == "<=" else bool(value >= threshold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cci", default=r"<WORKROOT>\hard_innovation\d2_first_principles\data\cci")
    ap.add_argument("--cs2", default=r"<WORKROOT>\hard_innovation\d2_first_principles\data\cs2smos")
    ap.add_argument("--out", default="out_g3_transfer")
    ap.add_argument("--months", nargs="+", default=list(MONTHS))
    ap.add_argument("--shuffles", type=int, default=6,
                    help="shuffled-null draws per (case, band); 0 disables")
    ap.add_argument("--seed", type=int, default=20261109)
    args = ap.parse_args(argv)

    for name, d in (("--cci", args.cci), ("--cs2", args.cs2)):
        if not os.path.isdir(d):
            print(f"[error] {name} is not a directory: {d}\n"
                  "        these fields are restricted and not redistributed; "
                  "pass the local paths explicitly", file=sys.stderr)
            return 2

    rng = np.random.default_rng(args.seed)
    rows = []
    for product in ("CCI", "CS2SMOS"):
        for month in args.months:
            try:
                p = _load(args.cci, args.cs2, product, month)
            except Exception as exc:                       # noqa: BLE001
                print(f"[skip] {product} {month}: {type(exc).__name__}: {exc}")
                continue
            band = band_of(p.grid.lat)
            per = {b: [] for b in BANDS + ("ALL",)}
            for h, mask, sig in _slices(p):
                for b in BANDS + ("ALL",):
                    m = mask if b == "ALL" else (mask & (band == b))
                    st = _stats(h, m, sig, rng, args.shuffles)
                    if st:
                        per[b].append(st)
            for b in BANDS + ("ALL",):
                if not per[b]:
                    continue
                agg = {k: _med([x[k] for x in per[b]])
                       for k in ("nu1", "be8", "be8_free", "phi", "snr", "q_sigma",
                                 "nu1_null", "be8_null", "be8_free_null")}
                agg["n_valid"] = int(np.median([x["n_valid"] for x in per[b]]))
                agg["product"], agg["month"], agg["band"] = product, month, b
                rows.append(agg)
                print(f"{product:8s} {month} {b:5s} "
                      f"nu={agg['nu1']:.3f} be8={agg['be8']:.2f} "
                      f"be8free={agg['be8_free']:.2f} phi={agg['phi']:.3f} "
                      f"snr={agg['snr']:.2f} q={agg['q_sigma']:.3f} "
                      f"n={agg['n_valid']}")

    if not rows:
        print("[error] no case could be evaluated", file=sys.stderr)
        return 2

    def count(fn):
        vals = [fn(r) for r in rows]
        return {"pass": sum(v is True for v in vals),
                "fail": sum(v is False for v in vals),
                "not_assessable": sum(v is None for v in vals),
                "of": len(vals)}

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "block_cells": BLOCK,
        "constants": {"nu_max": G3_NU_MAX,
                      "block_excess_min_be_free": G3_BLOCK_EXCESS_MIN,
                      # the historical sigma-normalised constant; never compare
                      # a be_free reading against it (be_free = be8 * w)
                      "block_excess_min_be8_old": BE8_MIN_OLD,
                      "snr_min": G3_SNR_MIN},
        "rows": len(rows),
        # the old three-clause G3 (sigma-normalised block excess and S/N)
        "g3_three_clause": {
            "nu": count(lambda r: _pass(r["nu1"], "<=", G3_NU_MAX)),
            "be": count(lambda r: _pass(r["be8"], ">=", BE8_MIN_OLD)),
            "snr": count(lambda r: _pass(r["snr"], ">=", G3_SNR_MIN)),
            "all_three": count(lambda r: (None if any(
                v is None for v in (_pass(r["nu1"], "<=", G3_NU_MAX),
                                    _pass(r["be8"], ">=", BE8_MIN_OLD),
                                    _pass(r["snr"], ">=", G3_SNR_MIN))) else
                (_pass(r["nu1"], "<=", G3_NU_MAX)
                 and _pass(r["be8"], ">=", BE8_MIN_OLD)
                 and _pass(r["snr"], ">=", G3_SNR_MIN)))),
        },
        # the sigma-free two-clause G3 that ships from the v4 contract on
        "g3_sigma_free": {
            "nu": count(lambda r: _pass(r["nu1"], "<=", G3_NU_MAX)),
            "be_free": count(lambda r: _pass(r["be8_free"], ">=", G3_BLOCK_EXCESS_MIN)),
            "all_two": count(lambda r: (None if any(
                v is None for v in (_pass(r["nu1"], "<=", G3_NU_MAX),
                                    _pass(r["be8_free"], ">=", G3_BLOCK_EXCESS_MIN))) else
                (_pass(r["nu1"], "<=", G3_NU_MAX)
                 and _pass(r["be8_free"], ">=", G3_BLOCK_EXCESS_MIN)))),
        },
        "sigma_free_range": [float(np.nanmin([r["be8_free"] for r in rows])),
                             float(np.nanmax([r["be8_free"] for r in rows]))],
        "sigma_normalised_range": [float(np.nanmin([r["be8"] for r in rows])),
                                   float(np.nanmax([r["be8"] for r in rows]))],
        "nu_range": [float(np.nanmin([r["nu1"] for r in rows])),
                     float(np.nanmax([r["nu1"] for r in rows]))],
        "median_by_product": {
            k: {prod: float(np.nanmedian([r[k] for r in rows if r["product"] == prod]))
                for prod in {r["product"] for r in rows}}
            for k in ("nu1", "be8", "be8_free", "phi", "snr", "q_sigma")},
        "phi_by_product": {
            prod: [float(round(r["phi"], 3)) for r in rows
                   if r["product"] == prod and np.isfinite(r["phi"])]
            for prod in {r["product"] for r in rows}},
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "g3_transfer.csv"), "w", encoding="utf-8") as f:
        cols = ["product", "month", "band", "block_cells", "n_valid", "nu1", "be8",
                "be8_free", "phi", "snr", "q_sigma", "nu1_null", "be8_null",
                "be8_free_null"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(args.out, "g3_transfer.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nwritten: {os.path.join(args.out, 'g3_transfer.csv')}, "
          f"{os.path.join(args.out, 'g3_transfer.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
