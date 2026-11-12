"""Bootstrap error bars for alpha and the variance shares.

Two levels are reported, and they are not interchangeable:

  band-month   spatial block bootstrap (8x8 cells = 200 km blocks) over the
               valid cells of that band.  Blocks, not cells, are resampled:
               alpha is spatially correlated, so a cell bootstrap would
               understate the standard error.  A naive cell bootstrap is run
               alongside to show the inflation factor.
  gridpoint    there is no sampling CI for a single cell: alpha is a
               deterministic function of the declared inputs.  What can
               vary at a fixed cell is the prior assumption, so the gridpoint
               interval is the prior envelope from prior_sensitivity.py
               (`ci_type = prior_envelope`), not a bootstrap interval.

Also computed:
  - the minimum observed separation (0.2185) against its bootstrap SE, with
    Bonferroni correction for 9 band-month tests;
  - a 3-month consistency test (does the separation survive month-to-month
    variation, not just within one month?);
  - CCI 80+ interannual alpha (0.768 / 0.552 / 0.703): mean, sd, range,
    and whether a climatological band alpha can be reported at all.

Outputs (CSV/NPZ before the figure):
  out/stats/ci_band_month.csv
  out/stats/ci_separation.csv
  out/stats/ci_interannual.csv
  out/stats/ci_gridpoint.csv
  out/stats/ci_bootstrap_draws.npz
  out/stats/ci.md
  out/stats/ci.png
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ice_identifiability import BudgetPolicy, Priors, band_of, compute_budget, load_product  # noqa: E402
from ice_identifiability.ncio import overlay_fields  # noqa: E402

WORK = r"<WORKROOT>\hard_innovation\d2_first_principles"
CCI = os.path.join(WORK, "data", "cci")
CS2 = os.path.join(ROOT, "out", "derived", "cs2smos")
OUT = os.path.join(ROOT, "out", "stats")
ENVELOPE = os.path.join(ROOT, "out", "sensitivity", "gridpoint_envelope.npz")
BANDS = ("60-70", "70-80", "80+")
MONTHS = ["2018-04", "2019-04", "2020-04"]
POLICY = BudgetPolicy(raw_channel="posterior")
PRIORS = Priors()
BLOCK = 8
N_BOOT = 800
N_BOOT_NAIVE = 400
STATS = ("alpha", "share_a", "share_b", "share_c", "share_d")


def _pivot(values, mask, block):
    """Flat valid values plus the block index of each cell."""
    ny, nx = values.shape[0] // block, values.shape[1] // block
    by = np.arange(values.shape[0]) // block
    bx = np.arange(values.shape[1]) // block
    bid = (by[:, None] * nx + bx[None, :])
    vals = values[mask]
    ids = bid[mask]
    return vals, ids


def _boot(vals, ids, n_boot, rng, by_block=True):
    """Return n_boot medians.  Block resampling, or naive cell resampling."""
    if by_block:
        uniq, inv = np.unique(ids, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        inv_sorted = inv[order]
        split = np.searchsorted(inv_sorted, np.arange(uniq.size + 1))
        groups = [order[split[i]:split[i + 1]] for i in range(uniq.size)]
        k = uniq.size
        out = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, k, k)
            idx = np.concatenate([groups[p] for p in pick])
            out[i] = np.median(vals[idx])
        return out
    n = vals.size
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = np.median(vals[idx])
    return out


def _ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(draws, ddof=1))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cci = load_product(CCI, months=MONTHS, label="CCI")
    cs2 = load_product(CS2, months=MONTHS, label="CS2SMOS", need_snow=False)
    cs2 = overlay_fields(cs2, CCI, kinds=("snow", "snow_unc"), months=MONTHS)
    band = band_of(cci.grid.lat)

    rng = np.random.default_rng(20261107)
    rows, draws_store, sep_rows = [], {}, []
    for label, prod in (("CCI", cci), ("CS2SMOS_shared_snow", cs2)):
        res = compute_budget(prod, PRIORS, POLICY)
        for ti, month in enumerate(prod.time_labels):
            for b in BANDS:
                sel = res.mask[ti] & (band == b)
                if int(sel.sum()) < 200:
                    continue
                arrays = {"alpha": res.alpha[ti]}
                for k in ("a", "b", "c", "d"):
                    arrays[f"share_{k}"] = res.shares[k][ti]
                for stat in STATS:
                    vals, ids = _pivot(arrays[stat], sel, BLOCK)
                    vals = vals[np.isfinite(vals)]
                    if vals.size < 50:
                        continue
                    ids = ids[np.isfinite(arrays[stat][sel])]
                    point = float(np.median(vals))
                    blk = _boot(vals, ids, N_BOOT, rng, True)
                    nai = _boot(vals, ids, N_BOOT_NAIVE, rng, False)
                    lo, hi, se_b = _ci(blk)
                    lo_n, hi_n, se_n = _ci(nai)
                    n_blocks = int(np.unique(ids).size)
                    rows.append({
                        "product": label, "month": month, "band": b,
                        "alpha_object": res.alpha_object, "stat": stat,
                        "n_cells": int(vals.size), "n_blocks": n_blocks,
                        "point": point,
                        "boot_median": float(np.median(blk)),
                        "ci_low": lo, "ci_high": hi, "se_block": se_b,
                        "ci_low_naive": lo_n, "ci_high_naive": hi_n,
                        "se_naive": se_n,
                        "se_inflation": (se_b / se_n if se_n > 0 else np.nan),
                        "ci_type": "block_bootstrap_200km",
                    })
                    if stat == "alpha":
                        draws_store[f"alpha_{label}_{month}_{b}"] = blk
    ci = pd.DataFrame(rows)
    ci.to_csv(os.path.join(OUT, "ci_band_month.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "ci_bootstrap_draws.npz"), **draws_store)

    # ---- separation: CCI vs CS2, per band-month
    alpha = ci[ci.stat == "alpha"]
    for b in BANDS:
        for month in MONTHS:
            a = alpha[(alpha["product"] == "CCI") & (alpha.band == b) & (alpha.month == month)]
            c = alpha[(alpha["product"] == "CS2SMOS_shared_snow") & (alpha.band == b)
                      & (alpha.month == month)]
            if a.empty or c.empty:
                continue
            key_a = f"alpha_CCI_{month}_{b}"
            key_c = f"alpha_CS2SMOS_shared_snow_{month}_{b}"
            if key_a not in draws_store or key_c not in draws_store:
                continue
            da, dc = draws_store[key_a], draws_store[key_c]
            delta = float(c["point"].iloc[0] - a["point"].iloc[0])
            diff = dc - da
            se = float(np.std(diff, ddof=1))
            z = delta / se if se > 0 else np.nan
            p = float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else np.nan
            sep_rows.append({
                "band": b, "month": month,
                "alpha_cci": float(a["point"].iloc[0]),
                "alpha_cs2": float(c["point"].iloc[0]),
                "delta": delta,
                "ci_low": float(np.percentile(diff, 2.5)),
                "ci_high": float(np.percentile(diff, 97.5)),
                "se_diff": se, "z": z, "p_two_sided": p,
                "bonferroni_alpha": 0.05 / 9,
                "significant_bonferroni": bool(p < 0.05 / 9) if np.isfinite(p) else False,
            })
    sep = pd.DataFrame(sep_rows)
    sep.to_csv(os.path.join(OUT, "ci_separation.csv"), index=False)

    # ---- 3-month consistency of the separation
    cons_rows = []
    for b in BANDS:
        s = sep[sep.band == b]
        if len(s) < 2:
            continue
        m = float(s.delta.mean())
        sd = float(s.delta.std(ddof=1))
        t = m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan
        p = float(2 * sps.t.sf(abs(t), df=len(s) - 1)) if np.isfinite(t) else np.nan
        cons_rows.append({"band": b, "n_months": len(s),
                          "delta_mean": m, "delta_sd_across_months": sd,
                          "t": t, "p_two_sided": p,
                          "consistent": bool(p < 0.05) if np.isfinite(p) else False})
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(os.path.join(OUT, "ci_separation_consistency.csv"), index=False)

    # ---- interannual variability of the band alpha
    inter_rows = []
    for b in BANDS:
        for label in ("CCI", "CS2SMOS_shared_snow"):
            s = alpha[(alpha["product"] == label) & (alpha.band == b)].sort_values("month")
            if s.empty:
                continue
            vals = s["point"].values
            wr = {f"alpha_{m.replace('-', '_')}": float(v) for m, v in zip(s.month, vals)}
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            rng_ = float(np.max(vals) - np.min(vals))
            # rule adopted in SPEC: a climatological band value needs
            # range <= 0.05 across the three years
            reportable = bool(rng_ <= 0.05)
            inter_rows.append({"product": label, "band": b, "n_months": len(vals),
                               **wr, "mean": mean, "sd_across_months": sd,
                               "range": rng_,
                               "verdict": "climatology_ok" if reportable
                               else "per_month_only"})
    inter = pd.DataFrame(inter_rows)
    inter.to_csv(os.path.join(OUT, "ci_interannual.csv"), index=False)

    # ---- gridpoint: prior envelope (not a sampling interval)
    gp_rows = []
    if os.path.exists(ENVELOPE):
        z = np.load(ENVELOPE, allow_pickle=True)
        lat, lon, bandg = z["lat"], z["lon"], z["band"]
        for suffix, label in (("cci", "CCI"), ("cs2", "CS2SMOS_shared_snow")):
            mask = z[f"mask_{suffix}"]
            a = z[f"alpha_{suffix}"]
            lo, hi = z[f"alpha_low_{suffix}"], z[f"alpha_high_{suffix}"]
            sel = mask & np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
            gp_rows.append(pd.DataFrame({
                "product": label,
                "lat": np.broadcast_to(lat, a.shape)[sel],
                "lon": np.broadcast_to(lon, a.shape)[sel],
                "band": bandg[sel],
                "alpha": a[sel],
                "ci_low": lo[sel], "ci_high": hi[sel],
                "halfwidth": 0.5 * (hi[sel] - lo[sel]),
                "ci_type": "prior_envelope",
            }))
    if gp_rows:
        pd.concat(gp_rows, ignore_index=True).to_csv(
            os.path.join(OUT, "ci_gridpoint.csv"), index=False)

    # ---- figure (after CSV/NPZ)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"CCI": "#2b6cb0", "CS2SMOS_shared_snow": "#c53030"}
    for label in ("CCI", "CS2SMOS_shared_snow"):
        for b, ls in zip(BANDS, ("-", "--", ":")):
            s = alpha[(alpha["product"] == label) & (alpha.band == b)]
            if s.empty:
                continue
            x = [MONTHS.index(m) for m in s.month]
            y = s["point"].values
            ax.errorbar(x, y, yerr=[y - s.ci_low.values, s.ci_high.values - y],
                        fmt="o" + ls, color=colors[label], ms=3, lw=1,
                        capsize=2, label=f"{label} {b}" if label == "CCI" else None)
    ax.set_xticks(range(len(MONTHS)))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("band-median alpha (95% block-bootstrap CI)")
    ax.set_title("Blue = CCI, red = CS2; line styles = 60-70 / 70-80 / 80+")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ci.png"), dpi=150)
    plt.close(fig)

    # ---- summary
    min_sep = float(sep.delta.min()) if not sep.empty else np.nan
    min_row = sep.loc[sep.delta.idxmin()] if not sep.empty else None
    infl = float(ci["se_inflation"].median())
    cci80 = inter[(inter["product"] == "CCI") & (inter.band == "80+")]
    lines = ["# Bootstrap error bars", "",
             f"Block bootstrap: {N_BOOT} draws, {BLOCK}x{BLOCK} cells (200 km) per block, "
             f"percentile CI. Naive cell bootstrap: {N_BOOT_NAIVE} draws, for comparison only.",
             "",
             f"- median standard-error inflation (block / naive): **{infl:.2f}x**. "
             "Treating 25 km cells as independent understates the SE by this factor.",
             "", "## Separation CCI - CS2", "",
             "| band | month | alpha CCI | alpha CS2 | delta | 95% CI | z | p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sep.itertuples():
        lines.append(f"| {r.band} | {r.month} | {r.alpha_cci:.3f} | {r.alpha_cs2:.3f} | "
                     f"{r.delta:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.z:.1f} | "
                     f"{r.p_two_sided:.1e} |")
    lines += ["", f"- minimum observed separation: **{min_sep:.3f}** "
              f"({min_row.band}, {min_row.month}); z={min_row.z:.1f}, "
              f"Bonferroni-significant={bool(min_row.significant_bonferroni)} "
              f"(threshold 0.05/9).",
              "", "## Three-month consistency (does the separation survive months?)", "",
              "| band | mean delta | sd across months | t (df=2) | consistent |",
              "|---|---|---|---|---|"]
    for r in cons.itertuples():
        lines.append(f"| {r.band} | {r.delta_mean:.3f} | {r.delta_sd_across_months:.3f} | "
                     f"{r.t:.1f} | {bool(r.consistent)} |")
    lines += ["", "## Band alpha across the three years", "",
              "| product | band | 2018-04 | 2019-04 | 2020-04 | mean | sd | range | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in inter.itertuples():
        vals = [getattr(r, f"alpha_{m.replace('-', '_')}")
                for m in MONTHS if hasattr(r, f"alpha_{m.replace('-', '_')}")]
        pad = "".join(" | —" for _ in range(len(MONTHS) - len(vals)))
        lines.append(f"| {r.product} | {r.band} | " + " | ".join(f"{v:.3f}" for v in vals)
                     + pad + f" | {r.mean:.3f} | {r.sd_across_months:.3f} | "
                     f"{r.range:.3f} | {r.verdict} |")
    lines += ["", "Rule adopted: a band alpha may be reported as a multi-year value only "
              "if the range across the three months is <= 0.05. CCI 80+ fails this rule "
              f"(range {float(cci80['range'].iloc[0]):.3f}), so its band alpha is "
              "per-month only; 60-70 and 70-80 pass. This is written into SPEC section 7.",
              "", "## Gridpoint interval", "",
              "`ci_gridpoint.csv` carries the prior envelope (min/max alpha over the 1176 "
              "prior combinations), not a bootstrap interval. A single cell has no sampling "
              "error to bootstrap: alpha is deterministic given the declared inputs.",
              "", "Files: ci_band_month.csv, ci_separation.csv, "
              "ci_separation_consistency.csv, ci_interannual.csv, ci_gridpoint.csv, "
              "ci_bootstrap_draws.npz, ci.png.", ""]
    with open(os.path.join(OUT, "ci.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"min_separation": min_sep,
                      "min_band": str(min_row.band), "min_month": str(min_row.month),
                      "min_z": float(min_row.z),
                      "se_inflation_median": infl,
                      "interannual": inter[["product", "band", "mean", "sd_across_months",
                                            "range", "verdict"]].to_dict("records")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
