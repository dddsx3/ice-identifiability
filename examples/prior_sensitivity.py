"""Prior sensitivity of the audit verdict, alpha and the variance shares.

Sweep (all values inside the documented envelopes in constants.PRIOR_RANGES;
values outside are reported as `not_assessable` and kept out of the envelope):

    rho_i        880 .. 940 kg m-3   (7 values)
    rho_s        200 .. 400 kg m-3   (6 values)
    n0           1.30 .. 1.90        (7 values, dry-snow refractive index)
    eps_halfwidth 0.20 .. 0.50       (4 values, half-width of the uniform
                                      prior on n0; sd = hw / sqrt(3))

Products: CCI (declared posterior object) and CS2SMOS with the CCI snow base
(the only variant in which CS2 is assessable), both for 2019-04.

Outputs (CSV/NPZ before figures):
  out/sensitivity/sensitivity_runs.csv        one row per (combo, product, band)
  out/sensitivity/sensitivity_effect.csv      marginal effect of each parameter
  out/sensitivity/sensitivity_flips.csv       every ordering / status flip found
  out/sensitivity/sensitivity_verdict_months.csv  corner combos x 3 months
  out/sensitivity/gridpoint_envelope.npz      per-cell alpha min/max over combos
  out/sensitivity/sensitivity.md
  out/sensitivity/sensitivity.png
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ice_identifiability import BudgetPolicy, Priors, band_of, compute_budget, load_product  # noqa: E402
from ice_identifiability.constants import PRIOR_RANGES, outside_envelope  # noqa: E402
from ice_identifiability.ncio import overlay_fields  # noqa: E402

WORK = r"<WORKROOT>\hard_innovation\d2_first_principles"
CCI = os.path.join(WORK, "data", "cci")
CS2 = os.path.join(ROOT, "out", "derived", "cs2smos")
OUT = os.path.join(ROOT, "out", "sensitivity")
BANDS = ("60-70", "70-80", "80+")
POLICY = BudgetPolicy(raw_channel="posterior")
MONTHS = ["2018-04", "2019-04", "2020-04"]
MAIN_MONTH = "2019-04"

RHO_I = np.round(np.arange(880, 941, 10), 1)
RHO_S = np.round(np.arange(200, 401, 40), 1)
N0 = np.round(np.arange(1.30, 1.91, 0.10), 2)
EPS = np.round(np.arange(0.20, 0.51, 0.10), 2)


def _shares_medians(res, band, b):
    sel = res.mask[0] & (band == b)
    if int(sel.sum()) < 50:
        return None
    out = {"alpha": float(np.nanmedian(res.alpha[0][sel]))}
    for k in ("a", "b", "c", "d"):
        out[f"share_{k}"] = float(np.nanmedian(res.shares[k][0][sel]))
    out["ratio_raw_over_floor"] = float(
        np.nanmedian(res.sigma_raw[0][sel] / res.sigma_floor[0][sel]))
    out["n"] = int(sel.sum())
    out["alpha_cell_min"] = float(np.nanmin(res.alpha[0][sel]))
    out["alpha_cell_max"] = float(np.nanmax(res.alpha[0][sel]))
    return out


def _g8_status(ratio):
    if not np.isfinite(ratio):
        return "not_assessable"
    if ratio < 0.5:
        return "fail"
    if ratio < 1.0:
        return "warn"
    return "pass"


def _load_pair(months):
    cci = load_product(CCI, months=months, label="CCI")
    cs2 = load_product(CS2, months=months, label="CS2SMOS", need_snow=False)
    cs2 = overlay_fields(cs2, CCI, kinds=("snow", "snow_unc"), months=months)
    return cci, cs2


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cci, cs2 = _load_pair([MAIN_MONTH])
    band = band_of(cci.grid.lat)
    combos = [(ri, rs, n0, e) for ri in RHO_I for rs in RHO_S for n0 in N0 for e in EPS]

    rows = []
    env = {}
    for label, prod in (("CCI", cci), ("CS2SMOS_shared_snow", cs2)):
        env[label] = {"low": np.full(prod.h.data.shape[1:], np.inf),
                      "high": np.full(prod.h.data.shape[1:], -np.inf)}
    for ri, rs, n0, e in combos:
        pri = Priors(rho_i=ri, rho_s=rs, refraction_n0=n0, eps_halfwidth=e)
        bad = outside_envelope(pri)
        status = "assessable" if not bad else "not_assessable"
        for label, prod in (("CCI", cci), ("CS2SMOS_shared_snow", cs2)):
            res = compute_budget(prod, pri, POLICY)
            for b in BANDS:
                m = _shares_medians(res, band, b)
                if m is None:
                    continue
                rows.append({"rho_i": ri, "rho_s": rs, "n0": n0, "eps_halfwidth": e,
                             "product": label, "band": b,
                             "param_status": status,
                             **m, "G8": _g8_status(m["ratio_raw_over_floor"])})
            # per-cell envelope (only checkable combinations contribute)
            if status == "assessable":
                a = res.alpha[0]
                inner = np.isfinite(a)
                env[label]["low"] = np.where(inner, np.minimum(env[label]["low"], a), env[label]["low"])
                env[label]["high"] = np.where(inner, np.maximum(env[label]["high"], a), env[label]["high"])
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "sensitivity_runs.csv"), index=False)

    # ---- marginal effects
    eff_rows = []
    for param, values in (("rho_i", RHO_I), ("rho_s", RHO_S),
                          ("n0", N0), ("eps_halfwidth", EPS)):
        for v in values:
            sub = runs[np.isclose(runs[param], v)]
            for prod in sub["product"].unique():
                for b in BANDS:
                    s = sub[(sub["product"] == prod) & (sub.band == b)]
                    if s.empty:
                        continue
                    eff_rows.append({"param": param, "value": float(v), "product": prod,
                                     "band": b, "alpha_min": float(s.alpha.min()),
                                     "alpha_median": float(s.alpha.median()),
                                     "alpha_max": float(s.alpha.max()),
                                     "n_combos": int(len(s))})
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(os.path.join(OUT, "sensitivity_effect.csv"), index=False)

    # ---- ordering and status flips
    flips = []
    piv = runs.pivot_table(index=["rho_i", "rho_s", "n0", "eps_halfwidth", "band"],
                           columns="product", values="alpha", aggfunc="first").reset_index()
    piv["delta_cs2_minus_cci"] = piv["CS2SMOS_shared_snow"] - piv["CCI"]
    for b in BANDS:
        s = piv[piv.band == b].dropna(subset=["delta_cs2_minus_cci"])
        neg = int((s["delta_cs2_minus_cci"] < 0).sum())
        flips.append({"kind": "alpha_ordering", "band": b,
                      "description": "sign(alpha_CS2 - alpha_CCI)",
                      "min": float(s["delta_cs2_minus_cci"].min()),
                      "max": float(s["delta_cs2_minus_cci"].max()),
                      "n_combos": int(len(s)), "n_flipped": neg,
                      "verdict": "no flip" if neg == 0 else "FLIPS"})
    # dominant share per band/product
    for prod in ("CCI", "CS2SMOS_shared_snow"):
        for b in BANDS:
            s = runs[(runs["product"] == prod) & (runs.band == b)]
            dom = s[["share_a", "share_b", "share_c", "share_d"]].idxmax(axis=1)
            counts = dom.value_counts()
            flips.append({"kind": "dominant_share", "band": b, "description": prod,
                          "min": None, "max": None, "n_combos": int(len(s)),
                          "n_flipped": int(counts.size - 1),
                          "verdict": "stable" if counts.size == 1 else "FLIPS",
                          "detail": "; ".join(f"{k}:{v}" for k, v in counts.items())})
    # G8 status flips
    for prod in ("CCI", "CS2SMOS_shared_snow"):
        for b in BANDS:
            s = runs[(runs["product"] == prod) & (runs.band == b)]
            st = s["G8"].value_counts()
            flips.append({"kind": "G8_status", "band": b, "description": prod,
                          "min": float(s["ratio_raw_over_floor"].min()),
                          "max": float(s["ratio_raw_over_floor"].max()),
                          "n_combos": int(len(s)), "n_flipped": int(st.size - 1),
                          "verdict": "stable" if st.size == 1 else "FLIPS",
                          "detail": "; ".join(f"{k}:{v}" for k, v in st.items())})
    # where exactly does the G8 verdict of CS2 change (per parameter)
    cs2_runs = runs[runs["product"] == "CS2SMOS_shared_snow"]
    for param in ("rho_i", "rho_s", "n0", "eps_halfwidth"):
        g = cs2_runs.groupby(param)["G8"].agg(lambda s: "/".join(sorted(set(s))))
        flips.append({"kind": "G8_by_param", "band": "all", "description": f"CS2 x {param}",
                      "min": None, "max": None, "n_combos": int(len(cs2_runs)),
                      "n_flipped": int((g != g.iloc[0]).sum()),
                      "verdict": "stable" if (g == g.iloc[0]).all() else "FLIPS",
                      "detail": "; ".join(f"{k}:{v}" for k, v in g.items())})
    flip_df = pd.DataFrame(flips)
    flip_df.to_csv(os.path.join(OUT, "sensitivity_flips.csv"), index=False)

    # ---- corner-combo verdict across months (does the verdict depend on priors?)
    corners = [(lo, mid_rs, n0lo, elo) for lo in (RHO_I[0], RHO_I[-1])
               for mid_rs in (RHO_S[0], RHO_S[-1])
               for n0lo in (N0[0], N0[-1]) for elo in (EPS[0], EPS[-1])]
    corners = corners[:8]
    cci_m, cs2_m = _load_pair(MONTHS)
    band_m = band_of(cci_m.grid.lat)
    vm_rows = []
    for ri, rs, n0, e in corners:
        pri = Priors(rho_i=ri, rho_s=rs, refraction_n0=n0, eps_halfwidth=e)
        for label, prod in (("CCI", cci_m), ("CS2SMOS_shared_snow", cs2_m)):
            res = compute_budget(prod, pri, POLICY)
            for ti, month in enumerate(prod.time_labels):
                for b in BANDS:
                    sel = res.mask[ti] & (band_m == b)
                    if int(sel.sum()) < 50:
                        continue
                    ratio = float(np.nanmedian(res.sigma_raw[ti][sel] / res.sigma_floor[ti][sel]))
                    vm_rows.append({"rho_i": ri, "rho_s": rs, "n0": n0,
                                    "eps_halfwidth": e, "product": label, "month": month,
                                    "band": b,
                                    "alpha": float(np.nanmedian(res.alpha[ti][sel])),
                                    "ratio_raw_over_floor": ratio, "G8": _g8_status(ratio)})
    pd.DataFrame(vm_rows).to_csv(os.path.join(OUT, "sensitivity_verdict_months.csv"),
                                index=False)

    # ---- gridpoint envelope
    central = compute_budget(cci, Priors(), POLICY)
    cent_cs2 = compute_budget(cs2, Priors(), POLICY)
    np.savez_compressed(
        os.path.join(OUT, "gridpoint_envelope.npz"),
        lat=cci.grid.lat, lon=cci.grid.lon,
        band=band.astype("U6"),
        mask_cci=central.mask[0],
        alpha_cci=central.alpha[0],
        alpha_low_cci=env["CCI"]["low"], alpha_high_cci=env["CCI"]["high"],
        mask_cs2=cent_cs2.mask[0],
        alpha_cs2=cent_cs2.alpha[0],
        alpha_low_cs2=env["CS2SMOS_shared_snow"]["low"],
        alpha_high_cs2=env["CS2SMOS_shared_snow"]["high"],
        rho_i=RHO_I, rho_s=RHO_S, n0=N0, eps_halfwidth=EPS,
    )

    # ---- figures (after CSV/NPZ)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, param in zip(axes.ravel(), ("rho_i", "rho_s", "n0", "eps_halfwidth")):
        for prod, color in (("CCI", "#2b6cb0"), ("CS2SMOS_shared_snow", "#c53030")):
            s = eff[(eff.param == param) & (eff["product"] == prod)]
            if s.empty:
                continue
            for b, ls in zip(BANDS, ("-", "--", ":")):
                sb = s[s.band == b]
                ax.plot(sb.value, sb.alpha_median, ls, marker="o", ms=2.5, color=color,
                        label=f"{prod} {b}" if param == "rho_i" else None)
                ax.fill_between(sb.value, sb.alpha_min, sb.alpha_max, color=color, alpha=0.10)
        ax.set_xlabel(param)
        ax.set_ylabel("band-median alpha")
        ax.set_title(f"{param}: solid=CCl, red=CS2 (band line styles 60-70/70-80/80+)",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sensitivity.png"), dpi=150)
    plt.close(fig)

    # ---- summary
    def _span(prod, b, col="alpha"):
        s = runs[(runs["product"] == prod) & (runs.band == b)]
        return float(s[col].min()), float(s[col].median()), float(s[col].max())

    lines = ["# Prior sensitivity", "",
             f"Sweep: {len(combos)} combinations "
             f"({len(RHO_I)} rho_i x {len(RHO_S)} rho_s x {len(N0)} n0 x {len(EPS)} eps), "
             f"products CCI and CS2SMOS(shared CCI snow), month {MAIN_MONTH}.",
             "All values are inside `constants.PRIOR_RANGES`; anything outside would be "
             "reported as `not_assessable` and excluded from the envelope.",
             "", "## Band-median alpha under the prior envelope", "",
             "| band | CCI min / median / max | CS2 min / median / max | CS2-CCI min / max | ordering |",
             "|---|---|---|---|---|"]
    for b in BANDS:
        a0, a1, a2 = _span("CCI", b)
        c0, c1, c2 = _span("CS2SMOS_shared_snow", b)
        s = piv[piv.band == b]
        dmin, dmax = float(s.delta_cs2_minus_cci.min()), float(s.delta_cs2_minus_cci.max())
        lines.append(f"| {b} | {a0:.3f} / {a1:.3f} / {a2:.3f} | "
                     f"{c0:.3f} / {c1:.3f} / {c2:.3f} | {dmin:.3f} / {dmax:.3f} | "
                     f"{'stable' if dmin > 0 else 'FLIPS'} |")
    amin = float(runs["alpha_cell_min"].min())
    amax = float(runs["alpha_cell_max"].max())
    lines += ["", f"- per-cell alpha over all runs: min {amin:.4f}, max {amax:.4f}; "
              "0 < alpha < 1 is an algebraic bound, so a *sign* flip is impossible by "
              "construction. The only meaningful flips are ordering, dominant share "
              "and G8 status.",
              "", "## Status and share flips", "",
              "| kind | band | subject | verdict | detail |", "|---|---|---|---|---|"]
    for r in flip_df.itertuples():
        det = getattr(r, "detail", "") or ""
        lines.append(f"| {r.kind} | {r.band} | {r.description} | {r.verdict} | {det} |")
    lines += ["", "Full tables: sensitivity_runs.csv, sensitivity_effect.csv, "
              "sensitivity_flips.csv, sensitivity_verdict_months.csv, "
              "gridpoint_envelope.npz, sensitivity.png.", ""]
    with open(os.path.join(OUT, "sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"combos": len(combos),
                      "alpha_cell_min": amin, "alpha_cell_max": amax,
                      "alpha_ordering": {b: {
                          "min": float(piv[piv.band == b].delta_cs2_minus_cci.min()),
                          "max": float(piv[piv.band == b].delta_cs2_minus_cci.max())}
                          for b in BANDS}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
