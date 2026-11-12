"""Markdown report writer.  Everything here is rendered from values already
written to disk by audit.run_audit (CSV/JSON); no computation happens here."""

from __future__ import annotations

import os


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool,)):
        return "yes" if x else "no"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


_GATE_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL",
              "not_assessable": "N/A"}


def render(name: str, result: dict, command: str, outdir: str) -> str:
    g = result["gates"]
    lines = []
    add = lines.append

    add(f"# ice-identifiability profile: {name}")
    add("")
    add(f"**Verdict:** `{result['verdict']}` — {result['verdict_reason']}")
    add("")
    add("> `alpha` is a monotone rescale of the declared raw channel `a` against the fixed "
        "prior budget, not a discovery, and it is reported only as a footnote to the gates. "
        "When a premise gate rejects the premises -- G3 (no signal), G4 (missing nuisance "
        "inputs) or G8 (declared sigma below the nuisance floor) -- no alpha summary is "
        "given and the verdict records a reason code instead.")
    add("")

    add("## Inputs and provenance")
    add("")
    add("| item | value |")
    add("|---|---|")
    inp = result["inputs"]
    for k in ("product_path", "thickness_var", "thickness_units", "sigma_var",
              "sigma_units", "snow_var", "snow_source_path", "snow_sigma_source",
              "raw_channel", "alpha_object", "months", "sha256"):
        add(f"| {k} | `{inp.get(k)}` |")
    add("")
    add("Priors (all assumptions, recorded for traceability):")
    add("")
    add("| constant | value | provenance |")
    add("|---|---|---|")
    for k, v in result["priors"].items():
        if k == "provenance":
            continue
        prov = result["priors"].get("provenance", {}).get(k, "")
        add(f"| `{k}` | {_fmt(v, 4)} | {prov} |")
    add("")

    add("## Gates")
    add("")
    add("| gate | name | status | value | threshold | reason |")
    add("|---|---|---|---|---|---|")
    for x in g:
        add(f"| {x['id']} | {x['name']} | **{_GATE_ICON[x['status']]}** | "
            f"{_fmt(x['value'])} | {_fmt(x['threshold'])} | {x['reason']} |")
    add("")

    add("## Measured structure (data only)")
    add("")
    add("| quantity | value | reading |")
    add("|---|---|---|")
    st = result["structure"]
    add(f"| nu(lag 1) | {_fmt(st.get('nu_lag1'))} | 1 = white noise, 0 = smooth |")
    add(f"| nu(lag 8) | {_fmt(st.get('nu_lag8'))} | should rise with lag for a coherent field |")
    cell = ""
    if st.get("cell_km_x") is not None and st.get("cell_km_y") is not None:
        cell = (f" (cell {_fmt(st.get('cell_km_x'), 1)} x "
                f"{_fmt(st.get('cell_km_y'), 1)} km; realised block "
                f"{_fmt(st.get('block_extent_km_x'), 0)} x "
                f"{_fmt(st.get('block_extent_km_y'), 0)} km)")
    add(f"| block excess, {_fmt(st.get('block_cells'), 0)}x{_fmt(st.get('block_cells'), 0)}"
        f"{cell} | {_fmt(st.get('block_excess_primary'))} | 1 = exactly the declared white noise |")
    add(f"| block excess, sigma-free {_fmt(st.get('block_cells'), 0)}x"
        f"{_fmt(st.get('block_cells'), 0)}{cell} | "
        f"{_fmt(st.get('block_excess_free_primary'))} | 1 = white noise at this block "
        "size; the quantity G3 tests |")
    add(f"| median S/N | {_fmt(st.get('snr_median'))} | median(|h|/sigma_declared) |")
    add(f"| q_sigma | {_fmt(st.get('q_sigma'))} | grid white level / declared sigma^2 |")
    add(f"| valid cells (all months) | {_fmt(st.get('n_valid'), 0)} | |")
    add("")

    if result.get("reference"):
        add("## Independent reference")
        add("")
        r = result["reference"]
        if r.get("status") == "assessed":
            add("| quantity | value |")
            add("|---|---|")
            for k in ("n_pairs", "bias", "rms", "nu_lag1", "f_irr"):
                add(f"| {k} | {_fmt(r.get(k))} |")
            add("")
            add("`f_irr = 1 - nu_lag1(d)` is the spatially coherent share of the "
                "product-reference difference: it is what will not average away.")
            add("")
        else:
            add(f"Reference not assessable: {r.get('reason')}")
            add("")

    if result["verdict"] == "assessable" and result.get("bands"):
        add("## Alpha and budget by latitude band")
        add("")
        add(f"Object: `{inp.get('alpha_object')}`")
        add("")
        add("| band | n | alpha median | alpha p10 | alpha p90 | sigma_post (m) | "
            "sigma_floor (m) | share a | share b | share c | share d |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for band, s in result["bands"].items():
            add(f"| {band} | {_fmt(s['n'], 0)} | {_fmt(s['alpha_median'])} | "
                f"{_fmt(s['alpha_p10'])} | {_fmt(s['alpha_p90'])} | "
                f"{_fmt(s['sigma_post_median'])} | {_fmt(s['sigma_floor_median'])} | "
                f"{_fmt(s['share_a'])} | {_fmt(s['share_b'])} | {_fmt(s['share_c'])} | "
                f"{_fmt(s['share_d'])} |")
        add("")
        add("share a = raw channel, b = snow depth, c = density, d = penetration; "
            "the four shares sum to 1 by construction (bookkeeping, not a result).")
        add("")

    add("## Conclusions that this product does not support")
    add("")
    if result["claims_not_supported"]:
        add("| claim | status | why |")
        add("|---|---|---|")
        for c in result["claims_not_supported"]:
            add(f"| {c['claim']} | **{c['status']}** | {c['why']} |")
    else:
        add("None flagged.")
    add("")

    add("## Reproduce")
    add("")
    add("```")
    add(command)
    add("```")
    add("")
    add(f"Outputs in `{os.path.abspath(outdir)}`: `alpha.csv`, `budget.csv`, "
        "`gates.json`, `report.md`, `manifest.json`.")
    add("")
    add("Known failure modes (sigma=0/fill, unknown units, grid mismatch, missing "
        "nuisance inputs, white-noise input) are tabulated in Section 7 of the paper.")
    add("")
    return "\n".join(lines)
