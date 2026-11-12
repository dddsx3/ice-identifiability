"""Regenerate the three summary tables from the per-run records.

The submitted archive shipped ``out/gate_matrix.csv``, ``out/gate_values.csv`` and
``out/run_summary.csv`` but no script that produces them; they had been written by
hand from the per-run JSON.  This script is that script.  It reads only
``out/<run>/gates.json`` (shipped for every run, including the ones whose fields
are not redistributable) and writes the three tables, so the tables cannot drift
from the records they summarise.

    python examples/make_summary_tables.py --out out

Columns are taken from the record, not recomputed: a number that is not in
``gates.json`` does not appear in the tables.  The one exception is the G8 ratio
per band, which is read from ``inputs.ratio_by_band`` and is the value the gate
decided on.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

GATES = [f"G{i}" for i in range(9)]
BANDS = ("60-70", "70-80", "80+")


def _load(outdir):
    runs = []
    for name in sorted(os.listdir(outdir)):
        path = os.path.join(outdir, name, "gates.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                runs.append((name, json.load(f)))
    return runs


def _gate(result, gid):
    return next((g for g in result.get("gates", []) if g["id"] == gid), None)


def _v(result, gid):
    g = _gate(result, gid)
    return None if g is None else g.get("value")


def _i(result, gid, key):
    g = _gate(result, gid)
    return None if g is None else (g.get("inputs") or {}).get(key)


def _f(x, nd=4):
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def write_gate_matrix(runs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run"] + GATES + ["verdict"])
        for name, r in runs:
            row = [name]
            for gid in GATES:
                g = _gate(r, gid)
                row.append("" if g is None else g["status"])
            row.append(r.get("verdict"))
            w.writerow(row)


def write_gate_values(runs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "gate", "status", "value", "threshold", "reason"])
        for name, r in runs:
            for g in r.get("gates", []):
                w.writerow([name, g["id"], g["status"], _f(g.get("value"), 12),
                            g.get("threshold") or "", g.get("reason") or ""])


def write_run_summary(runs, path):
    head = ["run", "product", "verdict", "alpha_reason_code", "g8_status", "g8_ratio",
            "g8_ratio_60-70", "g8_ratio_70-80", "g8_ratio_80+",
            "alpha_object", "thickness_var", "sigma_var", "snow_var",
            "n_valid", "nu_lag1", "nu_lag8", "block_cells",
            "block_excess_primary", "block_excess_free", "block_km_x", "block_km_y",
            "block_excess_8", "q_sigma", "snr_median",
            "alpha_median_60-70", "alpha_median_70-80", "alpha_median_80+",
            "share_a_60-70", "share_a_70-80", "share_a_80+"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for name, r in sorted(runs):
            g8 = _gate(r, "G8") or {}
            by_band = (g8.get("inputs") or {}).get("ratio_by_band") or {}
            bands = r.get("bands") or {}
            w.writerow([
                name, r.get("label"), r.get("verdict"), r.get("alpha_reason_code") or "",
                g8.get("status", ""), _f(g8.get("value")),
                *[_f(by_band.get(b)) for b in BANDS],
                _i(r, "G4", "alpha_object"), _i(r, "G0", "thickness_var"),
                _i(r, "G1", "sigma_var"), _i(r, "G4", "snow_var"),
                _f(_i(r, "G3", "n_valid"), 5) if _i(r, "G3", "n_valid") else _f(
                    (r.get("structure") or {}).get("n_valid"), 5),
                _f((r.get("structure") or {}).get("nu_lag1")),
                _f((r.get("structure") or {}).get("nu_lag8")),
                _f((r.get("structure") or {}).get("block_cells"), 3),
                _f((r.get("structure") or {}).get("block_excess_primary")),
                _f((r.get("structure") or {}).get("block_excess_free_primary")),
                _f((r.get("structure") or {}).get("block_extent_km_x"), 4),
                _f((r.get("structure") or {}).get("block_extent_km_y"), 4),
                _f((r.get("structure") or {}).get("block_excess_8")),
                _f((r.get("structure") or {}).get("q_sigma")),
                _f((r.get("structure") or {}).get("snr_median")),
                *[_f(bands.get(b, {}).get("alpha_median")) for b in BANDS],
                *[_f(bands.get(b, {}).get("share_a")) for b in BANDS],
            ])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    runs = _load(args.out)
    write_gate_matrix(runs, os.path.join(args.out, "gate_matrix.csv"))
    write_gate_values(runs, os.path.join(args.out, "gate_values.csv"))
    write_run_summary(runs, os.path.join(args.out, "run_summary.csv"))
    print(f"wrote gate_matrix.csv, gate_values.csv and run_summary.csv for "
          f"{len(runs)} runs into {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
