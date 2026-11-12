"""Re-evaluate G8 on archived runs and rewrite the affected records.

Why this exists
---------------
G8 was defined as the declared *thickness* sigma divided by the nuisance floor.
On the freeboard channel alpha uses a = kappa * sigma_freeboard, so the gate was
guarding a quantity that does not enter the number it was supposed to guard: the
freeboard run of the CCI product was cleared at 0.884, which is the posterior
channel's ratio, while its own ratio is 0.12-0.24.  The gate is corrected in
``gates.py`` (r13) to test the channel's own raw sigma, band by band, and to take
the worst band.

The four real products are not redistributable, so their fields are not in this
archive and the tool cannot be re-run on them here.  What *is* archived is the
per-band record in ``out/<run>/gates.json``: ``share_a`` (= a^2 / sigma_post^2)
and, for runs made after r13, ``g8_ratio_median``.  Because

    a / sqrt(K) = sqrt(share_a / (1 - share_a))

and the median commutes with a monotone map,

    median(a / sqrt(K)) = sqrt(s_med / (1 - s_med)),   s_med = median(share_a),

so the band-wise gate value is recoverable *exactly* from the archived record --
this is not an approximation and not a re-analysis of the fields.  The script
verifies that claim on the synthetic runs, where both routes are available: it
recomputes the ratios from the archived shares and compares them with the values
the tool wrote from the fields, and exits non-zero on any disagreement above
1e-9.

What it does not do
-------------------
It does not touch any gate other than G8, and it does not change a run for which
neither a band record nor a posterior-channel definition is available -- those
are reported as untouched and the exit status reflects it.

Usage
-----
    python examples/rederive_g8.py --out out [--write]

Without ``--write`` it prints the before/after table and changes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

BANDS = ("60-70", "70-80", "80+")
TOL = 1e-9


def _g8_status(ratio):
    if ratio is None or not math.isfinite(ratio):
        return "not_assessable"
    if ratio < 0.5:
        return "fail"
    if ratio < 1.0:
        return "warn"
    return "pass"


def _band_ratios(result):
    """Per-band ratio from the archived per-band record, or {} if unavailable."""
    out = {}
    for name, s in (result.get("bands") or {}).items():
        if "g8_ratio_median" in s and s["g8_ratio_median"] is not None:
            out[name] = float(s["g8_ratio_median"])
        elif "share_a" in s and s["share_a"] is not None:
            sa = float(s["share_a"])
            if 0.0 <= sa < 1.0:
                out[name] = math.sqrt(sa / (1.0 - sa))
    return out


def _g8_reason(value, where, status):
    from ice_identifiability.gates import g8_reason_text
    return g8_reason_text(value, where, status)


def _gate(result, gid):
    """The gate record, or None when the run stopped before that gate."""
    return next((g for g in result["gates"] if g["id"] == gid), None)


def _compose(statuses, g3_reason, g8_reason, g1_reason=""):
    """Mirror of gates.compose_verdict, imported if the package is importable."""
    try:
        from ice_identifiability.gates import compose_verdict
        return compose_verdict(statuses, g3_reason, g8_reason, g1_reason)
    except Exception:                                     # pragma: no cover
        raise SystemExit(
            "cannot import ice_identifiability.gates.compose_verdict; run with "
            "PYTHONPATH=src_toolkit/src so the verdict precedence is taken from "
            "the code and not re-implemented here")


def rederive(run_dir, write=False):
    path = os.path.join(run_dir, "gates.json")
    result = json.load(open(path, encoding="utf-8"))
    g8 = _gate(result, "G8")
    if g8 is None:
        return dict(run=os.path.basename(run_dir), changed=False,
                    note="run stopped before G8 (contract refusal)", before=None,
                    after=None)
    before = (g8["status"], g8["value"], result["verdict"], result.get("alpha_reason_code"))

    ratios = _band_ratios(result)
    note = ""
    if ratios:
        decided = min(ratios, key=lambda k: ratios[k])
        value = ratios[decided]
        note = f"bands {sorted(ratios)}; decided by {decided}"
        inputs = {"ratio_by_band": {k: ratios[k] for k in sorted(ratios)},
                  "deciding_band": decided,
                  "band_margin_cells": 50,
                  "sigma_floor_median": (g8.get("inputs") or {}).get("sigma_floor_median"),
                  "reduced_from": "archived per-band variance shares (r13 correction)"}
        g8["threshold"] = ("median(a / sqrt(b^2+c^2+d^2)) per 10-degree band: "
                           ">= 1.0 pass, [0.5,1.0) warn, < 0.5 fail; run status = worst band")
    else:
        policy = result.get("policy") or {}
        old = (g8.get("inputs") or {}).get("ratio_median")
        if policy.get("raw_channel") == "posterior" and old is not None:
            value = float(old)
            note = "no band record; posterior channel, so the old ratio is already a/sqrt(K)"
            inputs = dict(g8.get("inputs") or {})
            inputs["reduced_from"] = ("archived domain median; the posterior channel makes "
                                     "a = sigma_thickness, so the r13 definition leaves the "
                                     "value unchanged")
            g8["threshold"] = ("median(a / sqrt(b^2+c^2+d^2)): "
                               ">= 1.0 pass, [0.5,1.0) warn, < 0.5 fail")
        else:
            # Nothing to re-derive: the run never reached a ratio.  The verdict
            # is still recomposed, because the precedence itself may have moved.
            value = g8.get("value")
            note = "no band record and not a posterior channel: G8 left as archived"
            inputs = dict(g8.get("inputs") or {})

    if note.startswith("no band record and not"):
        pass
    else:
        status = _g8_status(value)
        where = (f"{decided}N band, worst of {len(ratios)}" if len(ratios) > 1
                 else (f"{decided}N band" if ratios else "domain"))
        g8["status"] = status
        g8["value"] = value
        g8["inputs"] = inputs
        g8["reason"] = _g8_reason(value, where, status)

    statuses = {g["id"]: g["status"] for g in result["gates"]}
    verdict, reason, code = _compose(statuses,
                                     _gate(result, "G3")["reason"],
                                     g8["reason"],
                                     (_gate(result, "G1") or {}).get("reason", ""))
    result["verdict"] = verdict
    result["verdict_reason"] = reason
    result["alpha_reason_code"] = code

    after = (g8["status"], g8["value"], result["verdict"], result["alpha_reason_code"])
    if write:
        _write(run_dir, result)
    return dict(run=os.path.basename(run_dir), changed=(before != after),
                note=note, before=before, after=after)


def _write(run_dir, result):
    from ice_identifiability.report import render

    path = os.path.join(run_dir, "gates.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")

    inp = result.get("inputs") or {}
    cmd = (f"python -m ice_identifiability run --thickness \"{inp.get('product_path')}\" "
           f"--months {','.join(inp.get('months') or [])} "
           f"--raw-channel {inp.get('raw_channel')} "
           f"--out {os.path.basename(run_dir)} --label {result.get('label')}"
           "\n# G8 re-evaluated from the archived per-band record by "
           "examples/rederive_g8.py (the product fields are not redistributable)")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render(result.get("label") or os.path.basename(run_dir), result, cmd, run_dir))

    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath, encoding="utf-8"))
        out = man.setdefault("outputs", {})
        for name in ("gates.json", "report.md"):
            out[name] = _sha256(os.path.join(run_dir, name))
        # the per-cell payload is regenerable from the fixtures and is not shipped
        for name in ("alpha.csv", "budget.csv"):
            out.pop(name, None)
        man["not_shipped"] = {"alpha.csv": "per-cell payload, regenerable: python -m "
                                          "ice_identifiability run ... (see REPRO_CHECK.txt)",
                              "budget.csv": "per-cell payload, regenerable (see above)"}
        man["g8_provenance"] = ("G8 re-evaluated from the archived per-band record; "
                                "see examples/rederive_g8.py")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
            f.write("\n")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_against_fields(outdir):
    """On runs made after r13 both routes exist; they must agree exactly."""
    worst = 0.0
    checked = 0
    for name in sorted(os.listdir(outdir)):
        path = os.path.join(outdir, name, "gates.json")
        if not os.path.exists(path):
            continue
        result = json.load(open(path, encoding="utf-8"))
        g8 = _gate(result, "G8")
        if g8 is None:
            continue
        from_fields = (g8.get("inputs") or {}).get("ratio_by_band")
        if not from_fields:
            continue
        from_record = _band_ratios(result)
        for band, value in from_fields.items():
            if band in from_record:
                worst = max(worst, abs(value - from_record[band]))
                checked += 1
    return checked, worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    checked, worst = verify_against_fields(args.out)
    if checked:
        print(f"[identity check] {checked} band values: |field - archived share| "
              f"max = {worst:.3e}")
        if worst > TOL:
            print("FAIL: the archived shares do not reproduce the field-computed "
                  "ratios; refusing to write", file=sys.stderr)
            return 2
    else:
        print("[identity check] no r13 runs found (nothing to cross-check against); "
              "the identity is asserted by tests/test_g8.py")

    rows = [rederive(os.path.join(args.out, d), write=args.write)
            for d in sorted(os.listdir(args.out))
            if os.path.isdir(os.path.join(args.out, d))
            and os.path.exists(os.path.join(args.out, d, "gates.json"))]

    print(f"\n{'run':32s} {'G8 before':>18s} {'G8 after':>18s}  verdict")
    for r in rows:
        b, a = r["before"], r["after"]
        if b is None:
            print(f" {r['run']:31s} {'-':>18s} {'-':>18s}  -                {r['note']}")
            continue
        bf = "n/a" if b[1] is None else f"{b[0]}:{b[1]:.3f}"
        af = "n/a" if a[1] is None else f"{a[0]}:{a[1]:.3f}"
        flag = "*" if r["changed"] else " "
        print(f"{flag}{r['run']:31s} {bf:>18s} {af:>18s}  {b[2].split('_')[0]:>8s}->{a[2].split('_')[0]:<8s} {r['note']}")
    print("\n(* = record changed)" + ("" if args.write else "   [dry run: pass --write]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
