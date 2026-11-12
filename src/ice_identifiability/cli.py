"""ice-audit command line.

Exit codes:
  0  verdict assessable
  2  verdict not_assessable / partial (missing or unverifiable inputs)
  3  verdict unidentifiable (field is indistinguishable from white noise)
  4  verdict invalid_input
  5  the tool stopped on an input error (diagnostics still written)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__
from .audit import run_audit, write_empty_tables
from .budget import BudgetPolicy
from .constants import Priors
from .ncio import NotAssessable, load_product, overlay_fields

_EXIT = {"assessable": 0, "not_assessable": 2, "unidentifiable": 3,
         "invalid_input": 4}


def _policies(args) -> BudgetPolicy:
    return BudgetPolicy(
        min_concentration=args.min_concentration,
        min_thickness=args.min_thickness,
        max_thickness=args.max_thickness,
        raw_channel=args.raw_channel,
        require_snow=True,
        sigma_snow_override=args.sigma_snow,
        sigma_snow_source=args.sigma_snow_source,
        block_km=args.block_km,
    )


def _priors(args) -> Priors:
    return Priors(
        rho_w=args.rho_w, rho_i=args.rho_i, rho_s=args.rho_s,
        sig_rho_i=args.sig_rho_i, sig_rho_s=args.sig_rho_s,
        refraction_n0=args.refraction_n0, eps_halfwidth=args.eps_halfwidth,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ice-audit",
        description="Audit the identifiability of a sea-ice thickness product "
                    "with a self-reported uncertainty field.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one audit")
    r.add_argument("--thickness", required=True,
                   help="thickness NetCDF file, directory of *.nc, or glob")
    r.add_argument("--unc", help="file/dir carrying the declared uncertainty "
                                 "(omit if it lives in the thickness file)")
    r.add_argument("--snow", help="file/dir carrying snow depth and its uncertainty")
    r.add_argument("--freeboard", help="file/dir carrying freeboard and its uncertainty")
    r.add_argument("--reference", help="independent reference on the same grid")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--label", help="name used in the report (default: input basename)")
    r.add_argument("--months", help="comma-separated YYYY-MM labels to select")

    r.add_argument("--raw-channel", choices=("auto", "freeboard", "posterior"),
                   default="auto",
                   help="what plays the role of the raw channel a; see SPEC.md §3")
    r.add_argument("--min-concentration", type=float, default=15.0,
                   help="mask cells with concentration <= this (percent)")
    r.add_argument("--min-thickness", type=float, default=0.05)
    r.add_argument("--max-thickness", type=float, default=6.0)
    r.add_argument("--block-km", type=float, default=200.0,
                   help="physical size of the block-variance test (default 200 km, "
                        "the scale at which the G3 constant was calibrated). The "
                        "tool refuses a G3 verdict when the grid cannot realise it.")

    r.add_argument("--sigma-snow", type=float,
                   help="explicit snow-depth sigma override in metres (requires "
                        "--sigma-snow-source)")
    r.add_argument("--sigma-snow-source", help="provenance string for --sigma-snow")

    # priors (documented assumptions, all overridable and recorded)
    r.add_argument("--rho-w", type=float, default=1027.0)
    r.add_argument("--rho-i", type=float, default=910.0)
    r.add_argument("--rho-s", type=float, default=320.0)
    r.add_argument("--sig-rho-i", type=float, default=30.0)
    r.add_argument("--sig-rho-s", type=float, default=60.0)
    r.add_argument("--refraction-n0", type=float, default=1.3784048752090222)
    r.add_argument("--eps-halfwidth", type=float, default=0.5)
    return p


def _stop(outdir: str, reason: str, gate: str, command: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    diag = {
        "tool": "ice-identifiability", "version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "not_assessable", "verdict_reason": reason,
        "gate": gate, "gates": [{"id": gate, "name": "input contract",
                                 "status": "not_assessable", "reason": reason}],
        "claims_not_supported": [{"claim": "any identifiability statement",
                                  "status": "not assessable", "why": reason}],
    }
    with open(os.path.join(outdir, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=1, ensure_ascii=False)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# ice-identifiability: not_assessable\n\n"
                f"The tool stopped before computing anything.\n\n"
                f"- gate: `{gate}`\n- reason: {reason}\n\n"
                f"Reproduce:\n\n```\n{command}\n```\n")
    write_empty_tables(outdir)
    print(f"not_assessable [{gate}]: {reason}", file=sys.stderr)
    return 5


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = "ice-audit " + " ".join(sys.argv[1:] if argv is None else argv)
    months = [m.strip() for m in args.months.split(",")] if args.months else None

    if args.sigma_snow is not None and not args.sigma_snow_source:
        return _stop(args.out,
                     "--sigma-snow requires --sigma-snow-source (an unverifiable "
                     "parameter cannot be silently defaulted)", "G4", command)

    try:
        product = load_product(args.thickness, months=months, label=args.label)
        if args.unc:
            overlay_fields(product, args.unc, kinds=("thickness_unc",), months=months)
        if args.snow:
            overlay_fields(product, args.snow, kinds=("snow", "snow_unc"), months=months)
        if args.freeboard:
            overlay_fields(product, args.freeboard,
                           kinds=("freeboard", "freeboard_unc"), months=months)
        reference = (load_product(args.reference, label="reference", months=months)
                     if args.reference else None)
    except NotAssessable as exc:
        return _stop(args.out, exc.reason, exc.gate, command)

    try:
        result = run_audit(product, _priors(args), _policies(args), args.out,
                           command=command, reference=reference)
    except NotAssessable as exc:
        return _stop(args.out, exc.reason, exc.gate, command)
    print(f"verdict: {result['verdict']} — {result['verdict_reason']}")
    print(f"report:  {os.path.join(os.path.abspath(args.out), 'report.md')}")
    return _EXIT.get(result["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
