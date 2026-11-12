"""Regenerate Figure 5 (validation) from the shipped records.

The submitted archive shipped ``figures/fig5_validation.pdf`` with no script that
produced it.  This script produces it from data that ships with the archive: the
generator's truth (``examples/data/synth_truth.json``) and the two runs on
``synth_sigma_grid`` (``out/S1_synth_sigma``, ``out/S2_synth_sigma_freeboard``).

Panel (a): recovered alpha against the target, one series per channel.  The two
channels of the fixture declare the same 0.9 m thickness error, so the two series
coincide; the panel is a test of object invariance, not of a difference.
Panel (b): the error decomposed.  The tool reproduces the analytic 25 km value to
about 1e-8; what separates that value from the fine-scale target is the
aggregation penalty, which is four orders of magnitude larger.

    python examples/make_validation_figure.py --out out --figdir figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ("60-70", "70-80")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--data", default=os.path.join("examples", "data"))
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args(argv)

    truth = json.load(open(os.path.join(args.data, "synth_truth.json"),
                           encoding="utf-8"))
    tgt = truth["alpha_true_band"]
    fine = truth["alpha_coarse_analytic"]

    runs = {}
    for name in ("S1_synth_sigma", "S2_synth_sigma_freeboard"):
        d = json.load(open(os.path.join(args.out, name, "gates.json"), encoding="utf-8"))
        runs[name] = {b: d["bands"][b]["alpha_median"] for b in BANDS}

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)

    # (a) recovered against target --------------------------------------------
    a = ax[0]
    xs = [tgt[b] for b in BANDS]
    a.plot([0.4, 0.72], [0.4, 0.72], ls=":", lw=1.0, color="0.55", zorder=1,
           label="1:1")
    a.plot(xs, [runs["S1_synth_sigma"][b] for b in BANDS], "o", ms=6, mfc="none",
           mec="C0", mew=1.4, zorder=3, label="declared-posterior channel (S1)")
    a.plot(xs, [runs["S2_synth_sigma_freeboard"][b] for b in BANDS], "x", ms=6,
           color="C1", mew=1.4, zorder=4, label="freeboard channel (S2)")
    a.set_xlim(0.40, 0.72)
    a.set_ylim(0.40, 0.72)
    a.set_xlabel(r"target $\alpha$")
    a.set_ylabel(r"recovered $\alpha$")
    a.set_title("(a) both channels, one declaration", fontsize=9)
    a.legend(fontsize=6.5, loc="upper left", frameon=False)
    a.tick_params(labelsize=7)

    # (b) error decomposition --------------------------------------------------
    b = ax[1]
    labels, vals = [], []
    for band in BANDS:
        pole = fine["posterior"][band]
        labels.append(band.replace("-", "--") + "$^\\circ$N\naggregation")
        vals.append(abs(pole - tgt[band]))
        labels.append(band.replace("-", "--") + "$^\\circ$N\nsolver")
        vals.append(abs(runs["S1_synth_sigma"][band] - pole))
    pos = range(len(vals))
    b.bar(pos, vals, color=["C0", "C1"] * len(BANDS), width=0.65)
    b.set_yscale("log")
    b.set_ylim(1e-9, 1e-3)
    b.set_xticks(list(pos))
    b.set_xticklabels(labels, fontsize=6)
    b.set_ylabel(r"$|\Delta\alpha|$")
    b.set_title("(b) aggregation dominates the residual", fontsize=9)
    b.tick_params(axis="y", labelsize=7)
    b.axhline(1e-8, color="0.55", ls=":", lw=0.8)

    os.makedirs(args.figdir, exist_ok=True)
    out = os.path.join(args.figdir, "fig5_validation.pdf")
    fig.savefig(out, bbox_inches=None, facecolor="white")
    print("wrote", out)
    for band in BANDS:
        pole = fine["posterior"][band]
        print(f"  {band}: target {tgt[band]:.5f}  analytic25km {pole:.6f}  "
              f"tool {runs['S1_synth_sigma'][band]:.6f}  "
              f"aggregation {abs(pole - tgt[band]):.2e}  "
              f"solver {abs(runs['S1_synth_sigma'][band] - pole):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
