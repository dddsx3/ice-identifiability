"""Graphical abstract for the EMS submission.

Drawn programmatically with Matplotlib (no generative image model). Rerun with
`python make_ga.py`; it writes figures/graphical_abstract.png and a vector PDF.
The canvas is 5.12 x 2.05 in, i.e. the journal's 13 x 5 cm display size at 1:1,
and every label is at least 7 pt at that size.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_W, FIG_H = 5.12, 2.05
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=400)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")

BLUE = "#1f4e79"
LIGHT = "#eef2f6"
ACCENT = "#b34700"


def box(x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=7.0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.4)


def arrow(x1, x2, y=23.0):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=BLUE))


ax.text(1.0, 36.5, "An audit, not a score", fontsize=10.5, fontweight="bold",
        color=BLUE, ha="left")
ax.text(1.0, 31.8, "an untestable check is never a pass", fontsize=7.2, color=BLUE, ha="left")

box(0.5, 15.0, 15.5, 12, "product\nfile and\nuncertainty", fs=6.8)
arrow(16.3, 18.5)

box(18.8, 12.5, 26, 17, "nine ordered\ngates\nG0 to G8", fs=7.0)
arrow(45.1, 47.3)

box(47.6, 15.0, 16.5, 12, "failed gate\noverrides\nlater gates", fs=6.8)
arrow(64.4, 66.6)

box(66.9, 12.5, 32, 17, "four verdicts\nfive exit codes\n0 · 2 · 3 · 4 · 5", fs=7.0, ec=ACCENT)

box(23.5, 1.5, 75.4, 8.5,
    "ATL20: no effective sample size\n$\\rightarrow$ G5 warns; no error apportionment",
    fs=7.0, ec=ACCENT, fc="#fdf3ec")

ax.text(1.0, 4.4, "worked\ncounter\nexample", fontsize=6.4, color=ACCENT, fontweight="bold", ha="left")

fig.savefig("figures/graphical_abstract.png", dpi=400, bbox_inches=None, facecolor="white")
fig.savefig("figures/graphical_abstract.pdf", bbox_inches=None, facecolor="white")
print("ok")
