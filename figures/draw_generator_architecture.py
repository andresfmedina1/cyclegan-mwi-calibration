# -*- coding: utf-8 -*-
"""
draw_generator_architecture.py  —  Fig. 4.2

Changes from v2:
  - Real input/output heatmaps from calibration_example_run8.mat
    (input = raw experimental ΔS^meas, output = ML-calibrated)
  - Reciprocity layer shown as a proper network block (dashed border)
  - S_ij / S_ji notation replaced with S_AB / S_BA

Run:
    /Users/felipemedina/anaconda3/envs/cyclegan/bin/python3 draw_generator_architecture.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
import scipy.io as sio

CDIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(CDIR, "fig_4_2_generator_architecture.pdf")

# ── load real calibration example ─────────────────────────────────────────────
d   = sio.loadmat(os.path.join(CDIR, "calibration_example_run8.mat"))
RAW_RE  = d["raw_real"].astype(float)          # 6×12  input  (ΔS^meas, real part)
CAL_RE  = d["calibrated_real"].astype(float)   # 6×12  output (ML-calibrated)
TGT_RE  = d["target_real"].astype(float)       # 6×12  target (ΔE^sim)

# ── palette ───────────────────────────────────────────────────────────────────
C_CONV        = "#7B6FBE"
C_CONCAT      = "#4E9E52"
C_OUTPUT_CONV = "#2E4057"
C_REC_EDGE    = "#B06A00"
C_REC_FILL    = "#FFF3CD"
C_REC_TEXT    = "#7A4500"
C_ARROW       = "#333333"
C_SKIP        = "#888888"
C_POOL        = "#993333"

FW, FH = 15, 9   # ~6" print width → scale factor 0.40 → design fonts ×2.5
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor("white")

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────

def add_heatmap(cx, cy, w, h, data, cmap="viridis"):
    """Embed a real data matrix as a heatmap at (cx,cy) inches."""
    left   = (cx - w / 2) / FW
    bottom = (cy - h / 2) / FH
    ax_h   = fig.add_axes([left, bottom, w / FW, h / FH])
    ax_h.imshow(data, cmap=cmap, aspect="auto", origin="upper", interpolation="nearest")
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_linewidth(2.0)
        sp.set_color("#CCCCCC")


def block(cx, cy, w, h, color, alpha=0.93, dashed=False):
    """Rounded rectangle. dashed=True → non-trainable style."""
    ls = (0, (5, 3)) if dashed else "solid"
    lw = 2.4 if dashed else 1.8
    ec = C_REC_EDGE if dashed else "white"
    r = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.17",
        linewidth=lw, edgecolor=ec, linestyle=ls,
        facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(r)


def lbl_top(cx, cy, h, text, fs=14, bold=True, color="#111111"):
    ax.text(cx, cy + h / 2 + 0.14, text,
            ha="center", va="bottom", fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=color, zorder=5)


def lbl_bot(cx, cy, h, text, fs=12, color="#555555"):
    ax.text(cx, cy - h / 2 - 0.15, text,
            ha="center", va="top", fontsize=fs,
            color=color, style="italic", zorder=5)


def arrow(x0, y0, x1, y1, color=C_ARROW, lw=2.2, ms=18, shrink=5):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms, shrinkA=shrink, shrinkB=shrink),
        zorder=6)


def arrow_skip(x0, y0, x1, y1):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.8,
                        linestyle="dashed", mutation_scale=15,
                        shrinkA=4, shrinkB=4),
        zorder=6)


def polyline_arrow(pts, color=C_ARROW, lw=2.2, ms=18):
    """Multi-segment path with arrowhead at the last segment."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw,
            solid_capstyle="round", zorder=5)
    ax.annotate("",
        xy=pts[-1], xytext=pts[-2],
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms, shrinkA=0, shrinkB=5),
        zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT  (all in inches — FW=15, FH=9, print width ≈6", scale ≈0.40)
# ══════════════════════════════════════════════════════════════════════════════

Y_TOP = 6.3
Y_BOT = 2.6

BW   = 0.70     # conv block width
BH   = 2.0      # conv block height
BH_B = 1.70     # bottleneck height

HM_W, HM_H = 1.15, 1.60   # heatmap tile size

X_IN   = 0.80
X_E1   = 2.20
X_E2   = 3.20
X_BN1  = 4.85
X_BN2  = 5.85
X_CONC = 7.30   # concat
X_D1   = 8.65   # decoder conv
X_OUT  = 9.95   # 1×1 output conv  (width 0.80)
X_REC  = 11.55  # reciprocity layer (RBW=1.25)
X_OMAP = 13.40  # output heatmap

# ══════════════════════════════════════════════════════════════════════════════
# DRAW
# ══════════════════════════════════════════════════════════════════════════════

# ─── Title ───────────────────────────────────────────────────────────────────
ax.text(FW / 2, FH - 0.30,
        "Generator architecture used in this work",
        ha="center", va="top", fontsize=20, fontweight="bold",
        color="#111111", zorder=5)

# ─── Input heatmap ───────────────────────────────────────────────────────────
add_heatmap(X_IN, Y_TOP, HM_W, HM_H, RAW_RE, cmap="viridis")
ax.text(X_IN, Y_TOP + HM_H/2 + 0.14, "Input",
        ha="center", va="bottom", fontsize=15, fontweight="bold",
        color="#111111", zorder=5)
ax.text(X_IN, Y_TOP - HM_H/2 - 0.17,
        r"$\Delta S^{\mathrm{meas}}$",
        ha="center", va="top", fontsize=13, color="#555555",
        style="italic", zorder=5)
ax.text(X_IN, Y_TOP - HM_H/2 - 0.52,
        "(6×12×2)",
        ha="center", va="top", fontsize=11, color="#888888",
        style="italic", zorder=5)

# ─── Encoder blocks ──────────────────────────────────────────────────────────
block(X_E1, Y_TOP, BW, BH, C_CONV)
block(X_E2, Y_TOP, BW, BH, C_CONV)
lbl_top((X_E1+X_E2)/2, Y_TOP, BH, "8 × 3×3\nConv2D", fs=13)

arrow(X_IN + HM_W/2, Y_TOP, X_E1 - BW/2, Y_TOP, lw=2.0, ms=16)
arrow(X_E1 + BW/2,   Y_TOP, X_E2 - BW/2, Y_TOP, lw=2.0, ms=16)

# ─── MaxPooling elbow ────────────────────────────────────────────────────────
x_turn = X_E2 + BW/2 + 0.28    # columna del segmento vertical
polyline_arrow(
    [(X_E2 + BW/2, Y_TOP),
     (x_turn,      Y_TOP),
     (x_turn,      Y_BOT),
     (X_BN1 - BW/2, Y_BOT)],
    color=C_POOL, lw=2.2, ms=16)
ax.text(x_turn + 0.14, (Y_TOP + Y_BOT) / 2 + 0.12,
        "2×2 MaxPooling",
        ha="left", va="center", fontsize=12,
        color=C_POOL, fontweight="bold", zorder=5)

# ─── Bottleneck ──────────────────────────────────────────────────────────────
block(X_BN1, Y_BOT, BW, BH_B, C_CONV)
block(X_BN2, Y_BOT, BW, BH_B, C_CONV)
lbl_top(X_BN1, Y_BOT, BH_B, "8  3×3\nConv2D",  fs=11.5)
lbl_top(X_BN2, Y_BOT, BH_B, "16  3×3\nConv2D", fs=11.5)
ax.text(X_BN1, Y_BOT - BH_B/2 - 0.15, "Transpose",
        ha="center", va="top", fontsize=11, color="#555555", style="italic", zorder=5)
ax.text((X_BN1+X_BN2)/2, Y_BOT - BH_B/2 - 0.55, "3×6",
        ha="center", va="top", fontsize=12, color="#555555", style="italic", zorder=5)

arrow(X_BN1 + BW/2,    Y_BOT, X_BN2 - BW/2, Y_BOT, lw=2.0, ms=16)

# ─── Bottleneck → Concat ─────────────────────────────────────────────────────
x_elbow = X_BN2 + BW/2 + 0.35
polyline_arrow(
    [(X_BN2 + BW/2, Y_BOT),
     (x_elbow, Y_BOT),
     (x_elbow, Y_TOP),
     (X_CONC - 0.29, Y_TOP)],
    color=C_ARROW, lw=2.0, ms=16)

# ─── Skip connection ─────────────────────────────────────────────────────────
y_skip = Y_TOP + BH/2 + 0.50
arrow_skip(X_E2 + BW/2, y_skip, X_CONC - 0.28, y_skip)
ax.text((X_E2 + X_CONC) / 2, y_skip + 0.13,
        "skip connection",
        ha="center", va="bottom", fontsize=11.5,
        color=C_SKIP, style="italic", zorder=5)
ax.plot([X_E2 + BW/2, X_E2 + BW/2], [Y_TOP + BH/2, y_skip],
        color=C_SKIP, lw=1.7, linestyle="dashed", zorder=4)
ax.plot([X_CONC - 0.28, X_CONC - 0.28], [y_skip, Y_TOP + BH/2 - 0.28],
        color=C_SKIP, lw=1.7, linestyle="dashed", zorder=4)

# ─── Concat ──────────────────────────────────────────────────────────────────
block(X_CONC, Y_TOP, 0.58, BH, C_CONCAT)
lbl_top(X_CONC, Y_TOP, BH, "Concat", fs=13)

arrow(X_CONC + 0.29, Y_TOP, X_D1 - (BW+0.15)/2, Y_TOP, lw=2.0, ms=16)

# ─── Decoder conv ────────────────────────────────────────────────────────────
block(X_D1, Y_TOP, BW + 0.15, BH, C_CONV)
lbl_top(X_D1, Y_TOP, BH, "8  3×3\nConv2D", fs=13)
lbl_bot(X_D1, Y_TOP, BH, "6×12", fs=12)

arrow(X_D1 + (BW+0.15)/2, Y_TOP, X_OUT - 0.80/2, Y_TOP, lw=2.0, ms=16)

# ─── 1×1 output conv ─────────────────────────────────────────────────────────
block(X_OUT, Y_TOP, 0.80, BH, C_OUTPUT_CONV)
lbl_top(X_OUT, Y_TOP, BH, "2  1×1\nConv2D\n(linear)", fs=12)

arrow(X_OUT + 0.80/2, Y_TOP, X_REC - 1.20/2, Y_TOP,
      color=C_REC_EDGE, lw=2.2, ms=17)

# ─── Reciprocity layer block (dashed border = non-trainable) ─────────────────
RBW = 1.20
RBH = BH

block(X_REC, Y_TOP, RBW, RBH, C_REC_FILL, alpha=0.97, dashed=True)

ax.text(X_REC, Y_TOP + RBH/2 + 0.14,
        "Reciprocity layer",
        ha="center", va="bottom", fontsize=13,
        fontweight="bold", color=C_REC_EDGE, zorder=5)

ax.text(X_REC, Y_TOP + 0.30,
        r"avg$(S_{AB},\,S_{BA})$",
        ha="center", va="center", fontsize=12.5,
        color=C_REC_TEXT, zorder=5)
ax.text(X_REC, Y_TOP - 0.22,
        r"$\Rightarrow\;S_{AB} = S_{BA}$",
        ha="center", va="center", fontsize=11.5,
        color=C_REC_TEXT, zorder=5)

ax.text(X_REC, Y_TOP - RBH/2 - 0.17,
        r"non-trainable  |  $G_{E \to S}$ only",
        ha="center", va="top", fontsize=10.5,
        color=C_REC_EDGE, style="italic", zorder=5)

arrow(X_REC + RBW/2, Y_TOP, X_OMAP - HM_W/2, Y_TOP,
      color=C_REC_EDGE, lw=2.2, ms=17)

# ─── Output heatmap ──────────────────────────────────────────────────────────
add_heatmap(X_OMAP, Y_TOP, HM_W, HM_H, CAL_RE, cmap="viridis")
ax.text(X_OMAP, Y_TOP + HM_H/2 + 0.14, "Output",
        ha="center", va="bottom", fontsize=15, fontweight="bold",
        color="#111111", zorder=5)
ax.text(X_OMAP, Y_TOP - HM_H/2 - 0.17,
        r"$\Delta \hat{E}^{\mathrm{sim}}$",
        ha="center", va="top", fontsize=13, color="#555555",
        style="italic", zorder=5)
ax.text(X_OMAP, Y_TOP - HM_H/2 - 0.52,
        "(6×12×2)",
        ha="center", va="top", fontsize=11, color="#888888",
        style="italic", zorder=5)

# ─── Bottom note ─────────────────────────────────────────────────────────────
ax.text(FW / 2, 0.35,
        "No global input→output skip connection  "
        "(following Jeffrey et al., who found it provided no benefit)",
        ha="center", va="bottom", fontsize=11.5,
        color="#777777", style="italic", zorder=5)

# ─── Save ────────────────────────────────────────────────────────────────────
fig.savefig(OUT, bbox_inches="tight", dpi=200, facecolor="white")
plt.close(fig)
print("Saved:", OUT)
