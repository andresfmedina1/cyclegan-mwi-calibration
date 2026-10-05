# -*- coding: utf-8 -*-
"""
draw_discriminator_architecture.py  —  Fig. 4.3

Discriminator architecture (disc_dim=16, disc_depth=2, disc_kernel=5):
  Input (6x12x2)
  → Conv2D(16, 5x5, stride=2) + LeakyReLU(0.2)  → 3x6
  → Conv2D(32, 5x5, stride=2) + LeakyReLU(0.2)  → 2x3
  → Conv2D(64, 5x5, stride=1) + LeakyReLU(0.2)  → 2x3
  → Conv2D(1,  5x5, stride=1)                    → 2x3
  → Flatten → Dense(1) + tanh → scalar score

Run:
    /Users/felipemedina/anaconda3/envs/cyclegan/bin/python3 draw_discriminator_architecture.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
import scipy.io as sio

CDIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(CDIR, "fig_4_3_discriminator_architecture.pdf")

# ── load real data for input heatmap ─────────────────────────────────────────
d      = sio.loadmat(os.path.join(CDIR, "calibration_example_run8.mat"))
RAW_RE = d["raw_real"].astype(float)    # 6×12 — real experimental input

# ── palette ───────────────────────────────────────────────────────────────────
C_CONV     = "#7B6FBE"      # purple  — conv blocks
C_DENSE    = "#2E4057"      # dark    — Dense / flatten
C_TANH     = "#2A7A5A"      # teal    — tanh block
C_SCORE    = "#4A4A7A"      # dark blue — output score
C_ARROW    = "#333333"
C_LRU      = "#994444"      # label colour for LeakyReLU note
C_STRIDE   = "#335588"      # stride annotation

FW, FH = 18, 9
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor("white")

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────

def add_heatmap(cx, cy, w, h, data, cmap="viridis", sigma=0):
    d2 = gaussian_filter(data, sigma=sigma) if sigma > 0 else data
    left   = (cx - w / 2) / FW
    bottom = (cy - h / 2) / FH
    ax_h   = fig.add_axes([left, bottom, w / FW, h / FH])
    ax_h.imshow(d2, cmap=cmap, aspect="auto", origin="upper",
                interpolation="nearest")
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_linewidth(1.8); sp.set_color("#CCCCCC")


def block(cx, cy, w, h, color, alpha=0.93, radius=0.13):
    r = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.8, edgecolor="white",
        facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(r)


def lbl_above(cx, cy, h, text, fs=13, bold=True, color="#111111"):
    ax.text(cx, cy + h / 2 + 0.14, text,
            ha="center", va="bottom", fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=color, zorder=5)


def lbl_below(cx, cy, h, text, fs=11.5, color="#555555"):
    ax.text(cx, cy - h / 2 - 0.15, text,
            ha="center", va="top", fontsize=fs,
            color=color, style="italic", zorder=5)


def lbl_inside(cx, cy, text, fs=11, color="white"):
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold", zorder=5)


def arrow(x0, y0, x1, y1, color=C_ARROW, lw=2.0, ms=16, shrink=5):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=ms, shrinkA=shrink, shrinkB=shrink),
        zorder=6)


def dim_badge(cx, cy, text, fs=12.5):
    """Small pill showing spatial dimensions below a block."""
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color="#333355",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#E8E8F4",
                      edgecolor="#AAAACC", linewidth=1.0),
            zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

Y_MID = 5.2          # vertical centre of all blocks
HM_W, HM_H = 1.38, 1.90

# Block dimensions
BH    = 2.40
BW_C1 = 0.98
BW_C2 = 0.90
BW_C3 = 0.90
BW_C4 = 0.72
BW_FL = 0.34
BW_DE = 0.66
BW_TH = 0.66

# X centres  (spaced wider: ×1.2 from previous layout)
X_IN   = 0.98
X_C1   = 3.00
X_C2   = 4.95
X_C3   = 6.80
X_C4   = 8.45
X_FL   = 9.85
X_DE   = 11.05
X_TH   = 12.35
X_OUT  = 14.10   # output score badge

# ══════════════════════════════════════════════════════════════════════════════
# DRAW
# ══════════════════════════════════════════════════════════════════════════════

# Title
ax.text(FW / 2, FH - 0.30,
        "Discriminator architecture used in this work",
        ha="center", va="top", fontsize=24, fontweight="bold",
        color="#111111", zorder=5)

# ─── Input heatmap ───────────────────────────────────────────────────────────
add_heatmap(X_IN, Y_MID, HM_W, HM_H, RAW_RE, cmap="viridis")
ax.text(X_IN, Y_MID + HM_H/2 + 0.14, "Input",
        ha="center", va="bottom", fontsize=18, fontweight="bold",
        color="#111111", zorder=5)
ax.text(X_IN, Y_MID - HM_H/2 - 0.17,
        r"$\Delta S^{\mathrm{meas}}$ or $\Delta E^{\mathrm{sim}}$",
        ha="center", va="top", fontsize=14,
        color="#555555", style="italic", zorder=5)
ax.text(X_IN, Y_MID - HM_H/2 - 0.55,
        "(6×12×2)",
        ha="center", va="top", fontsize=13,
        color="#888888", style="italic", zorder=5)

arrow(X_IN + HM_W/2, Y_MID, X_C1 - BW_C1/2, Y_MID, lw=2.0, ms=16)

# ─── Conv Block 1: 16 filters, 5×5, stride 2 → 3×6 ──────────────────────────
block(X_C1, Y_MID, BW_C1, BH, C_CONV)
lbl_above(X_C1, Y_MID, BH, "16  5×5\nConv2D", fs=15)
ax.text(X_C1, Y_MID, "stride 2", ha="center", va="center",
        fontsize=14, color="white", fontweight="bold", zorder=5)
ax.text(X_C1, Y_MID - BH/2 - 0.20,
        "LeakyReLU", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
ax.text(X_C1, Y_MID - BH/2 - 0.52,
        r"($\alpha$ = 0.2)", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
dim_badge(X_C1, Y_MID - BH/2 - 1.00, "3×6")

arrow(X_C1 + BW_C1/2, Y_MID, X_C2 - BW_C2/2, Y_MID, lw=2.0, ms=16)

# ─── Conv Block 2: 32 filters, 5×5, stride 2 → 2×3 ──────────────────────────
block(X_C2, Y_MID, BW_C2, BH, C_CONV)
lbl_above(X_C2, Y_MID, BH, "32  5×5\nConv2D", fs=15)
ax.text(X_C2, Y_MID, "stride 2", ha="center", va="center",
        fontsize=14, color="white", fontweight="bold", zorder=5)
ax.text(X_C2, Y_MID - BH/2 - 0.20,
        "LeakyReLU", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
ax.text(X_C2, Y_MID - BH/2 - 0.52,
        r"($\alpha$ = 0.2)", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
dim_badge(X_C2, Y_MID - BH/2 - 1.00, "2×3")

arrow(X_C2 + BW_C2/2, Y_MID, X_C3 - BW_C3/2, Y_MID, lw=2.0, ms=16)

# ─── Conv Block 3: 64 filters, 5×5, stride 1 → 2×3 ──────────────────────────
block(X_C3, Y_MID, BW_C3, BH, C_CONV)
lbl_above(X_C3, Y_MID, BH, "64  5×5\nConv2D", fs=15)
ax.text(X_C3, Y_MID, "stride 1", ha="center", va="center",
        fontsize=14, color="white", fontweight="bold", zorder=5)
ax.text(X_C3, Y_MID - BH/2 - 0.20,
        "LeakyReLU", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
ax.text(X_C3, Y_MID - BH/2 - 0.52,
        r"($\alpha$ = 0.2)", ha="center", va="top",
        fontsize=13, color=C_LRU, fontweight="bold", zorder=5)
dim_badge(X_C3, Y_MID - BH/2 - 1.00, "2×3")

arrow(X_C3 + BW_C3/2, Y_MID, X_C4 - BW_C4/2, Y_MID, lw=2.0, ms=16)

# ─── Conv Block 4: 1 filter, 5×5, stride 1 → 2×3×1 ──────────────────────────
block(X_C4, Y_MID, BW_C4, BH, C_DENSE)
lbl_above(X_C4, Y_MID, BH, "1  5×5\nConv2D", fs=15, color="#111111")
ax.text(X_C4, Y_MID, "stride 1", ha="center", va="center",
        fontsize=14, color="#AACCEE", fontweight="bold", zorder=5)
dim_badge(X_C4, Y_MID - BH/2 - 0.45, "2×3×1")

arrow(X_C4 + BW_C4/2, Y_MID, X_FL - BW_FL/2, Y_MID, lw=2.0, ms=16)

# ─── Flatten (thin vertical bar) ─────────────────────────────────────────────
block(X_FL, Y_MID, BW_FL, BH * 0.70, C_DENSE, alpha=0.85)
lbl_above(X_FL, Y_MID, BH * 0.70, "Flatten", fs=14)
dim_badge(X_FL, Y_MID - BH*0.70/2 - 0.45, "6")

arrow(X_FL + BW_FL/2, Y_MID, X_DE - BW_DE/2, Y_MID, lw=2.0, ms=16)

# ─── Dense(1) ────────────────────────────────────────────────────────────────
block(X_DE, Y_MID, BW_DE, BH * 0.80, C_DENSE)
lbl_above(X_DE, Y_MID, BH * 0.80, "Dense (1)", fs=15)

arrow(X_DE + BW_DE/2, Y_MID, X_TH - BW_TH/2, Y_MID, lw=2.0, ms=16)

# ─── tanh activation ─────────────────────────────────────────────────────────
block(X_TH, Y_MID, BW_TH, BH * 0.80, C_TANH)
lbl_above(X_TH, Y_MID, BH * 0.80, "tanh", fs=15.5, color="#111111")
ax.text(X_TH, Y_MID,
        r"$(-1,\,1)$",
        ha="center", va="center", fontsize=13, color="white",
        fontweight="bold", zorder=5)

arrow(X_TH + BW_TH/2, Y_MID, X_OUT - 0.55, Y_MID, lw=2.0, ms=16)

# ─── Output score badge ───────────────────────────────────────────────────────
score_r = 0.62
score_circle = plt.Circle((X_OUT, Y_MID), score_r, color=C_SCORE,
                           zorder=3, linewidth=2, edgecolor="white")
ax.add_patch(score_circle)
ax.text(X_OUT, Y_MID + 0.08, "D(x)",
        ha="center", va="center", fontsize=15.5,
        fontweight="bold", color="white", zorder=5)
ax.text(X_OUT, Y_MID - 0.25, "score",
        ha="center", va="center", fontsize=12,
        color="#AAAACC", zorder=5)

# legend: 1=real, 0=fake
ax.text(X_OUT, Y_MID - score_r - 0.25,
        "1 = real  |  0 = fake",
        ha="center", va="top", fontsize=12.5,
        color="#555555", style="italic", zorder=5)
ax.text(X_OUT, Y_MID + score_r + 0.14,
        "Realness score",
        ha="center", va="bottom", fontsize=15.5,
        fontweight="bold", color="#111111", zorder=5)

# ─── HeUniform + LSGAN annotation box ────────────────────────────────────────
ann_x, ann_y = 9.0, 1.30
ann_w, ann_h = 8.0, 0.95
ann_box = FancyBboxPatch(
    (ann_x - ann_w/2, ann_y - ann_h/2), ann_w, ann_h,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=1.4, edgecolor="#AAAACC",
    facecolor="#F0F0FA", alpha=0.92, zorder=3)
ax.add_patch(ann_box)
ax.text(ann_x, ann_y + 0.17,
        r"HeUniform initialisation  |  LSGAN objective: targets 1 (real) and 0 (generated)",
        ha="center", va="center", fontsize=13,
        color="#333355", zorder=5)
ax.text(ann_x, ann_y - 0.20,
        r"tanh bounds output to $(-1,\,1)$, preventing abrupt gradient updates "
        r"(Jeffrey et al., 2023)",
        ha="center", va="center", fontsize=12.5,
        color="#555577", style="italic", zorder=5)

# ─── Bottom note ─────────────────────────────────────────────────────────────
ax.text(FW / 2, 0.32,
        "Depth reduced from 5 to 2 stride-2 levels to accommodate the 6×12 input; "
        "filter sequence 16→32 scaled proportionally",
        ha="center", va="bottom", fontsize=14,
        color="#777777", style="italic", zorder=5)

# ─── Save ────────────────────────────────────────────────────────────────────
fig.savefig(OUT, bbox_inches="tight", dpi=200, facecolor="white")
plt.close(fig)
print("Saved:", OUT)
