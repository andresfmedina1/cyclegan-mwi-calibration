#!/usr/bin/env python3
"""
draw_data_pipeline.py  —  Fig. 4.4 replacement
Pipeline: |ΔS^meas| 12×12 (dB) → compress → |ΔS^meas| 6×12 (dB) → Re / Im split → 6×12×2
All four panels share viridis colormap. The 12×12 and 6×12 share the same dB scale.

Figure is designed at ~textwidth (13 in) so LaTeX scales it ≈0.5x → fonts at 20pt appear
as ~10pt in the final thesis document.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import scipy.io as sio

CDIR = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(CDIR, "fig_4_4_data_pipeline.pdf")

# ── load data ─────────────────────────────────────────────────────────────────
d     = sio.loadmat(os.path.join(CDIR, "acq_0015_sample.mat"))
raw_c = d["raw_real"].astype(float) + 1j * d["raw_imag"].astype(float)  # ΔS 12×12

def compress(mat):
    """12×12 → 6×12: [A→B | B→A^T]"""
    return np.concatenate([mat[:6, 6:], mat[6:, :6].T], axis=1)

ds6   = compress(raw_c)
ds6re = ds6.real
ds6im = ds6.imag

# ── dB magnitude matrices ─────────────────────────────────────────────────────
DB_FLOOR = -90.0

def to_db(mat, floor=DB_FLOOR):
    return np.clip(20.0 * np.log10(np.abs(mat) + 1e-20), floor, None)

ds12_db = to_db(raw_c)
ds6_db  = to_db(ds6)

diag_val = ds6_db.max() + 5.0
for k in range(12):
    ds12_db[k, k] = diag_val

VMIN_DB = DB_FLOOR
VMAX_DB = diag_val

symlin = max(np.abs(ds6re).max(), np.abs(ds6im).max()) * 1.05

CMAP = "viridis"

# ── figure — designed at ~13 in wide (≈2× A4 textwidth) ──────────────────────
FW, FH = 13, 8
fig = plt.figure(figsize=(FW, FH), facecolor="white")

gs = GridSpec(
    2, 5, figure=fig,
    left=0.055, right=0.960,
    top=0.85,   bottom=0.14,
    width_ratios=[2.8, 0.55, 2.1, 0.55, 2.0],
    hspace=0.50,
    wspace=0.05,
)

ax12  = fig.add_subplot(gs[:, 0])
ax6   = fig.add_subplot(gs[:, 2])
ax_re = fig.add_subplot(gs[0, 4])
ax_im = fig.add_subplot(gs[1, 4])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — 12×12  |ΔS^meas|  dB
# ══════════════════════════════════════════════════════════════════════════════
im12 = ax12.imshow(ds12_db, cmap=CMAP,
                   vmin=VMIN_DB, vmax=VMAX_DB,
                   aspect="equal", origin="upper", zorder=1)

ax12.add_patch(Rectangle((-0.5, -0.5), 6, 6,
               facecolor="none", edgecolor="#CCCCCC",
               hatch="////", linewidth=0, zorder=2, alpha=0.5))
ax12.add_patch(Rectangle((5.5, 5.5), 6, 6,
               facecolor="none", edgecolor="#CCCCCC",
               hatch="////", linewidth=0, zorder=2, alpha=0.5))

ax12.axhline(5.5, color="white", lw=1.6, zorder=3)
ax12.axvline(5.5, color="white", lw=1.6, zorder=3)

tbkw = dict(ha="center", va="center", fontsize=11, zorder=6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.82))
ax12.text(2.5, 2.5,  "A–A\n(switch\nisolation)",  color="#220055",  **tbkw)
ax12.text(8.5, 8.5,  "B–B\n(switch\nisolation)",  color="#220055",  **tbkw)
ax12.text(8.5, 2.5,  "A→B\n(valid)",               color="#1A5C2A",  **tbkw)
ax12.text(2.5, 8.5,  "B→A\n(valid)",               color="#1A5C2A",  **tbkw)

ax12.annotate(
    "Diagonal\n(self-coupling,\nexcluded)",
    xy=(5.0, 5.0), xytext=(9.5, 4.5),
    fontsize=9, color="#111111", ha="center", zorder=7,
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#AAAAAA", alpha=0.90),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

lbl12 = [f"A{j+1}" if j < 6 else f"B{j-5}" for j in range(12)]
ax12.set_xticks(range(12)); ax12.set_xticklabels(lbl12, fontsize=8, rotation=45, ha="right")
ax12.set_yticks(range(12)); ax12.set_yticklabels(lbl12, fontsize=8)
ax12.set_xlabel("Antenna $j$ (receiver)", fontsize=9, labelpad=4)
ax12.set_ylabel("Antenna $i$ (transmitter)", fontsize=9)
ax12.set_title(r"$|\Delta S^{\mathrm{meas}}|$ — full 12×12 (dB)",
               fontsize=11, fontweight="bold", pad=8)

cb12 = fig.colorbar(im12, ax=ax12, fraction=0.032, pad=0.02)
cb12.set_label("dB", fontsize=9)
cb12.ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — 6×12  |ΔS^meas|  dB
# ══════════════════════════════════════════════════════════════════════════════
im6 = ax6.imshow(ds6_db, cmap=CMAP,
                 vmin=VMIN_DB, vmax=VMAX_DB,
                 aspect="auto", origin="upper", zorder=1)

ax6.axvline(5.5, color="white", lw=1.5, linestyle="--", zorder=3)

tbkw6 = dict(ha="center", va="center", fontsize=11, zorder=6, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="none", alpha=0.82))
ax6.text(2.5, 2.5, "A→B", color="#1A5C2A", **tbkw6)
ax6.text(8.5, 2.5, r"B→A$^\top$", color="#1A5C2A", **tbkw6)

lbl6 = [f"B{j+1}" if j < 6 else f"B{j-5}" for j in range(12)]
ax6.set_xticks(range(12)); ax6.set_xticklabels(lbl6, fontsize=8, rotation=45, ha="right")
ax6.set_yticks(range(6));  ax6.set_yticklabels([f"A{i+1}" for i in range(6)], fontsize=8)
ax6.set_xlabel("Column $j$", fontsize=9, labelpad=4)
ax6.set_title(r"$|\Delta S^{\mathrm{meas}}|$ — compressed 6×12 (dB)",
              fontsize=11, fontweight="bold", pad=8)

cb6 = fig.colorbar(im6, ax=ax6, fraction=0.040, pad=0.02)
cb6.set_label("dB", fontsize=9)
cb6.ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3a — Real part
# ══════════════════════════════════════════════════════════════════════════════
im_re = ax_re.imshow(ds6re, cmap=CMAP,
                     vmin=-symlin, vmax=symlin,
                     aspect="auto", origin="upper")
ax_re.set_xticks([]); ax_re.set_yticks(range(6))
ax_re.set_yticklabels([f"A{i+1}" for i in range(6)], fontsize=8)
ax_re.set_title("Channel 0 — Real", fontsize=11, fontweight="bold")
fig.colorbar(im_re, ax=ax_re, fraction=0.06, pad=0.02).ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3b — Imaginary part
# ══════════════════════════════════════════════════════════════════════════════
im_im = ax_im.imshow(ds6im, cmap=CMAP,
                     vmin=-symlin, vmax=symlin,
                     aspect="auto", origin="upper")
ax_im.set_xticks([]); ax_im.set_yticks(range(6))
ax_im.set_yticklabels([f"A{i+1}" for i in range(6)], fontsize=8)
ax_im.set_title("Channel 1 — Imag.", fontsize=11, fontweight="bold")
ax_im.set_xlabel(r"6×12×2 input tensor", fontsize=9, labelpad=4,
                 color="#444444", style="italic")
fig.colorbar(im_im, ax=ax_im, fraction=0.06, pad=0.02).ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — drawn within gap axes
# ══════════════════════════════════════════════════════════════════════════════
arrkw = dict(xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0,
                             mutation_scale=18, shrinkA=2, shrinkB=2))

ax_g1 = fig.add_subplot(gs[:, 1])
ax_g1.axis("off")
ax_g1.annotate("", xy=(0.88, 0.50), xytext=(0.48, 0.50), **arrkw)
ax_g1.text(0.50, 0.85, "compress\n(cross-group\nonly)", ha="center", va="bottom",
           fontsize=8, color="#222222", transform=ax_g1.transAxes,
           bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC"))

ax_g2 = fig.add_subplot(gs[:, 3])
ax_g2.axis("off")
ax_g2.annotate("", xy=(0.92, 0.76), xytext=(0.50, 0.50), **arrkw)
ax_g2.annotate("", xy=(0.92, 0.24), xytext=(0.50, 0.50), **arrkw)
ax_g2.text(0.70, 0.53, "Re / Im\nsplit", ha="center", va="bottom",
           fontsize=8, color="#222222", transform=ax_g2.transAxes,
           bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC"))

# ── title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    r"Data compression pipeline: "
    r"$|\Delta S^{\mathrm{meas}}|$ ($12{\times}12$, dB) "
    r"$\rightarrow$ $|\Delta S^{\mathrm{meas}}|$ ($6{\times}12$, dB) "
    r"$\rightarrow$ $6{\times}12{\times}2$ (Re / Im, linear)",
    fontsize=11, fontweight="bold", y=0.97)

fig.savefig(OUT, bbox_inches="tight", dpi=300, facecolor="white")
plt.close(fig)
print("Saved:", OUT)
