# -*- coding: utf-8 -*-
"""
gen_chapter4_figures_8010.py

Regenera las figuras de las Secciones 4.7-4.11 de la tesis, usando los
checkpoints reentrenados con el split 80/10/10. Replica FIELMENTE la
organizacion/diseno de las figuras originales (extraidas de
images/fig_4_*.pdf en el zip de la tesis) -- mismo layout de paneles,
mismos titulos, mismos colores por defecto de matplotlib -- solo cambia
los datos subyacentes por los del split 80/10/10.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python gen_chapter4_figures_8010.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "./figures_8010"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# matplotlib color-cycle defaults (tab10), usados tal cual el original
C0, C1, C2, C3, C4 = "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"


def load_history(model_dir):
    with open(os.path.join(model_dir, "history.json")) as f:
        return json.load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


AFTER = load_json("./eval_train_test_full_results.json")
BEFORE = load_json("./eval_before_8010_results.json")
LRONLY_AFTER = load_json("./lronly_after_results.json")


def get_after(name, split):
    return AFTER[name][split]


def get_before(name, split):
    return BEFORE[name][split]


def get_lronly_after(split):
    return LRONLY_AFTER[split]


# =========================================================================
# Figure 4.4a -- Baseline discriminator loss (collapse)
# =========================================================================
def fig_4_4a_baseline_dloss():
    h = load_history("./checkpoints/Baseline_8010_3way")
    every = h.get("verbose_every", 5) if isinstance(h, dict) and "verbose_every" in h else 5
    d = h["d_loss"]
    epochs_full = np.arange(1, len(d) + 1)
    # puntos marcados cada 5 epocas, igual que el original (verbose_every=5)
    mark_idx = np.arange(0, len(d), 5)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(epochs_full, d, "-o", color=C0, markevery=list(mark_idx), markersize=4.5,
            linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("D loss")
    ax.set_title(r"Discriminator loss ($\gamma_{cyc}=\gamma_{id}=100$, shared lr)"
                 "\ncollapses within the first epochs -- 80/10/10 split")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_4_4a_baseline_dloss.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


# =========================================================================
# Figure 4.5a -- Baseline / lr_D reduced / Final: D loss (3 panels) + bars
# =========================================================================
def fig_4_5a_tuning_comparison():
    h_base = load_history("./checkpoints/Baseline_8010_3way")
    h_lr = load_history("./checkpoints/LRonly_8010_3way")
    h_run8 = load_history("./checkpoints/Run8_8010_3way")

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(range(1, len(h_base["d_loss"]) + 1), h_base["d_loss"], color=C0, linewidth=1.6)
    ax0.set_title("Baseline\n" r"($\gamma$=100, shared lr)")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("D loss")
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(range(1, len(h_lr["d_loss"]) + 1), h_lr["d_loss"], color=C0, linewidth=1.6)
    ax1.set_title(r"lr$_D$ reduced" "\n" r"($\gamma$=100)")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(range(1, len(h_run8["d_loss"]) + 1), h_run8["d_loss"], color=C0, linewidth=1.6)
    ax2.set_title(r"Final ($\gamma$=20, lr$_D$/5, scale=3)")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)

    # delta bars (train split, single y-axis, verde=cos, rojo=rmse)
    names = ["Baseline_8010_3way", "LRonly_8010_3way", "Run8_8010_3way"]
    labels = ["Baseline", r"lr$_D$ reduced", "Final"]
    d_cos, d_rmse = [], []
    for n in names:
        b = get_before(n, "train")
        a = get_lronly_after("train") if n == "LRonly_8010_3way" else get_after(n, "train")
        d_cos.append(a["cos"] - b["cos"])
        d_rmse.append(a["rmse"] - b["rmse"])

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(labels))
    w = 0.32
    ax3.bar(x - w / 2, d_cos, width=w, label="cosine similarity", color="tab:green")
    ax3.bar(x + w / 2, d_rmse, width=w, label="relative RMSE", color="tab:red")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Change (after $-$ before calibration)")
    title = "Cosine improves, RMSE does not" if d_rmse[-1] > 0 else \
        "Cosine improves; RMSE only improves once tuning is complete"
    ax3.set_title(title)
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("Effect of hyperparameter tuning on discriminator stability and "
                 "calibration quality -- 80/10/10 split", fontsize=13, y=0.99)
    out = os.path.join(OUT_DIR, "fig_4_5a_tuning_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


# =========================================================================
# Generic "3-top-panels + 1-wide-bottom-panel" layout, used by 4.5b / 4.6
# =========================================================================
def _three_plus_wide(history, suptitle, out_name, d_title="Discriminator loss",
                      d_vline=None, d_vline_text=None):
    n_epochs = len(history["g_loss"])
    epochs = range(1, n_epochs + 1)

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.5, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, history["g_loss"], label="G total", color=C0, linewidth=1.3)
    ax0.plot(epochs, history["g_adv"], label="adversarial", color=C1, linewidth=1.1)
    ax0.plot(epochs, history["g_cycle"], label="cycle", color=C2, linewidth=1.1)
    ax0.plot(epochs, history["g_identity"], label="identity", color=C3, linewidth=1.1)
    ax0.plot(epochs, history["g_scale"], label="scale", color=C4, linewidth=1.1)
    ax0.set_title("Generator losses (training)")
    ax0.set_xlabel("epoch")
    ax0.legend(fontsize=8.3)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(epochs, history["d_loss"], color=C0, linewidth=1.4)
    if d_vline is not None:
        ax1.axvline(d_vline, color=C1, linestyle="--", linewidth=1.4)
        ax1.text(d_vline + n_epochs * 0.02, ax1.get_ylim()[1] * 0.55, d_vline_text,
                  color=C1, fontsize=9, va="top")
    ax1.set_title(d_title)
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, history["g_cycle"], label="cycle (train)", color=C0)
    ax2.plot(epochs, history["val_cycle"], label="cycle (val)", color=C0, linestyle="--")
    ax2.plot(epochs, history["g_identity"], label="identity (train)", color=C1)
    ax2.plot(epochs, history["val_identity"], label="identity (val)", color=C1, linestyle="--")
    ax2.set_title("Train vs. validation (cycle + identity)")
    ax2.set_xlabel("epoch")
    ax2.legend(fontsize=8.3)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(epochs, history["g_scale"], label="scale (train)", color=C2)
    ax3.plot(epochs, history["val_scale"], label="scale (val)", color=C2, linestyle="--")
    ax3.set_title("Train vs. validation (scale)")
    ax3.set_xlabel("epoch")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle(suptitle, fontsize=13.5, y=0.99)
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def fig_4_5b_full_training_curves():
    h = load_history("./checkpoints/Run8_8010_3way")
    _three_plus_wide(h, "Complete training curves: final detection "
                      "configuration -- 80/10/10 split",
                      "fig_4_5b_full_training_curves.pdf")


def fig_4_6_tt_late_collapse():
    h = load_history("./checkpoints/TTUnsup_8010_3way")  # extended to 100 epochs
    _three_plus_wide(h, "Complete training curves: monitoring (unsupervised), "
                      "0-100 epochs -- 80/10/10 split",
                      "fig_4_6_tt_late_collapse.pdf",
                      d_title="Discriminator loss (late collapse)",
                      d_vline=20, d_vline_text="promoted\ncheckpoint\n(epoch 20)")


# =========================================================================
# Figure 4.7 -- Unsupervised vs supervised: 3 D-loss panels + wide bars
# =========================================================================
def fig_4_7_supervised_dloss():
    h_unsup = load_history("./checkpoints/TTUnsup_8010_3way")  # 100 epochs
    h_sup = load_history("./checkpoints/TTSup_8010_3way")      # 200 epochs

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.32)

    n_unsup = len(h_unsup["d_loss"])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(range(1, n_unsup + 1), h_unsup["d_loss"], color=C0, linewidth=1.5)
    ax0.axvline(20, color=C1, linestyle="--", linewidth=1.3)
    ax0.text(20 + n_unsup * 0.02, ax0.get_ylim()[1] * 0.6, "promoted\ncheckpoint",
             color=C1, fontsize=9, va="top")
    ax0.set_title("Unsupervised (0-100 ep.)\ncollapses late")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("D loss")
    ax0.grid(alpha=0.3)

    n_sup = len(h_sup["d_loss"])
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(range(1, n_sup + 1), h_sup["d_loss"], color=C0, linewidth=1.5)
    ax1.set_title("Supervised (0-200 ep.)\nnever collapses")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(range(1, n_unsup + 1), h_unsup["d_loss"], color=C3, label="unsupervised", linewidth=1.5)
    ax2.plot(range(1, n_unsup + 1), h_sup["d_loss"][:n_unsup], color=C0, label="supervised", linewidth=1.5)
    ax2.set_title("Direct comparison\n(first 100 epochs)")
    ax2.set_xlabel("epoch")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    names = ["TTUnsup_8010_3way", "TTSup_8010_3way"]
    labels = ["Unsupervised", "Supervised"]
    d_cos, d_rmse = [], []
    for n in names:
        b, a = get_before(n, "train"), get_after(n, "train")
        d_cos.append(a["cos"] - b["cos"])
        d_rmse.append(a["rmse"] - b["rmse"])

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(labels))
    w = 0.32
    ax3.bar(x - w / 2, d_cos, width=w, label=r"$\Delta$ cosine similarity", color="tab:green")
    ax3.bar(x + w / 2, d_rmse, width=w, label=r"$\Delta$ relative RMSE", color="tab:red")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Change (after $-$ before calibration)")
    title = ("Only the supervised loss yields a genuine RMSE improvement"
              if d_rmse[-1] < 0 <= d_rmse[0] else
              "Supervision brings a larger cosine gain and a smaller RMSE penalty")
    ax3.set_title(title)
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("Discriminator stability and calibration quality: unsupervised vs. "
                 "supervised monitoring -- 80/10/10 split", fontsize=13, y=0.99)
    out = os.path.join(OUT_DIR, "fig_4_7_supervised_dloss.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


# =========================================================================
# Figure 4.7c -- Supervised TT: 2x3 grid, last panel = text summary
# =========================================================================
def fig_4_7c_full_curves():
    h = load_history("./checkpoints/TTSup_8010_3way")
    n_epochs = len(h["g_loss"])
    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))

    ax = axes[0, 0]
    ax.plot(epochs, h["g_loss"], label="G total", color=C0, linewidth=1.2)
    ax.plot(epochs, h["g_adv"], label="adversarial", color=C1, linewidth=1.0)
    ax.plot(epochs, h["g_cycle"], label="cycle", color=C2, linewidth=1.0)
    ax.plot(epochs, h["g_identity"], label="identity", color=C3, linewidth=1.0)
    ax.plot(epochs, h["g_scale"], label="scale", color=C4, linewidth=1.0)
    ax.plot(epochs, h["g_supervised"], label="supervised", color="tab:brown", linewidth=1.0)
    ax.set_title("Generator losses (training)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7.8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, h["d_loss"], color=C0, linewidth=1.4)
    ax.set_title("Discriminator loss (never collapses)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, h["g_supervised"], label="supervised (train)", color=C3)
    ax.plot(epochs, h["val_supervised"], label="supervised (val)", color=C3, linestyle="--")
    ax.set_title("Train vs. validation (supervised)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8.3)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, h["g_cycle"], label="cycle (train)", color=C0)
    ax.plot(epochs, h["val_cycle"], label="cycle (val)", color=C0, linestyle="--")
    ax.plot(epochs, h["g_identity"], label="identity (train)", color=C1)
    ax.plot(epochs, h["val_identity"], label="identity (val)", color=C1, linestyle="--")
    ax.set_title("Train vs. validation (cycle + identity)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8.3)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, h["g_scale"], label="scale (train)", color=C2)
    ax.plot(epochs, h["val_scale"], label="scale (val)", color=C2, linestyle="--")
    ax.set_title("Train vs. validation (scale)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8.3)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    d0, d1 = h["d_loss"][0], h["d_loss"][-1]
    box_text = (f"Supervised monitoring\n({n_epochs} epochs)\n\n"
                f"D loss: {d0:.2f} $\\to$ {d1:.2f}\n(never collapses)\n\n"
                "Only configuration with a\ndirect L1 supervised term\n"
                "(paired data, Sec. 4.10)")
    ax.text(0.5, 0.5, box_text, ha="center", va="center", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="0.92", edgecolor="0.6"),
            transform=ax.transAxes)

    fig.suptitle(f"Complete training curves: supervised monitoring "
                 f"(0-{n_epochs} epochs) -- 80/10/10 split", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "fig_4_7c_full_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


# =========================================================================
# Figure 4.8 -- Generalization / partition flowchart
# =========================================================================
def _box(ax, xy, w, h, text, facecolor, edgecolor, fontsize=10.5, bold_first_line=True):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.02",
                            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6)
    ax.add_patch(patch)
    lines = text.split("\n")
    if bold_first_line:
        ax.text(x, y + h * 0.28, lines[0], ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=edgecolor)
        rest = "\n".join(lines[1:])
        if rest:
            ax.text(x, y - h * 0.12, rest, ha="center", va="center",
                     fontsize=fontsize - 1.5, color=edgecolor)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=edgecolor)


def _arrow(ax, xy_from, xy_to, color="0.35"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.4,
                                 shrinkA=2, shrinkB=2))


def fig_4_8_generalization_split():
    with open("./checkpoints/TTSup_8010_3way/config.json") as f:
        cfg = json.load(f)
    n_train, n_val, n_test = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    n_isc2 = 10274
    n_avail = n_train + n_val + n_test
    n_total = n_avail + n_isc2

    fig, ax = plt.subplots(figsize=(12.5, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    root_xy = (6, 7)
    isc2_xy = (1.9, 5)
    avail_xy = (7.8, 5)
    train_xy = (5.6, 2.6)
    val_xy = (8.1, 2.6)
    test_xy = (10.4, 2.6)

    _box(ax, root_xy, 3.6, 1.0, f"Full monitoring dataset\nN = {n_total:,}",
         "#ded6f0", "#5b3fa0")
    _box(ax, isc2_xy, 3.0, 1.35,
         f"Isc2 -- pure alcohol (holdout)\nN = {n_isc2:,} ({100*n_isc2/n_total:.0f}%)\n"
         "excluded from training AND\nnormalization statistics",
         "#f7dcc4", "#c1670e", fontsize=9.8)
    _box(ax, avail_xy, 3.6, 1.0, f"Available (other liquids)\nN = {n_avail:,} "
         f"({100*n_avail/n_total:.0f}%)", "#c9e3f2", "#1f6fa6")
    _box(ax, train_xy, 2.0, 0.9, f"Train\nN = {n_train:,}", "#d3ecd3", "#3a8a3a", fontsize=10)
    _box(ax, val_xy, 1.7, 0.9, f"Val\nN = {n_val:,}\n(random)", "#f3e6c4", "#a6841f", fontsize=9.3)
    _box(ax, test_xy, 1.9, 0.9, f"Test (in-dist.)\nN = {n_test:,}\n(random)",
         "#f3e6c4", "#a6841f", fontsize=9.3)

    _arrow(ax, (root_xy[0] - 0.5, root_xy[1] - 0.5), (isc2_xy[0] + 0.3, isc2_xy[1] + 0.68))
    _arrow(ax, (root_xy[0] + 0.5, root_xy[1] - 0.5), (avail_xy[0] - 0.3, avail_xy[1] + 0.5))
    _arrow(ax, (avail_xy[0] - 1.1, avail_xy[1] - 0.5), (train_xy[0], train_xy[1] + 0.45))
    _arrow(ax, (avail_xy[0], avail_xy[1] - 0.5), (val_xy[0], val_xy[1] + 0.45))
    _arrow(ax, (avail_xy[0] + 1.1, avail_xy[1] - 0.5), (test_xy[0], test_xy[1] + 0.45))

    _arrow(ax, (isc2_xy[0], isc2_xy[1] - 0.68), (isc2_xy[0], 0.9), color="#c1670e")
    ax.text(isc2_xy[0], 0.55, '"generalization" result\n(pure alcohol, never seen)',
            ha="center", va="center", fontsize=9.5, color="#c1670e")

    _arrow(ax, (test_xy[0], test_xy[1] - 0.45), (test_xy[0], 0.9), color="#a6841f")
    ax.text(test_xy[0], 0.55, '"in-distribution"\nresult', ha="center", va="center",
            fontsize=9.5, color="#a6841f")

    ax.set_title("Generalization assessment: full-category holdout vs. 80/10/10 "
                  "train/val/test split", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_4_8_generalization_split.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    fig_4_4a_baseline_dloss()
    fig_4_5a_tuning_comparison()
    fig_4_5b_full_training_curves()
    fig_4_6_tt_late_collapse()
    fig_4_7_supervised_dloss()
    fig_4_7c_full_curves()
    fig_4_8_generalization_split()
    print("\nDone (excluding calibration_example_* images, generated separately).")
