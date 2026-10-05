# -*- coding: utf-8 -*-
"""
gen_calibration_examples_8010.py

Regenera calibration_example_baseline.pdf y calibration_example_supervised.pdf
(Secciones 4.7 y 4.10) usando los checkpoints del split 80/10/10. Replica
FIELMENTE el diseno original: 3 paneles APILADOS VERTICALMENTE (raw /
calibrado / referencia sintetica), parte real, matriz 6x12 con linea
divisoria blanca entre las columnas B1-B6 y B1-B6 (dos bloques A->B / B->A),
colormap tipo parula, UNA sola escala de color compartida entre los 3
paneles (simetrica, calculada sobre los 3 juntos), titulos en negrita con
notacion matematica igual al original.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python gen_calibration_examples_8010.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cyclegan_model import CycleGan
from data_io import (Normalizer, cosine_sim_batch, find_analog_index,
                      load_compressed_datasets, tensor_to_complex)

OUT_DIR = "./figures_8010"
os.makedirs(OUT_DIR, exist_ok=True)

_PARULA_PTS = [
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424], [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661], [0.5709, 0.7485, 0.4494], [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188], [0.6834, 0.7435, 0.4044], [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768], [0.7858, 0.7356, 0.3633], [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360], [0.8824, 0.7274, 0.3217], [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886], [0.9739, 0.7314, 0.2666], [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164], [0.9955, 0.7861, 0.1967], [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633], [0.9697, 0.8481, 0.1475], [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132], [0.9598, 0.9218, 0.0948], [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
]
PARULA = LinearSegmentedColormap.from_list("parula", _PARULA_PTS)

ANT_LABELS = ["B1", "B2", "B3", "B4", "B5", "B6", "B1", "B2", "B3", "B4", "B5", "B6"]


def plot_stacked_three(out_path, raw_re, cal_re, syn_re):
    vmax = max(np.abs(raw_re).max(), np.abs(cal_re).max(), np.abs(syn_re).max())
    vmin = -vmax

    panels = [
        (raw_re, r"$\mathbf{Raw\ experimental\ input\ Re(\Delta S)}$"),
        (cal_re, r"$\mathbf{Calibrated\ output\ Re(G_{E \to S}(\Delta S))}$"),
        (syn_re, r"$\mathbf{True\ synthetic\ reference\ Re(\Delta E^{sim})}$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 12.5))
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, cmap=PARULA, vmin=vmin, vmax=vmax, aspect="equal")
        ax.axvline(5.5, color="white", linewidth=2.2)
        ax.set_xticks(range(12))
        ax.set_xticklabels(ANT_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"A{i+1}" for i in range(6)], fontsize=9)
        ax.set_xlabel("Antenna j", fontsize=10)
        ax.set_ylabel("Antenna i", fontsize=10)
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("linear amplitude (Re, a.u.)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def baseline_example():
    model_dir = "./checkpoints/Baseline_8010_3way"
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    A_tensor, B_tensor, A_meta, B_meta, _a, _b, _fA, _fB = load_compressed_datasets(
        cfg["synthetic_h5_path"], cfg["experimental_h5_path"])
    split = dict(np.load(os.path.join(model_dir, "split_indices.npz")))
    test_idx_B = split["B_test"]

    norm_A = Normalizer.from_dict(cfg["norm_A"])
    norm_B = Normalizer.from_dict(cfg["norm_B"])
    img_shape = A_tensor.shape[1:]
    cg = CycleGan(img_shape, model_dir, gen_depth=cfg["gen_depth"], disc_depth=cfg["disc_depth"],
                  apply_output_mask=cfg["apply_output_mask"])
    cg.load()

    pairs = []
    for b_idx in test_idx_B:
        a_idx = find_analog_index(B_meta[b_idx], A_meta, pos_tol_mm=1.0, match_position=True)
        if a_idx is not None:
            pairs.append((b_idx, a_idx))
    b_idx_arr = np.array([p[0] for p in pairs])
    a_idx_arr = np.array([p[1] for p in pairs])

    B_norm_all = norm_B.forward(B_tensor)
    calibrated_norm = cg.calibrate(B_norm_all[b_idx_arr])
    calibrated_complex = tensor_to_complex(norm_A.inverse(calibrated_norm))
    A_complex = tensor_to_complex(A_tensor)[a_idx_arr]
    B_raw_complex = tensor_to_complex(B_tensor)[b_idx_arr]

    cos = cosine_sim_batch(calibrated_complex, A_complex)
    best = int(np.argmax(cos))
    print(f"[baseline] representative sample: cos={cos[best]:.3f}")

    plot_stacked_three(os.path.join(OUT_DIR, "calibration_example_baseline.pdf"),
                        np.real(B_raw_complex[best]), np.real(calibrated_complex[best]),
                        np.real(A_complex[best]))


def supervised_example():
    model_dir = "./checkpoints/TTSup_8010_3way"
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    A_tensor, B_tensor, A_meta, B_meta, _a, _b, _fA, _fB = load_compressed_datasets(
        cfg["synthetic_h5_path"], cfg["experimental_h5_path"])
    split = dict(np.load(os.path.join(model_dir, "split_indices.npz")))
    test_idx = split["test"]

    norm_A = Normalizer.from_dict(cfg["norm_A"])
    norm_B = Normalizer.from_dict(cfg["norm_B"])
    img_shape = A_tensor.shape[1:]
    cg = CycleGan(img_shape, model_dir, gen_depth=cfg["gen_depth"], disc_depth=cfg["disc_depth"],
                  apply_output_mask=cfg["apply_output_mask"])
    cg.load()

    B_norm_test = norm_B.forward(B_tensor[test_idx])
    calibrated_norm = cg.calibrate(B_norm_test)
    calibrated_complex = tensor_to_complex(norm_A.inverse(calibrated_norm))
    A_complex = tensor_to_complex(A_tensor[test_idx])
    B_raw_complex = tensor_to_complex(B_tensor[test_idx])

    cos = cosine_sim_batch(calibrated_complex, A_complex)
    best = int(np.argmax(cos))
    print(f"[supervised] representative sample: cos={cos[best]:.3f}")

    plot_stacked_three(os.path.join(OUT_DIR, "calibration_example_supervised.pdf"),
                        np.real(B_raw_complex[best]), np.real(calibrated_complex[best]),
                        np.real(A_complex[best]))


if __name__ == "__main__":
    baseline_example()
    supervised_example()
    print("\nDone.")
