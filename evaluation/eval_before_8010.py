"""
eval_before_8010.py

Calcula las metricas "antes de calibrar" (raw, sin pasar por el
generador -- comparacion directa B vs. A emparejado, en escala fisica,
igual que rmse_before/cos_before de las demas evaluaciones) para los 5
checkpoints del split 80/10/10 usados en las figuras del Capitulo 4:
Baseline, LRonly, Run8 (unpaired/target-background), TTUnsup, TTSup
(paired/target-target). Train y test.

Se combina con las metricas "despues" ya calculadas
(eval_train_test_full_results.json) para las barras de Delta
cos/RMSE de las Figuras 4.5a y 4.7.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python eval_before_8010.py
"""
import json
import os

import numpy as np

from data_io import (Normalizer, cosine_sim_batch, find_analog_index,
                      load_compressed_datasets, relative_rmse_batch,
                      ssim_batch, tensor_to_complex)

CHECKPOINTS = {
    "Baseline_8010_3way": "unpaired",
    "LRonly_8010_3way":   "unpaired",
    "Run8_8010_3way":     "unpaired",
    "TTUnsup_8010_3way":  "paired",
    "TTSup_8010_3way":    "paired",
}


def load_split(model_dir, kind):
    d = dict(np.load(os.path.join(model_dir, "split_indices.npz")))
    if kind == "unpaired":
        return {"A": {"test": d["A_test"], "train": d["A_train"]},
                "B": {"test": d["B_test"], "train": d["B_train"]}}
    return {"test": d["test"], "train": d["train"]}


def _before_unpaired(A_tensor, B_tensor, A_meta, B_meta, b_idx_list):
    pairs = []
    for b_idx in b_idx_list:
        a_idx = find_analog_index(B_meta[b_idx], A_meta, pos_tol_mm=1.0, match_position=True)
        if a_idx is not None:
            pairs.append((b_idx, a_idx))
    b_idx_arr = np.array([p[0] for p in pairs])
    a_idx_arr = np.array([p[1] for p in pairs])

    B_complex = tensor_to_complex(B_tensor)[b_idx_arr]
    A_complex = tensor_to_complex(A_tensor)[a_idx_arr]

    rmse = relative_rmse_batch(B_complex, A_complex)
    cos = cosine_sim_batch(B_complex, A_complex)
    ssim_re, ssim_im = ssim_batch(A_complex, B_complex, ref=A_complex)
    return dict(n=len(pairs), rmse=float(np.mean(rmse)), cos=float(np.mean(cos)),
                ssim_re=float(np.mean(ssim_re)), ssim_im=float(np.mean(ssim_im)))


def _before_paired(A_tensor, B_tensor, idx):
    B_complex = tensor_to_complex(B_tensor[idx])
    A_complex = tensor_to_complex(A_tensor[idx])
    rmse = relative_rmse_batch(B_complex, A_complex)
    cos = cosine_sim_batch(B_complex, A_complex)
    ssim_re, ssim_im = ssim_batch(A_complex, B_complex, ref=A_complex)
    return dict(n=int(len(idx)), rmse=float(np.mean(rmse)), cos=float(np.mean(cos)),
                ssim_re=float(np.mean(ssim_re)), ssim_im=float(np.mean(ssim_im)))


def main():
    results = {}
    for name, kind in CHECKPOINTS.items():
        model_dir = f"./checkpoints/{name}"
        if not os.path.isdir(model_dir):
            print(f"[skip] {model_dir} no existe todavia")
            continue
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        A_tensor, B_tensor, A_meta, B_meta, _a, _b, _fA, _fB = load_compressed_datasets(
            cfg["synthetic_h5_path"], cfg["experimental_h5_path"])
        split = load_split(model_dir, kind)

        print(f"\nEvaluando (raw, antes de calibrar) {name}...")
        if kind == "unpaired":
            train_r = _before_unpaired(A_tensor, B_tensor, A_meta, B_meta, split["B"]["train"])
            test_r = _before_unpaired(A_tensor, B_tensor, A_meta, B_meta, split["B"]["test"])
        else:
            train_r = _before_paired(A_tensor, B_tensor, split["train"])
            test_r = _before_paired(A_tensor, B_tensor, split["test"])
        results[name] = {"train": train_r, "test": test_r}
        print(f"  train: rmse={train_r['rmse']:.4f} cos={train_r['cos']:.4f}  "
              f"test: rmse={test_r['rmse']:.4f} cos={test_r['cos']:.4f}")

    with open("./eval_before_8010_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nGuardado ./eval_before_8010_results.json")


if __name__ == "__main__":
    main()
