"""
eval_train_test_full.py

Evalua los checkpoints entrenados con el split 80/10/10 (Baseline, Run8,
TTUnsup, TTSup) sobre su TRAIN y su TEST genuinamente disjuntos, con RMSE
relativo, similitud coseno y SSIM contra la referencia sintetica.

NO reentrena nada -- solo carga pesos .keras + split_indices.npz +
config.json de cada checkpoint ya entrenado. Los checkpoints que no existan
se omiten.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python eval_train_test_full.py
"""
import json
import os

import numpy as np

from cyclegan_model import CycleGan
from data_io import (Normalizer, cosine_sim_batch, find_analog_index,
                      load_compressed_datasets, relative_rmse_batch,
                      ssim_batch, tensor_to_complex)

CHECKPOINTS = {
    "Baseline_8010_3way":   dict(kind="unpaired"),
    "Run8_8010_3way":       dict(kind="unpaired"),
    "TTUnsup_8010_3way":    dict(kind="paired"),
    "TTSup_8010_3way":      dict(kind="paired"),
}


def load_split(model_dir, kind):
    d = dict(np.load(os.path.join(model_dir, "split_indices.npz")))
    if kind == "unpaired":
        return {"A": {"test": d["A_test"], "train": d["A_train"]},
                "B": {"test": d["B_test"], "train": d["B_train"]}}
    return {"test": d["test"], "train": d["train"]}


def _eval_unpaired(model_dir, cfg, cg, norm_A, norm_B, A_tensor, B_tensor,
                    A_meta, B_meta, split_name, b_idx_list):
    pairs = []
    for b_idx in b_idx_list:
        a_idx = find_analog_index(B_meta[b_idx], A_meta, pos_tol_mm=1.0, match_position=True)
        if a_idx is not None:
            pairs.append((b_idx, a_idx))
    if not pairs:
        raise RuntimeError(f"{model_dir}: no se encontraron pares analogos en {split_name}")
    b_idx_arr = np.array([p[0] for p in pairs])
    a_idx_arr = np.array([p[1] for p in pairs])

    B_norm_all = norm_B.forward(B_tensor)
    calibrated_norm = cg.calibrate(B_norm_all[b_idx_arr])
    calibrated_complex = tensor_to_complex(norm_A.inverse(calibrated_norm))

    A_complex = tensor_to_complex(A_tensor)
    A_matched = A_complex[a_idx_arr]

    rmse = relative_rmse_batch(calibrated_complex, A_matched)
    cos = cosine_sim_batch(calibrated_complex, A_matched)
    ssim_re, ssim_im = ssim_batch(A_matched, calibrated_complex, ref=A_matched)

    return dict(n=len(pairs), rmse=float(np.mean(rmse)), cos=float(np.mean(cos)),
                ssim_re=float(np.mean(ssim_re)), ssim_im=float(np.mean(ssim_im)))


def _eval_paired(cfg, cg, norm_A, norm_B, A_tensor, B_tensor, idx):
    B_norm = norm_B.forward(B_tensor[idx])
    calibrated_norm = cg.calibrate(B_norm)
    calibrated_complex = tensor_to_complex(norm_A.inverse(calibrated_norm))
    A_complex = tensor_to_complex(A_tensor[idx])

    rmse = relative_rmse_batch(calibrated_complex, A_complex)
    cos = cosine_sim_batch(calibrated_complex, A_complex)
    ssim_re, ssim_im = ssim_batch(A_complex, calibrated_complex, ref=A_complex)

    return dict(n=int(len(idx)), rmse=float(np.mean(rmse)), cos=float(np.mean(cos)),
                ssim_re=float(np.mean(ssim_re)), ssim_im=float(np.mean(ssim_im)))


def evaluate_checkpoint(name, info):
    model_dir = f"./checkpoints/{name}"
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    A_tensor, B_tensor, A_meta, B_meta, _a, _b, _fA, _fB = load_compressed_datasets(
        cfg["synthetic_h5_path"], cfg["experimental_h5_path"])
    split = load_split(model_dir, info["kind"])

    norm_A = Normalizer.from_dict(cfg["norm_A"])
    norm_B = Normalizer.from_dict(cfg["norm_B"])

    img_shape = A_tensor.shape[1:]
    cg = CycleGan(img_shape, model_dir, gen_depth=cfg["gen_depth"],
                  disc_depth=cfg["disc_depth"], apply_output_mask=cfg["apply_output_mask"])
    cg.load()

    if info["kind"] == "unpaired":
        print(f"  [{name}] train (n_B={len(split['B']['train'])}) ...")
        train_r = _eval_unpaired(model_dir, cfg, cg, norm_A, norm_B, A_tensor, B_tensor,
                                  A_meta, B_meta, "train", split["B"]["train"])
        print(f"  [{name}] test (n_B={len(split['B']['test'])}) ...")
        test_r = _eval_unpaired(model_dir, cfg, cg, norm_A, norm_B, A_tensor, B_tensor,
                                 A_meta, B_meta, "test", split["B"]["test"])
    else:
        print(f"  [{name}] train (n={len(split['train'])}) ...")
        train_r = _eval_paired(cfg, cg, norm_A, norm_B, A_tensor, B_tensor, split["train"])
        print(f"  [{name}] test (n={len(split['test'])}) ...")
        test_r = _eval_paired(cfg, cg, norm_A, norm_B, A_tensor, B_tensor, split["test"])

    return {"train": train_r, "test": test_r}


def main():
    results = {}
    for name, info in CHECKPOINTS.items():
        model_dir = f"./checkpoints/{name}"
        if not os.path.isdir(model_dir) or not os.path.isfile(os.path.join(model_dir, "config.json")):
            print(f"[skip] {model_dir} no existe todavia")
            continue
        print(f"\nEvaluando {name} (train + test)...")
        results[name] = evaluate_checkpoint(name, info)

    with open("./eval_train_test_full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nGuardado ./eval_train_test_full_results.json")


if __name__ == "__main__":
    main()
