"""
eval_traditional_calibration_8010.py

Calcula la calibracion tradicional (Raw / Traditional / Cycle-GAN, con %
de muestras que mejoran) para los checkpoints del split 80/10/10, usando
la calibracion tradicional per-muestra de eval_traditional_calibration_v2.py
(mismo c_ij), evaluada sobre los indices train/test guardados en
split_indices.npz de cada checkpoint 80/10/10.

Genera las tablas agregadas de deteccion (Run8_8010_3way), monitoreo no
supervisado (TTUnsup_8010_3way) y monitoreo supervisado con holdout Isc2
(TTSup_8010_3way).

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python eval_traditional_calibration_8010.py
"""
import json

import numpy as np
import scipy.io as sio

from cyclegan_model import CycleGan
from data_io import (Normalizer, find_analog_index, load_compressed_datasets,
                      parse_eps_group, tensor_to_complex)
from eval_traditional_calibration_v2 import (
    ROBOT_H5_PATH, cosine_sim_batch, compress_ab_pairs,
    eval_traditional_target_target, load_targets_and_healthy,
    relative_rmse_batch, safe_ratio, ssim_batch, stats_from_idx,
    stats_from_mask, EXCLUDE_ACQ_NUMBERS, FREQ_HZ,
)


def _cyclegan_calibrate(model_dir, cfg, B_tensor_subset):
    """Corre CycleGan.calibrate() sobre un subconjunto (ya en escala fisica,
    SIN normalizar) de B_tensor, devuelve el resultado en escala fisica
    (complejo), usando el normalizador GUARDADO en config.json (fit sobre
    train-only, igual que el resto del pipeline)."""
    norm_A = Normalizer.from_dict(cfg["norm_A"])
    norm_B = Normalizer.from_dict(cfg["norm_B"])
    img_shape = B_tensor_subset.shape[1:]
    cg = CycleGan(img_shape, model_dir, gen_depth=cfg["gen_depth"], disc_depth=cfg["disc_depth"],
                  apply_output_mask=cfg["apply_output_mask"])
    cg.load()
    B_norm = norm_B.forward(B_tensor_subset)
    calibrated_norm = cg.calibrate(B_norm)
    return tensor_to_complex(norm_A.inverse(calibrated_norm))

E_SIM_HEALTHY = sio.loadmat(
    "/Users/felipemedina/Desktop/POLITO/Micriwave Brain/Robot_Acquisition/Ep_healthy.mat"
)["Ep_bg_vis"]


# =============================================================================
# 1) Target-background (deteccion) -- split de Run8_8010_3way
# =============================================================================
def eval_tb_8010():
    model_dir = "./checkpoints/Run8_8010_3way"
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)

    (A_tb, B_tb, A_meta, B_meta, _, _, _, _) = load_compressed_datasets(
        "./synthetic_dataset_6x12.h5", "./experimental_dataset_6x12.h5")

    dS_list, bg_list = load_targets_and_healthy(ROBOT_H5_PATH, FREQ_HZ, EXCLUDE_ACQ_NUMBERS)
    assert len(dS_list) == len(B_tb)

    A_complex = tensor_to_complex(A_tb)
    B_raw_complex = tensor_to_complex(B_tb)

    split = dict(np.load(f"{model_dir}/split_indices.npz"))
    train_set = set(split["B_train"].tolist())
    test_set = set(split["B_test"].tolist())

    pairs = []
    for b_idx in range(len(B_meta)):
        a_idx = find_analog_index(B_meta[b_idx], A_meta, pos_tol_mm=1.0, match_position=True)
        if a_idx is not None:
            pairs.append((b_idx, a_idx))
    b_indices = np.array([p[0] for p in pairs])
    a_indices = np.array([p[1] for p in pairs])
    is_train = np.array([b in train_set for b in b_indices])
    is_test = np.array([b in test_set for b in b_indices])

    trad_list = []
    for b_idx in b_indices:
        Sm_BG_prev = bg_list[b_idx]
        c_12x12 = safe_ratio(E_SIM_HEALTHY, Sm_BG_prev)
        c_6x12 = compress_ab_pairs(c_12x12)
        trad_list.append(c_6x12 * B_raw_complex[b_idx])
    trad_complex = np.stack(trad_list)

    cg_complex = _cyclegan_calibrate(model_dir, cfg, B_tb[b_indices])

    B_matched = B_raw_complex[b_indices]
    A_matched = A_complex[a_indices]

    rmse_b = relative_rmse_batch(B_matched, A_matched)
    rmse_trad = relative_rmse_batch(trad_complex, A_matched)
    rmse_cg = relative_rmse_batch(cg_complex, A_matched)
    cos_b = cosine_sim_batch(B_matched, A_matched)
    cos_trad = cosine_sim_batch(trad_complex, A_matched)
    cos_cg = cosine_sim_batch(cg_complex, A_matched)
    ssr_b, ssi_b = ssim_batch(A_matched, B_matched, ref=A_matched)
    ssr_trad, ssi_trad = ssim_batch(A_matched, trad_complex, ref=A_matched)
    ssr_cg, ssi_cg = ssim_batch(A_matched, cg_complex, ref=A_matched)

    def pack(mask):
        return {
            "traditional": stats_from_mask(rmse_b, rmse_trad, cos_b, cos_trad,
                                            ssr_b, ssr_trad, ssi_b, ssi_trad, mask),
            "cyclegan": stats_from_mask(rmse_b, rmse_cg, cos_b, cos_cg,
                                         ssr_b, ssr_cg, ssi_b, ssi_cg, mask),
        }

    return {"train": pack(is_train), "test": pack(is_test)}


# =============================================================================
# 2) Target-target: calibracion tradicional sobre TODO el dataset (no
#    depende del split), se reusa para las tablas unsup y sup
# =============================================================================
def compute_tt_traditional_full():
    (A_tt, B_tt, A_meta, B_meta, _, _, _, _) = load_compressed_datasets(
        "./synthetic_dataset_6x12_targettarget.h5", "./experimental_dataset_6x12_targettarget.h5")
    trad_complex_all, n_missing = eval_traditional_target_target(E_SIM_HEALTHY, A_tt, B_tt, B_meta)
    A_complex = tensor_to_complex(A_tt)
    B_raw_complex = tensor_to_complex(B_tt)
    valid_mask_all = np.array([np.any(trad_complex_all[i] != 0) for i in range(len(B_meta))])
    return A_complex, B_raw_complex, B_tt, B_meta, trad_complex_all, valid_mask_all, n_missing


def eval_tt_unsup_8010(A_complex, B_raw_complex, B_tensor, trad_complex_all, valid_mask_all):
    model_dir = "./checkpoints/TTUnsup_8010_3way"
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    split = dict(np.load(f"{model_dir}/split_indices.npz"))
    n = A_complex.shape[0]
    is_train = np.zeros(n, dtype=bool); is_train[split["train"]] = True
    is_test = np.zeros(n, dtype=bool); is_test[split["test"]] = True

    cg_complex_full = np.zeros_like(B_raw_complex)
    idx_calib = np.where(is_train | is_test)[0]
    cg_complex_full[idx_calib] = _cyclegan_calibrate(model_dir, cfg, B_tensor[idx_calib])

    rmse_b = relative_rmse_batch(B_raw_complex, A_complex)
    rmse_trad = relative_rmse_batch(trad_complex_all, A_complex)
    rmse_cg = relative_rmse_batch(cg_complex_full, A_complex)
    cos_b = cosine_sim_batch(B_raw_complex, A_complex)
    cos_trad = cosine_sim_batch(trad_complex_all, A_complex)
    cos_cg = cosine_sim_batch(cg_complex_full, A_complex)
    ssr_b, ssi_b = ssim_batch(A_complex, B_raw_complex, ref=A_complex)
    ssr_trad, ssi_trad = ssim_batch(A_complex, trad_complex_all, ref=A_complex)
    ssr_cg, ssi_cg = ssim_batch(A_complex, cg_complex_full, ref=A_complex)

    def pack(mask):
        m = mask & valid_mask_all
        return {
            "traditional": stats_from_mask(rmse_b, rmse_trad, cos_b, cos_trad,
                                            ssr_b, ssr_trad, ssi_b, ssi_trad, m),
            "cyclegan": stats_from_mask(rmse_b, rmse_cg, cos_b, cos_cg,
                                         ssr_b, ssr_cg, ssi_b, ssi_cg, m),
        }

    return {"train": pack(is_train), "test": pack(is_test)}


def eval_tt_sup_8010(A_complex, B_raw_complex, B_tensor, trad_complex_all, valid_mask_all, B_meta):
    model_dir = "./checkpoints/TTSup_8010_3way"
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    split = dict(np.load(f"{model_dir}/split_indices.npz"))
    train_idx = split["train"]  # ya excluye Isc2 por construccion
    is_isc2 = np.array([parse_eps_group(m["target_eps_label"]) == "Isc2" for m in B_meta])
    holdout_idx = np.where(is_isc2)[0]

    n = A_complex.shape[0]
    cg_complex_full = np.zeros_like(B_raw_complex)
    idx_calib = np.union1d(train_idx, holdout_idx)
    cg_complex_full[idx_calib] = _cyclegan_calibrate(model_dir, cfg, B_tensor[idx_calib])

    def eval_subset(idx):
        idx = np.asarray(idx)
        valid = valid_mask_all[idx]
        idx_v = idx[valid]
        rmse_b_s = relative_rmse_batch(B_raw_complex[idx_v], A_complex[idx_v])
        rmse_trad_s = relative_rmse_batch(trad_complex_all[idx_v], A_complex[idx_v])
        rmse_cg_s = relative_rmse_batch(cg_complex_full[idx_v], A_complex[idx_v])
        cos_b_s = cosine_sim_batch(B_raw_complex[idx_v], A_complex[idx_v])
        cos_trad_s = cosine_sim_batch(trad_complex_all[idx_v], A_complex[idx_v])
        cos_cg_s = cosine_sim_batch(cg_complex_full[idx_v], A_complex[idx_v])
        ssr_b_s, ssi_b_s = ssim_batch(A_complex[idx_v], B_raw_complex[idx_v], ref=A_complex[idx_v])
        ssr_trad_s, ssi_trad_s = ssim_batch(A_complex[idx_v], trad_complex_all[idx_v], ref=A_complex[idx_v])
        ssr_cg_s, ssi_cg_s = ssim_batch(A_complex[idx_v], cg_complex_full[idx_v], ref=A_complex[idx_v])
        return {
            "traditional": stats_from_idx(rmse_b_s, rmse_trad_s, cos_b_s, cos_trad_s,
                                           ssr_b_s, ssr_trad_s, ssi_b_s, ssi_trad_s),
            "cyclegan": stats_from_idx(rmse_b_s, rmse_cg_s, cos_b_s, cos_cg_s,
                                        ssr_b_s, ssr_cg_s, ssi_b_s, ssi_cg_s),
        }

    return {"train": eval_subset(train_idx), "test": eval_subset(holdout_idx)}


def main():
    results = {}

    print("=" * 70)
    print("Target-background (deteccion) -- split Run8_8010_3way")
    print("=" * 70)
    results["Detection_8010"] = eval_tb_8010()
    r = results["Detection_8010"]
    print(f"  train n={r['train']['traditional']['n']}  test n={r['test']['traditional']['n']}")

    print("\n" + "=" * 70)
    print("Calibracion tradicional target-target (dataset completo, se reusa)")
    print("=" * 70)
    (A_complex, B_raw_complex, B_tt_tensor, B_meta, trad_complex_all,
     valid_mask_all, n_missing) = compute_tt_traditional_full()
    print(f"  n_missing matches: {n_missing}")

    print("\nTarget-target no supervisado -- split TTUnsup_8010_3way")
    results["MonitoringUnsup_8010"] = eval_tt_unsup_8010(
        A_complex, B_raw_complex, B_tt_tensor, trad_complex_all, valid_mask_all)
    r = results["MonitoringUnsup_8010"]
    print(f"  train n={r['train']['traditional']['n']}  test n={r['test']['traditional']['n']}")

    print("\nTarget-target supervisado / Isc2 -- split TTSup_8010_3way")
    results["MonitoringSup_Isc2_8010"] = eval_tt_sup_8010(
        A_complex, B_raw_complex, B_tt_tensor, trad_complex_all, valid_mask_all, B_meta)
    r = results["MonitoringSup_Isc2_8010"]
    print(f"  train n={r['train']['traditional']['n']}  test(Isc2) n={r['test']['traditional']['n']}")

    with open("./eval_traditional_calibration_8010_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nGuardado ./eval_traditional_calibration_8010_results.json")


if __name__ == "__main__":
    main()
