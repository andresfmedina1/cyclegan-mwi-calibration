"""
eval_traditional_calibration_v2.py

Modulo auxiliar (sin script principal) con la calibracion TRADICIONAL per-muestra
y las metricas compartidas, importado por eval_traditional_calibration_8010.py:

  - Target-background: c_ij = E_sim_healthy_ij / S_meas_healthy_previa_ij, con la
    medicion 'healthy' REAL inmediatamente anterior a CADA adquisicion
    (load_targets_and_healthy, safe_ratio).
  - Target-target: c_ij = E_sim_T_ij / S_meas_T_ij, donde T es el SEGUNDO target
    del par (el que se resta, "B" en "A menos B"). Tanto el campo simulado como
    el medido son los de ESE target especifico:
      E_sim_T   = Esc_B (de simulation_dataset_V2.h5, matcheado por
                  tamano/liquido/posicion/rotacion) + E_sim_healthy
      S_meas_T  = medicion RAW del target B (measurements_database_robot_V2.h5),
                  matcheada con la misma clave (tamano, grupo de liquido,
                  posicion, rotacion) que usa 01b_preprocesamiento_target_target
                  para armar los pares (eval_traditional_target_target).

Las rutas ROBOT_H5_PATH y SIM_V2_H5_PATH estan escritas para la maquina original:
apuntarlas a data/raw/.
"""
import re

import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim

from data_io import (AB_MASK, _sorted_group_keys, _to_str, compress_ab_pairs,
                      expand_ab_pairs, load_experimental_raw_targets,
                      load_synthetic_domain, parse_eps_group, parse_target_size_id,
                      tensor_to_complex)

ROBOT_H5_PATH = ("/Users/felipemedina/Desktop/POLITO/Micriwave Brain/"
                  "Robot_Acquisition/measurements_database_robot_V2.h5")
SIM_V2_H5_PATH = ("/Users/felipemedina/Desktop/POLITO/Micriwave Brain/"
                   "OSMI---Open-Source-Microwave-Imaging-main/Examples/"
                   "Head_imaging/results/simulation_dataset_V2.h5")
FREQ_HZ = 1.3e9
EXCLUDE_ACQ_NUMBERS = set(range(1125, 1143))
THR_DENOM = 1e-6


# =============================================================================
# Metricas (identicas al resto de eval_*.py)
# =============================================================================
def relative_rmse_batch(a, b, mask=AB_MASK):
    a12 = np.stack([expand_ab_pairs(x) for x in a])
    b12 = np.stack([expand_ab_pairs(x) for x in b])
    diff = a12[:, mask] - b12[:, mask]
    num = np.sqrt(np.mean(np.abs(diff) ** 2, axis=1))
    den = np.sqrt(np.mean(np.abs(b12[:, mask]) ** 2, axis=1))
    return num / den


def cosine_sim_batch(a, b, mask=AB_MASK):
    a12 = np.stack([expand_ab_pairs(x) for x in a])
    b12 = np.stack([expand_ab_pairs(x) for x in b])
    av = np.concatenate([np.real(a12[:, mask]), np.imag(a12[:, mask])], axis=1)
    bv = np.concatenate([np.real(b12[:, mask]), np.imag(b12[:, mask])], axis=1)
    num = np.sum(av * bv, axis=1)
    den = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1) + 1e-12
    return num / den


def ssim_batch(a, b, ref, win_size=5):
    n = a.shape[0]
    ssim_re = np.empty(n)
    ssim_im = np.empty(n)
    for i in range(n):
        ref_re, ref_im = np.real(ref[i]), np.imag(ref[i])
        dr_re = ref_re.max() - ref_re.min()
        dr_im = ref_im.max() - ref_im.min()
        dr_re = dr_re if dr_re > 0 else 1e-12
        dr_im = dr_im if dr_im > 0 else 1e-12
        ssim_re[i] = ssim(np.real(a[i]), np.real(b[i]), data_range=dr_re, win_size=win_size)
        ssim_im[i] = ssim(np.imag(a[i]), np.imag(b[i]), data_range=dr_im, win_size=win_size)
    return ssim_re, ssim_im


def stats_from_mask(rmse_b, rmse_a, cos_b, cos_a, ssr_b, ssr_a, ssi_b, ssi_a, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    return {
        "n": n,
        "rmse_before": float(np.mean(rmse_b[mask])), "rmse_after": float(np.mean(rmse_a[mask])),
        "rmse_pct_improved": 100 * float(np.mean(rmse_a[mask] < rmse_b[mask])),
        "cos_before": float(np.mean(cos_b[mask])), "cos_after": float(np.mean(cos_a[mask])),
        "cos_pct_improved": 100 * float(np.mean(cos_a[mask] > cos_b[mask])),
        "ssim_re_before": float(np.mean(ssr_b[mask])), "ssim_re_after": float(np.mean(ssr_a[mask])),
        "ssim_re_pct_improved": 100 * float(np.mean(ssr_a[mask] > ssr_b[mask])),
        "ssim_im_before": float(np.mean(ssi_b[mask])), "ssim_im_after": float(np.mean(ssi_a[mask])),
        "ssim_im_pct_improved": 100 * float(np.mean(ssi_a[mask] > ssi_b[mask])),
    }


def stats_from_idx(rmse_b, rmse_a, cos_b, cos_a, ssr_b, ssr_a, ssi_b, ssi_a):
    n = len(rmse_b)
    return {
        "n": n,
        "rmse_before": float(np.mean(rmse_b)), "rmse_after": float(np.mean(rmse_a)),
        "rmse_pct_improved": 100 * float(np.mean(rmse_a < rmse_b)),
        "cos_before": float(np.mean(cos_b)), "cos_after": float(np.mean(cos_a)),
        "cos_pct_improved": 100 * float(np.mean(cos_a > cos_b)),
        "ssim_re_before": float(np.mean(ssr_b)), "ssim_re_after": float(np.mean(ssr_a)),
        "ssim_re_pct_improved": 100 * float(np.mean(ssr_a > ssr_b)),
        "ssim_im_before": float(np.mean(ssi_b)), "ssim_im_after": float(np.mean(ssi_a)),
        "ssim_im_pct_improved": 100 * float(np.mean(ssi_a > ssi_b)),
    }


def safe_ratio(num_12x12, den_12x12, mask=AB_MASK, thr=THR_DENOM):
    """c_ij = num/den solo dentro de mask y donde |den|>thr, 0 fuera (igual
    convencion que plot_calibration_matrices_local en el .m)."""
    valid = mask & (np.abs(den_12x12) > thr)
    c = np.zeros((12, 12), dtype=complex)
    c[valid] = num_12x12[valid] / den_12x12[valid]
    return c


# =============================================================================
# 1) TARGET-BACKGROUND -- c_ij = E_sim_healthy / S_meas_healthy_PREVIA (real,
#    por muestra)
# =============================================================================
def load_targets_and_healthy(robot_h5_path, freq_hz, exclude_acq_numbers):
    dS_list, bg_list = [], []
    with h5py.File(robot_h5_path, "r") as f:
        keys = _sorted_group_keys(f)
        freqs = f[keys[0]]["freq"][()].flatten()
        f_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        last_healthy_S = None
        for key in keys:
            acq_num = int(re.search(r"(\d+)$", key).group(1))
            if acq_num in exclude_acq_numbers:
                continue
            g = f[key]
            meas_type = _to_str(g.attrs.get("meas_type", "NA"))
            S = g["Sm_real"][f_idx] + 1j * g["Sm_imag"][f_idx]
            if meas_type == "healthy":
                last_healthy_S = S
                continue
            if meas_type != "target":
                continue
            if last_healthy_S is None:
                continue
            dS = (S - last_healthy_S) * AB_MASK
            dS_list.append(dS)
            bg_list.append(last_healthy_S)
    return dS_list, bg_list




# =============================================================================
# 2) TARGET-TARGET -- c_ij = E_sim_T / S_meas_T, T = target B del par
# =============================================================================
def build_match_index(meta_list, key_fn):
    idx = {}
    for i, m in enumerate(meta_list):
        key = key_fn(m)
        if key is not None and key not in idx:
            idx[key] = i
    return idx


def round_key(size_id, eps_group, px, py, rot, pos_tol=1.0, rot_tol=1.0):
    if size_id is None or eps_group is None:
        return None
    return (size_id, eps_group,
            round(px / pos_tol) * pos_tol, round(py / pos_tol) * pos_tol,
            round(rot / rot_tol) * rot_tol)


def eval_traditional_target_target(E_sim_healthy, A_tensor, B_tensor, B_meta, subset_idx=None):
    print("  Cargando simulation_dataset_V2.h5 (dominio A, para Esc_B)...")
    A_list_synth, A_meta_synth = load_synthetic_domain(SIM_V2_H5_PATH, verbose=False)
    A_index = build_match_index(
        A_meta_synth,
        lambda m: round_key(parse_target_size_id(m.get("target_type")),
                             parse_eps_group(m.get("target_eps_label")),
                             m.get("pos_x_mm", np.nan), m.get("pos_y_mm", np.nan),
                             m.get("rotation_deg", np.nan)))

    print("  Cargando measurements_database_robot_V2.h5 (dominio B crudo, para S_meas_B)...")
    S_list_raw, S_meta_raw = load_experimental_raw_targets(
        ROBOT_H5_PATH, freq_hz=FREQ_HZ, exclude_acq_numbers=EXCLUDE_ACQ_NUMBERS, verbose=False)
    B_index = build_match_index(
        S_meta_raw,
        lambda m: round_key(parse_target_size_id(m.get("target_type")),
                             parse_eps_group(m.get("target_eps_label")),
                             m.get("pos_x_mm", np.nan), m.get("pos_y_mm", np.nan),
                             m.get("rotation_deg", np.nan)))

    A_complex = tensor_to_complex(A_tensor)
    B_raw_complex = tensor_to_complex(B_tensor)

    idx_range = range(len(B_meta)) if subset_idx is None else subset_idx
    trad_complex = np.zeros_like(B_raw_complex)
    n_missing = 0
    for i in idx_range:
        m = B_meta[i]
        key = round_key(parse_target_size_id(m["target_type_B"]),
                         parse_eps_group(m["target_eps_label"]),
                         m["pos_x_B_mm"], m["pos_y_B_mm"], m["rotation_B_deg"])
        iA = A_index.get(key)
        iB = B_index.get(key)
        if iA is None or iB is None:
            n_missing += 1
            continue
        Esc_B = A_list_synth[iA]
        E_sim_T = Esc_B + E_sim_healthy
        S_meas_T = S_list_raw[iB]
        c_12x12 = safe_ratio(E_sim_T, S_meas_T)
        c_6x12 = compress_ab_pairs(c_12x12)
        trad_complex[i] = c_6x12 * B_raw_complex[i]

    if n_missing:
        print(f"  [AVISO] {n_missing} muestras sin match target-B (quedan en cero, "
              f"se excluyen de las metricas)")

    return trad_complex, n_missing
