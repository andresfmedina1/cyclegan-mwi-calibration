"""
eval_tt_sup_indist_test_8010.py

Calcula Raw / Traditional / Cycle-GAN (con % de muestras que mejoran) para
el TEST "in-distribution" del 10% de TTSup_8010_3way (split["test"],
n=3082, aleatorio dentro del pool sin Isc2) -- distinto del holdout de
Isc2 ya calculado en eval_traditional_calibration_8010.py. Reusa toda la
misma logica/formula de calibracion tradicional.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python eval_tt_sup_indist_test_8010.py
"""
import json

import numpy as np
import scipy.io as sio

from data_io import load_compressed_datasets, tensor_to_complex
from eval_traditional_calibration_8010 import (E_SIM_HEALTHY,
                                                _cyclegan_calibrate,
                                                compute_tt_traditional_full)
from eval_traditional_calibration_v2 import (cosine_sim_batch,
                                              relative_rmse_batch,
                                              ssim_batch, stats_from_idx)


def main():
    (A_complex, B_raw_complex, B_tt_tensor, B_meta, trad_complex_all,
     valid_mask_all, n_missing) = compute_tt_traditional_full()
    print(f"n_missing matches: {n_missing}")

    model_dir = "./checkpoints/TTSup_8010_3way"
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    split = dict(np.load(f"{model_dir}/split_indices.npz"))
    test_idx = split["test"]
    print(f"test (in-distribution, 10%) n={len(test_idx)}")

    cg_complex_test = _cyclegan_calibrate(model_dir, cfg, B_tt_tensor[test_idx])

    idx = np.asarray(test_idx)
    valid = valid_mask_all[idx]
    idx_v = idx[valid]
    print(f"valid (traditional match found) n={len(idx_v)}")

    rmse_b = relative_rmse_batch(B_raw_complex[idx_v], A_complex[idx_v])
    rmse_trad = relative_rmse_batch(trad_complex_all[idx_v], A_complex[idx_v])
    cos_b = cosine_sim_batch(B_raw_complex[idx_v], A_complex[idx_v])
    cos_trad = cosine_sim_batch(trad_complex_all[idx_v], A_complex[idx_v])
    ssr_b, ssi_b = ssim_batch(A_complex[idx_v], B_raw_complex[idx_v], ref=A_complex[idx_v])
    ssr_trad, ssi_trad = ssim_batch(A_complex[idx_v], trad_complex_all[idx_v], ref=A_complex[idx_v])

    # cyclegan fue calculado SOLO sobre test_idx (no sobre todo el dataset) --
    # hay que re-indexar con la mascara `valid` (posicion relativa dentro de test_idx)
    cg_complex_v = cg_complex_test[valid]
    rmse_cg = relative_rmse_batch(cg_complex_v, A_complex[idx_v])
    cos_cg = cosine_sim_batch(cg_complex_v, A_complex[idx_v])
    ssr_cg, ssi_cg = ssim_batch(A_complex[idx_v], cg_complex_v, ref=A_complex[idx_v])

    result = {
        "traditional": stats_from_idx(rmse_b, rmse_trad, cos_b, cos_trad,
                                       ssr_b, ssr_trad, ssi_b, ssi_trad),
        "cyclegan": stats_from_idx(rmse_b, rmse_cg, cos_b, cos_cg,
                                    ssr_b, ssr_cg, ssi_b, ssi_cg),
    }

    with open("./eval_tt_sup_indist_test_8010_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nGuardado ./eval_tt_sup_indist_test_8010_results.json")


if __name__ == "__main__":
    main()
