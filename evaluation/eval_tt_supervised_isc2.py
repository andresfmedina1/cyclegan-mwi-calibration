"""
eval_tt_supervised_isc2.py

Evalua el checkpoint de target-target supervisado (TTSup_8010_3way) sobre el
holdout de categoria COMPLETA Isc2 (10274 muestras, alcohol puro): una prueba
de generalizacion a un material nunca visto. Isc2 se excluyo del pool ANTES
del split 80/10/10, por lo que estos pesos nunca vieron Isc2 en
entrenamiento, validacion, test ni en las estadisticas de normalizacion.

NO reentrena nada -- solo carga pesos .keras ya entrenados.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python eval_tt_supervised_isc2.py
"""
import json

import numpy as np

from cyclegan_model import CycleGan
from data_io import (Normalizer, cosine_sim_batch, load_compressed_datasets,
                      parse_eps_group, relative_rmse_batch, ssim_batch,
                      tensor_to_complex)

CHECKPOINTS = ["TTSup_8010_3way"]

def main():
    A_tensor, B_tensor, A_meta, B_meta, _a, _b, _fA, _fB = load_compressed_datasets(
        "./synthetic_dataset_6x12_targettarget.h5",
        "./experimental_dataset_6x12_targettarget.h5")

    is_isc2 = np.array([parse_eps_group(m["target_eps_label"]) == "Isc2" for m in B_meta])
    isc2_idx = np.where(is_isc2)[0]
    print(f"Isc2 holdout: {len(isc2_idx)} muestras (de {len(B_meta)} totales)")

    A_complex_isc2 = tensor_to_complex(A_tensor[isc2_idx])

    results = {}
    for name in CHECKPOINTS:
        model_dir = f"./checkpoints/{name}"
        with open(f"{model_dir}/config.json") as f:
            cfg = json.load(f)

        norm_A = Normalizer.from_dict(cfg["norm_A"])
        norm_B = Normalizer.from_dict(cfg["norm_B"])

        img_shape = A_tensor.shape[1:]
        cg = CycleGan(img_shape, model_dir, gen_depth=cfg["gen_depth"],
                      disc_depth=cfg["disc_depth"], apply_output_mask=cfg["apply_output_mask"])
        cg.load()

        print(f"\nEvaluando {name} sobre Isc2 completo (n={len(isc2_idx)})...")
        B_norm_isc2 = norm_B.forward(B_tensor[isc2_idx])
        calibrated_norm = cg.calibrate(B_norm_isc2)
        calibrated_complex = tensor_to_complex(norm_A.inverse(calibrated_norm))

        rmse = relative_rmse_batch(calibrated_complex, A_complex_isc2)
        cos = cosine_sim_batch(calibrated_complex, A_complex_isc2)
        ssim_re, ssim_im = ssim_batch(A_complex_isc2, calibrated_complex, ref=A_complex_isc2)

        results[name] = dict(n=int(len(isc2_idx)), rmse=float(np.mean(rmse)),
                              cos=float(np.mean(cos)), ssim_re=float(np.mean(ssim_re)),
                              ssim_im=float(np.mean(ssim_im)))
        r = results[name]
        print(f"  RMSE={r['rmse']:.4f}  cos={r['cos']:.4f}  "
              f"SSIM_Re={r['ssim_re']:.4f}  SSIM_Im={r['ssim_im']:.4f}")

    with open("./eval_tt_supervised_isc2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    for name, r in results.items():
        print(f"{name}  Isc2-holdout (nunca visto en ningun split):  "
              f"RMSE={r['rmse']:.4f}  cos={r['cos']:.4f}  "
              f"SSIM_Re={r['ssim_re']:.4f}  SSIM_Im={r['ssim_im']:.4f}")
    print("\nGuardado ./eval_tt_supervised_isc2_results.json")


if __name__ == "__main__":
    main()
