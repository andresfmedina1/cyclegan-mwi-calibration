"""
retrain_tt_supervised.py

Entrena target-target SUPERVISADO, 200 epocas, en chunks (crash-safe, con
checkpoints intermedios cada 25 epocas), con split train/val/test 80/10/10
genuinamente disjunto.

DECISION DE DISENO -- Isc2: el grupo Isc2 (alcohol puro, 10274 muestras) se
reserva como holdout de categoria completa para medir generalizacion a un
material nunca visto (ver evaluation/eval_tt_supervised_isc2.py). Por eso se
excluye del pool ANTES de particionar: el pool disponible es
41096 - 10274 = 30822 muestras, y de ahi salen train/val/test aleatorios. Isc2
tampoco entra en las estadisticas de normalizacion.

Uso (entorno conda 'cyclegan', larga duracion -- correr en background):
    conda run -n cyclegan python retrain_tt_supervised.py
"""
import numpy as np

from data_io import load_compressed_dataset_h5, parse_eps_group
from retrain_common import run_paired_job

BASE_CFG = {
    "synthetic_h5_path": "./synthetic_dataset_6x12_targettarget.h5",
    "experimental_h5_path": "./experimental_dataset_6x12_targettarget.h5",
    "seed": 42,

    "gen_dim": 8, "gen_depth": 2, "gen_kernel": 3,
    "disc_dim": 16, "disc_depth": 2, "disc_kernel": 5,

    "gamma_cycle": 20.0, "gamma_identity": 20.0, "gamma_scale": 3.0,
    "gamma_supervised": 20.0,

    "lr_generator": 1e-4, "lr_discriminator": 1e-5,

    "epochs": 200, "batch_size": 32, "verbose_every": 5,

    "apply_output_mask": False,
    "enforce_symmetry_a2b": False, "enforce_symmetry_b2a": True,
}
CHECKPOINT_EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200]


def _isc2_exclusion_pool_idx():
    # Metadatos vienen del .h5 EXPERIMENTAL (B); el pareo por indice con A
    # garantiza que excluir por indice de B == excluir la misma fila de A.
    _mat, meta, _freq = load_compressed_dataset_h5(BASE_CFG["experimental_h5_path"])
    is_isc2 = np.array([parse_eps_group(m["target_eps_label"]) == "Isc2" for m in meta])
    return np.where(~is_isc2)[0]  # indices NO-Isc2, absolutos en 0..40999


if __name__ == "__main__":
    pool_idx = _isc2_exclusion_pool_idx()
    print(f"[retrain_tt_supervised] Pool disponible (sin Isc2): {len(pool_idx)} / 41096")
    for scheme in ["8010_3way"]:
        cfg = {**BASE_CFG, "model_dir": f"./checkpoints/TTSup_{scheme}"}
        print(f"\n{'#' * 70}\nTT-supervised -- scheme={scheme}\n{'#' * 70}")
        run_paired_job(cfg, scheme, checkpoint_epochs=CHECKPOINT_EPOCHS,
                        paired_training=True, pool_idx=pool_idx)
    print("\n[retrain_tt_supervised] Listo, esquema 8010_3way entrenado.")
