"""
retrain_tt_unsupervised.py

Reentrena target-target NO supervisado con split train/val/test
genuinamente disjunto (80/10/10). Split preserva el pareo A[i]/B[i]
en las 3 particiones (splits.split_paired), aunque el ENTRENAMIENTO siga
siendo unpaired por batch (paired_training=False, gamma_supervised=0.0).
Se entrena solo hasta la epoca 20 (el checkpoint promovido): continuar mas alla
lo empeora por colapso tardio de D (ver extend_ttunsup_8010.py).

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python retrain_tt_unsupervised.py
"""
from retrain_common import run_paired_job

BASE_CFG = {
    "synthetic_h5_path": "./synthetic_dataset_6x12_targettarget.h5",
    "experimental_h5_path": "./experimental_dataset_6x12_targettarget.h5",
    "seed": 42,

    "gen_dim": 8, "gen_depth": 2, "gen_kernel": 3,
    "disc_dim": 16, "disc_depth": 2, "disc_kernel": 5,

    "gamma_cycle": 20.0, "gamma_identity": 20.0, "gamma_scale": 3.0,
    "gamma_supervised": 0.0,

    "lr_generator": 1e-4, "lr_discriminator": 1e-5,

    "epochs": 20, "batch_size": 32, "verbose_every": 5,

    "apply_output_mask": False,
    "enforce_symmetry_a2b": False, "enforce_symmetry_b2a": True,
}

if __name__ == "__main__":
    for scheme in ["8010_3way"]:
        cfg = {**BASE_CFG, "model_dir": f"./checkpoints/TTUnsup_{scheme}"}
        print(f"\n{'#' * 70}\nTT-unsupervised -- scheme={scheme}\n{'#' * 70}")
        run_paired_job(cfg, scheme, checkpoint_epochs=[cfg["epochs"]],
                        paired_training=False, pool_idx=None)
    print("\n[retrain_tt_unsupervised] Listo, esquema 8010_3way entrenado.")
