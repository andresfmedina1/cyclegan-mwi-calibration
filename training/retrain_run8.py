"""
retrain_run8.py

Entrena Run8, la configuracion final de deteccion (target-background), con
split train/val/test 80/10/10 genuinamente disjunto: gamma_cycle=gamma_identity=20,
gamma_scale=3, lr_discriminator=2e-5 (asimetrico), enforce_symmetry_b2a=True.
La normalizacion se ajusta solo sobre train (ver retrain_common.py).

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python retrain_run8.py
"""
from retrain_common import run_unpaired_job

BASE_CFG = {
    "synthetic_h5_path": "./synthetic_dataset_6x12.h5",
    "experimental_h5_path": "./experimental_dataset_6x12.h5",
    "seed": 42,

    "gen_dim": 8, "gen_depth": 2, "gen_kernel": 3,
    "disc_dim": 16, "disc_depth": 2, "disc_kernel": 5,

    "gamma_cycle": 20.0, "gamma_identity": 20.0, "gamma_scale": 3.0,

    "lr_generator": 1e-4, "lr_discriminator": 2e-5,

    "epochs": 100, "batch_size": 8, "verbose_every": 5,

    "apply_output_mask": False,
    "enforce_symmetry_a2b": False, "enforce_symmetry_b2a": True,
    # gen_kernel_init/disc_kernel_init/disc_final_activation: no se pasan ->
    # defaults he_uniform/he_uniform/tanh, igual que BestSynth_V2/Run8.
}

if __name__ == "__main__":
    for scheme in ["8010_3way"]:
        cfg = {**BASE_CFG, "model_dir": f"./checkpoints/Run8_{scheme}"}
        print(f"\n{'#' * 70}\nRun8 -- scheme={scheme}\n{'#' * 70}")
        run_unpaired_job(cfg, scheme, checkpoint_epochs=[cfg["epochs"]])
    print("\n[retrain_run8] Listo, esquema 8010_3way entrenado.")
