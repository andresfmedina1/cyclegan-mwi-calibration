"""
retrain_baseline.py

Reentrena el Baseline (Run 1) con split train/val/test genuinamente
disjunto (80/10/10). Fielmente reproduce Run 1 (Run 1
- Baseline de la ablacion de target-background):
Glorot init, D con activacion final LINEAL, lr_generator=lr_discriminator
=1e-4 (simetrico), gamma_scale=0, SIN enforce_symmetry (esa capa de
reciprocidad se agrego despues de Run 1, junto con el resto del tuning).

Es la config que colapsa (ver Seccion 4.4 de la tesis): sin tabla formal
RMSE/coseno/SSIM en el texto (solo descripcion cualitativa), asi que sus
resultados cuantitativos aca son un aporte nuevo, no una re-comparacion.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python retrain_baseline.py
"""
from retrain_common import run_unpaired_job

BASE_CFG = {
    "synthetic_h5_path": "./synthetic_dataset_6x12.h5",
    "experimental_h5_path": "./experimental_dataset_6x12.h5",
    "seed": 42,

    "gen_dim": 8, "gen_depth": 2, "gen_kernel": 3,
    "disc_dim": 16, "disc_depth": 2, "disc_kernel": 5,

    "gamma_cycle": 100.0, "gamma_identity": 100.0, "gamma_scale": 0.0,

    "lr_generator": 1e-4, "lr_discriminator": 1e-4,

    "epochs": 100, "batch_size": 8, "verbose_every": 5,

    "apply_output_mask": False,
    "enforce_symmetry_a2b": False, "enforce_symmetry_b2a": False,

    # -- lo que hace a este run el "baseline" fiel a Run 1, no HeUniform+tanh --
    "gen_kernel_init": "glorot_uniform",
    "disc_kernel_init": "glorot_uniform",
    "disc_final_activation": "linear",
}

if __name__ == "__main__":
    for scheme in ["8010_3way"]:
        cfg = {**BASE_CFG, "model_dir": f"./checkpoints/Baseline_{scheme}"}
        print(f"\n{'#' * 70}\nBaseline -- scheme={scheme}\n{'#' * 70}")
        run_unpaired_job(cfg, scheme, checkpoint_epochs=[cfg["epochs"]])
    print("\n[retrain_baseline] Listo, esquema 8010_3way entrenado.")
