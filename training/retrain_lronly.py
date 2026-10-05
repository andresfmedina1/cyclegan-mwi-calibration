"""
retrain_lronly.py

Reentrena la configuracion intermedia de tuning ("solo LR reducido"):
gamma_cyc=gamma_id=100 (igual al baseline), pero con lr_discriminator=2e-5
(asimetrico) en vez de compartido a 1e-4. Esta es la configuracion usada
en la Figura 4.5a (fig_4_5a_tuning_comparison) como paso intermedio entre
el baseline que colapsa y la configuracion final tuneada (gamma=20/20,
scale=3). Solo se entrena el esquema 8010_3way -- es la unica pasada que
faltaba para poder regenerar esa figura con los resultados del split 80/10/10.

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python retrain_lronly.py
"""
from retrain_common import run_unpaired_job

BASE_CFG = {
    "synthetic_h5_path": "./synthetic_dataset_6x12.h5",
    "experimental_h5_path": "./experimental_dataset_6x12.h5",
    "seed": 42,

    "gen_dim": 8, "gen_depth": 2, "gen_kernel": 3,
    "disc_dim": 16, "disc_depth": 2, "disc_kernel": 5,

    "gamma_cycle": 100.0, "gamma_identity": 100.0, "gamma_scale": 0.0,

    "lr_generator": 1e-4, "lr_discriminator": 2e-5,

    "epochs": 100, "batch_size": 8, "verbose_every": 5,

    "apply_output_mask": False,
    "enforce_symmetry_a2b": False, "enforce_symmetry_b2a": False,

    # mismo init que el baseline -- solo cambia lr_discriminator
    "gen_kernel_init": "glorot_uniform",
    "disc_kernel_init": "glorot_uniform",
    "disc_final_activation": "linear",
}

if __name__ == "__main__":
    scheme = "8010_3way"
    cfg = {**BASE_CFG, "model_dir": f"./checkpoints/LRonly_{scheme}"}
    print(f"\n{'#' * 70}\nLR-only tuning -- scheme={scheme}\n{'#' * 70}")
    run_unpaired_job(cfg, scheme, checkpoint_epochs=[cfg["epochs"]])
    print("\n[retrain_lronly] Listo.")
