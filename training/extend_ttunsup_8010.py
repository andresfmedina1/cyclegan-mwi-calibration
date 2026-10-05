"""
extend_ttunsup_8010.py

Continua el entrenamiento de TTUnsup_8010_3way (target-target no
supervisado, split 80/10/10) mas alla de la epoca 20 hasta la 100, para
poder regenerar la Figura 4.6 (fig_4_6_tt_late_collapse) con el split
80/10/10 -- esa figura necesita ver el colapso tardio de D entre las
epocas 20-100, que el reentrenamiento original de este checkpoint no
cubria (se detuvo en 20 a proposito, por ser el checkpoint promovido).

Antes de sobrescribir, hace un backup de los pesos+history de la epoca 20
en checkpoints/TTUnsup_8010_3way_ep20/.

Usa exactamente el mismo split/normalizacion que la corrida original
(mismo seed=42, scheme=8010_3way, y normalizador leido de config.json en
vez de recalculado, para eliminar cualquier posibilidad de discrepancia
por redondeo).

Uso (entorno conda 'cyclegan'):
    conda run -n cyclegan python extend_ttunsup_8010.py
"""
import json
import os
import shutil

import numpy as np
import tensorflow as tf

from cyclegan_model import CycleGan
from data_io import Normalizer, load_compressed_datasets
from splits import split_paired
from retrain_common import plot_training_history

MODEL_DIR = "./checkpoints/TTUnsup_8010_3way"
BACKUP_DIR = "./checkpoints/TTUnsup_8010_3way_ep20"
TARGET_EPOCHS = 100


def main():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)

    # --- backup de los pesos/history de la epoca 20 ---
    if not os.path.isdir(BACKUP_DIR):
        shutil.copytree(MODEL_DIR, BACKUP_DIR)
        print(f"[extend] Backup de la epoca 20 guardado en {BACKUP_DIR}")
    else:
        print(f"[extend] Backup {BACKUP_DIR} ya existe, no se sobrescribe.")

    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    (A_tensor, B_tensor, A_meta, B_meta, _nA, _nB, freq_A, freq_B) = \
        load_compressed_datasets(cfg["synthetic_h5_path"], cfg["experimental_h5_path"])

    split = split_paired(A_tensor.shape[0], cfg["scheme"], seed=cfg["seed"], pool_idx=None)

    norm_A = Normalizer.from_dict(cfg["norm_A"])
    norm_B = Normalizer.from_dict(cfg["norm_B"])
    A_norm, B_norm = norm_A.forward(A_tensor), norm_B.forward(B_tensor)

    train_A, val_A = A_norm[split["train"]], A_norm[split["val"]]
    train_B, val_B = B_norm[split["train"]], B_norm[split["val"]]
    print(f"[extend] train/val = {len(train_A)}/{len(val_A)} (debe matchear config.json: "
          f"{cfg['n_train']}/{cfg['n_val']})")

    img_shape = A_tensor.shape[1:]
    cycle_gan = CycleGan(
        img_shape, MODEL_DIR,
        gen_dim=cfg["gen_dim"], gen_depth=cfg["gen_depth"], gen_kernel=cfg["gen_kernel"],
        disc_dim=cfg["disc_dim"], disc_depth=cfg["disc_depth"], disc_kernel=cfg["disc_kernel"],
        gamma_cycle=cfg["gamma_cycle"], gamma_identity=cfg["gamma_identity"],
        gamma_scale=cfg["gamma_scale"], gamma_supervised=cfg.get("gamma_supervised", 0.0),
        lr_generator=cfg["lr_generator"], lr_discriminator=cfg["lr_discriminator"],
        apply_output_mask=cfg["apply_output_mask"], ab_mask=None,
        enforce_symmetry_a2b=cfg["enforce_symmetry_a2b"],
        enforce_symmetry_b2a=cfg["enforce_symmetry_b2a"],
    )
    cycle_gan.load()
    print("[extend] Pesos de la epoca 20 cargados -- continuando hasta epoca "
          f"{TARGET_EPOCHS}.")

    # cycle_gan.load() NO restaura self.history (solo pesos) -- se
    # precarga manualmente con el history.json de la epoca 20 para que el
    # entrenamiento que sigue LO EXTIENDA en vez de empezar de cero.
    with open(os.path.join(MODEL_DIR, "history.json")) as f:
        prev_history = json.load(f)
    cycle_gan.history = {k: list(v) for k, v in prev_history.items()}
    assert len(cycle_gan.history["d_loss"]) == 20, \
        f"history.json previo no tiene 20 epocas (tiene {len(cycle_gan.history['d_loss'])})"

    n_chunk = TARGET_EPOCHS - 20
    cycle_gan.train(train_A, train_B, epochs=n_chunk, batch_size=cfg["batch_size"],
                     verbose_every=cfg["verbose_every"], test_A=val_A, test_B=val_B,
                     paired=cfg.get("paired_training", False))

    assert len(cycle_gan.history["d_loss"]) == TARGET_EPOCHS, \
        f"history final tiene {len(cycle_gan.history['d_loss'])} epocas, se esperaban {TARGET_EPOCHS}"

    cycle_gan.model_dir = MODEL_DIR
    cycle_gan.save()
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(cycle_gan.history, f)
    plot_training_history(cycle_gan.history, MODEL_DIR)

    cfg["epochs"] = TARGET_EPOCHS
    cfg["_note_extended_from_ep20"] = True
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\n[extend] Listo. {MODEL_DIR} ahora tiene {TARGET_EPOCHS} epocas "
          f"(pesos de epoca 20 preservados en {BACKUP_DIR}).")


if __name__ == "__main__":
    main()
