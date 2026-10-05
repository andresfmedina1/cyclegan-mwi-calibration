"""
retrain_common.py

Runner compartido por las corridas de entrenamiento con split 80/10/10,
genuinamente disjunto (fix de leakage donde el mismo "test"
se usaba tambien como validacion por-epoca) y normalizacion fit-en-train-only
(segunda fuga corregida de paso). Usado por retrain_baseline.py,
retrain_run8.py, retrain_tt_unsupervised.py, retrain_tt_supervised.py.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cyclegan_model import CycleGan
from data_io import Normalizer, load_compressed_datasets
from splits import save_split_meta_json, save_split_npz, split_paired, split_unpaired


def plot_training_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    has_val = len(history.get("val_cycle", [])) > 0

    n_panels = 4 if has_val else 2
    fig, ax = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

    ax[0].plot(history["g_loss"], label="G total")
    ax[0].plot(history["g_adv"], label="G adversarial")
    ax[0].plot(history["g_cycle"], label="G cycle")
    ax[0].plot(history["g_identity"], label="G identity")
    ax[0].plot(history["g_scale"], label="G scale")
    ax[0].set_title("Generator losses (training)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()

    ax[1].plot(history["d_loss"], label="D total")
    ax[1].set_title("Discriminator loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()

    if has_val:
        # Train vs. validacion para cycle + identity: si las curvas de
        # validacion se despegan hacia arriba mientras las de entrenamiento
        # siguen bajando, es la señal clasica de overfitting (el Cycle-GAN
        # esta memorizando el train set en vez de generalizar).
        ax[2].plot(history["g_cycle"], label="cycle (train)", color="tab:blue")
        ax[2].plot(history["val_cycle"], label="cycle (val)", color="tab:blue",
                   linestyle="--")
        ax[2].plot(history["g_identity"], label="identity (train)", color="tab:orange")
        ax[2].plot(history["val_identity"], label="identity (val)", color="tab:orange",
                   linestyle="--")
        ax[2].set_title("Train vs. validation (cycle + identity)")
        ax[2].set_xlabel("epoch")
        ax[2].legend()

        # Panel dedicado a la perdida de escala (train vs val): es la señal
        # mas directa de si L_scale esta ayudando a que la magnitud de la
        # salida se acerque a la del dominio destino.
        ax[3].plot(history["g_scale"], label="scale (train)", color="tab:green")
        ax[3].plot(history["val_scale"], label="scale (val)", color="tab:green",
                   linestyle="--")
        ax[3].set_title("Train vs. validation (scale)")
        ax[3].set_xlabel("epoch")
        ax[3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_history.jpg"))
    fig.savefig(os.path.join(out_dir, "training_history.pdf"))
    plt.close(fig)


def prepare_checkpoint_dir(model_dir):
    os.makedirs(model_dir, exist_ok=True)


def save_config_json(model_dir, cfg):
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def save_history_json(model_dir, history):
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f)


def build_cyclegan(img_shape, model_dir, cfg):
    """cfg debe incluir los kwargs de CycleGan.__init__ que aplican
    (arquitectura, gammas, lrs, symmetry, y opcionalmente
    gen_kernel_init/disc_kernel_init/disc_final_activation para el baseline)."""
    return CycleGan(
        img_shape, model_dir,
        gen_dim=cfg["gen_dim"], gen_depth=cfg["gen_depth"], gen_kernel=cfg["gen_kernel"],
        disc_dim=cfg["disc_dim"], disc_depth=cfg["disc_depth"], disc_kernel=cfg["disc_kernel"],
        gamma_cycle=cfg["gamma_cycle"], gamma_identity=cfg["gamma_identity"],
        gamma_scale=cfg["gamma_scale"], gamma_supervised=cfg.get("gamma_supervised", 0.0),
        lr_generator=cfg["lr_generator"], lr_discriminator=cfg["lr_discriminator"],
        apply_output_mask=cfg["apply_output_mask"], ab_mask=None,
        enforce_symmetry_a2b=cfg["enforce_symmetry_a2b"],
        enforce_symmetry_b2a=cfg["enforce_symmetry_b2a"],
        gen_kernel_init=cfg.get("gen_kernel_init", "he_uniform"),
        disc_kernel_init=cfg.get("disc_kernel_init", "he_uniform"),
        disc_final_activation=cfg.get("disc_final_activation", "tanh"),
    )


def _train_in_chunks(cycle_gan, train_A, train_B, val_A, val_B, cfg, model_dir,
                      checkpoint_epochs, paired):
    checkpoint_epochs = checkpoint_epochs or [cfg["epochs"]]
    epochs_done = 0
    for target_epoch in checkpoint_epochs:
        n_chunk = target_epoch - epochs_done
        print(f"[retrain_common] entrenando {n_chunk} epocas "
              f"({epochs_done} -> {target_epoch}) en {model_dir}")
        cycle_gan.train(train_A, train_B, epochs=n_chunk, batch_size=cfg["batch_size"],
                         verbose_every=cfg["verbose_every"],
                         test_A=val_A, test_B=val_B, paired=paired)
        epochs_done = target_epoch
        is_final = target_epoch == checkpoint_epochs[-1]
        cycle_gan.model_dir = model_dir if is_final else f"{model_dir}_chunk_ep{target_epoch}"
        prepare_checkpoint_dir(cycle_gan.model_dir)
        cycle_gan.save()
        save_history_json(cycle_gan.model_dir, cycle_gan.history)
        plot_training_history(cycle_gan.history, cycle_gan.model_dir)
    return cycle_gan


def run_unpaired_job(cfg, scheme, checkpoint_epochs=None):
    """
    target-background (Baseline, Run8): A/B unpaired, split independiente
    para cada dominio.
    checkpoint_epochs : lista opcional de epocas ACUMULADAS en las que
        guardar (p.ej. [100] para un solo guardado final). Si None,
        equivale a [cfg['epochs']].
    """
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    (A_tensor, B_tensor, A_meta, B_meta, _nA, _nB, freq_A, freq_B) = \
        load_compressed_datasets(cfg["synthetic_h5_path"], cfg["experimental_h5_path"])

    split = split_unpaired(len(A_tensor), len(B_tensor), scheme, seed=cfg["seed"])

    norm_A = Normalizer.fit(A_tensor[split["A"]["train"]])
    norm_B = Normalizer.fit(B_tensor[split["B"]["train"]])
    A_norm, B_norm = norm_A.forward(A_tensor), norm_B.forward(B_tensor)

    train_A, val_A, test_A = (A_norm[split["A"][k]] for k in ("train", "val", "test"))
    train_B, val_B, test_B = (B_norm[split["B"][k]] for k in ("train", "val", "test"))

    model_dir = cfg["model_dir"]
    prepare_checkpoint_dir(model_dir)
    save_split_npz(os.path.join(model_dir, "split_indices.npz"), split)
    save_split_meta_json(os.path.join(model_dir, "split_indices.json"), split)
    full_cfg = {**cfg, "scheme": scheme,
                "norm_A": norm_A.to_dict(), "norm_B": norm_B.to_dict(),
                "n_train_A": int(len(train_A)), "n_val_A": int(len(val_A)), "n_test_A": int(len(test_A)),
                "n_train_B": int(len(train_B)), "n_val_B": int(len(val_B)), "n_test_B": int(len(test_B))}
    save_config_json(model_dir, full_cfg)
    print(f"[retrain_common] {model_dir}: A train/val/test = "
          f"{len(train_A)}/{len(val_A)}/{len(test_A)}, "
          f"B train/val/test = {len(train_B)}/{len(val_B)}/{len(test_B)}")

    img_shape = A_tensor.shape[1:]
    cycle_gan = build_cyclegan(img_shape, model_dir, cfg)
    _train_in_chunks(cycle_gan, train_A, train_B, val_A, val_B, cfg, model_dir,
                      checkpoint_epochs, paired=False)

    return model_dir, split, norm_A, norm_B


def run_paired_job(cfg, scheme, checkpoint_epochs=None, paired_training=False,
                    pool_idx=None):
    """
    target-target (TT-unsupervised, TT-supervised): A/B paired por indice,
    UNA permutacion define train/val/test para ambos dominios a la vez.

    paired_training : pasado tal cual a cycle_gan.train(..., paired=...).
        False para TT-unsupervised (gamma_supervised=0, batches shuffled
        independientemente); True para TT-supervised (requiere L_supervised
        con batches index-aligned).
    pool_idx : ver splits.split_paired -- usado por TT-supervised para
        excluir Isc2 del pool de origen antes de partir train/val/test.
    """
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    (A_tensor, B_tensor, A_meta, B_meta, _nA, _nB, freq_A, freq_B) = \
        load_compressed_datasets(cfg["synthetic_h5_path"], cfg["experimental_h5_path"])
    assert A_tensor.shape[0] == B_tensor.shape[0], \
        "target-target debe estar pareado por indice (mismo N en A y B)"

    split = split_paired(A_tensor.shape[0], scheme, seed=cfg["seed"], pool_idx=pool_idx)

    norm_A = Normalizer.fit(A_tensor[split["train"]])
    norm_B = Normalizer.fit(B_tensor[split["train"]])
    A_norm, B_norm = norm_A.forward(A_tensor), norm_B.forward(B_tensor)

    train_A, val_A, test_A = (A_norm[split[k]] for k in ("train", "val", "test"))
    train_B, val_B, test_B = (B_norm[split[k]] for k in ("train", "val", "test"))

    model_dir = cfg["model_dir"]
    prepare_checkpoint_dir(model_dir)
    save_split_npz(os.path.join(model_dir, "split_indices.npz"), split)
    save_split_meta_json(os.path.join(model_dir, "split_indices.json"), split)
    full_cfg = {**cfg, "scheme": scheme, "paired_training": paired_training,
                "excludes_isc2": pool_idx is not None,
                "norm_A": norm_A.to_dict(), "norm_B": norm_B.to_dict(),
                "n_train": int(len(train_A)), "n_val": int(len(val_A)), "n_test": int(len(test_A))}
    save_config_json(model_dir, full_cfg)
    print(f"[retrain_common] {model_dir}: train/val/test = "
          f"{len(train_A)}/{len(val_A)}/{len(test_A)} "
          f"(pool_size={split.get('pool_size')})")

    img_shape = A_tensor.shape[1:]
    cycle_gan = build_cyclegan(img_shape, model_dir, cfg)
    _train_in_chunks(cycle_gan, train_A, train_B, val_A, val_B, cfg, model_dir,
                      checkpoint_epochs, paired=paired_training)

    return model_dir, split, norm_A, norm_B
