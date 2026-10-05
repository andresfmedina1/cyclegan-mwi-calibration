"""
splits.py

Genera el split train/validation/test 80/10/10, GENUINAMENTE disjunto (la
validacion por-epoca y el test final nunca comparten muestras), con seed=42
via np.random.default_rng. Soporta dominios unpaired (target-background: A y B
se parten con permutaciones INDEPENDIENTES) y paired (target-target: UNA
permutacion define los 3 subsets, usada identicamente en A y B, preservando el
pareo fisico A[i]/B[i] en cada particion).
"""
import json

import numpy as np

SCHEMES = {
    "8010_3way": {"kind": "fraction", "val": 0.10, "test": 0.10},
}


def _split_sizes(n, scheme):
    s = SCHEMES[scheme]
    if s["kind"] == "absolute":
        n_val, n_test = s["val"], s["test"]
    else:
        n_val = int(round(n * s["val"]))
        n_test = int(round(n * s["test"]))
    assert n_val + n_test < n, f"n={n} demasiado chico para scheme={scheme}"
    return n_val, n_test


def split_unpaired(n_a, n_b, scheme, seed=42):
    """
    Para dominios A/B UNPAIRED (target-background): rng.permutation llamado
    una vez por dominio, en orden A luego B, con 3 franjas:
    perm[:n_test]=test, perm[n_test:n_test+n_val]=val, resto=train.

    Devuelve dict:
      {"A": {"train": idx, "val": idx, "test": idx},
       "B": {"train": idx, "val": idx, "test": idx},
       "scheme": scheme, "seed": seed}
    """
    rng = np.random.default_rng(seed)
    out = {"scheme": scheme, "seed": seed}
    for dom, n in (("A", n_a), ("B", n_b)):
        n_val, n_test = _split_sizes(n, scheme)
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]
        out[dom] = {"train": train_idx, "val": val_idx, "test": test_idx}
    return out


def split_paired(n, scheme, seed=42, pool_idx=None):
    """
    Para dominios A/B PAIRED por indice (target-target): UNA permutacion
    define los 3 subsets, usados identicamente para A y B.

    pool_idx : np.ndarray opcional de indices (dentro de 0..n-1) desde los
        que muestrear -- usado por TT-supervised para excluir Isc2 del pool
        ANTES de particionar (ver retrain_tt_supervised.py). Si None, se usa
        el pool completo range(n). Los indices devueltos son siempre
        indices ABSOLUTOS en el tensor original (0..n-1), directamente
        usables para indexar A_tensor/B_tensor -- NO indices dentro del pool.

    Devuelve dict:
      {"train": idx, "val": idx, "test": idx, "scheme": scheme, "seed": seed,
       "pool_size": len(pool)}
    """
    pool = np.arange(n) if pool_idx is None else np.asarray(pool_idx)
    rng = np.random.default_rng(seed)
    n_val, n_test = _split_sizes(len(pool), scheme)
    perm = rng.permutation(pool)  # permutacion de los VALORES del pool
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx,
            "scheme": scheme, "seed": seed, "pool_size": int(len(pool))}


def save_split_npz(path, split_dict):
    """Guarda un split (unpaired o paired) en un .npz, todo int64."""
    flat = {}
    if "A" in split_dict:  # unpaired
        for dom in ("A", "B"):
            for k, v in split_dict[dom].items():
                flat[f"{dom}_{k}"] = np.asarray(v, dtype=np.int64)
    else:  # paired
        for k in ("train", "val", "test"):
            flat[k] = np.asarray(split_dict[k], dtype=np.int64)
    np.savez(path, **flat)


def save_split_meta_json(path, split_dict):
    """Guarda metadatos legibles (tamaños, scheme, seed) como JSON companion."""
    meta = {"scheme": split_dict["scheme"], "seed": split_dict["seed"]}
    if "A" in split_dict:
        meta["sizes"] = {dom: {k: int(len(v)) for k, v in split_dict[dom].items()}
                          for dom in ("A", "B")}
    else:
        meta["sizes"] = {k: int(len(split_dict[k])) for k in ("train", "val", "test")}
        meta["pool_size"] = split_dict["pool_size"]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
