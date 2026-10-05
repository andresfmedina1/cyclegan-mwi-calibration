"""
data_io.py

Carga y preprocesamiento de los dos dominios usados para entrenar el
Cycle-GAN de calibracion experimental (Ian Jeffrey et al. 2023,
"Experimental Microwave Imaging System Calibration via Cycle-GAN", IEEE TAP).

============================================================================
DOMINIO A -- sintetico / "limpio" (salida del pipeline FEM)
============================================================================
Fuente   : OSMI---Open-Source-Microwave-Imaging-main/Examples/Head_imaging/
           results/simulation_dataset.h5
Generado por: main_generate_simulation_dataset.m  (linea 365)
Contenido: grupos 'sim_XXXX', cada uno con:
    Esc_real, Esc_imag  [12x12] float64
    freq                escalar (1.3e9 Hz, unica frecuencia)
    attrs: meas_type, target_type, target_epsilon_re, target_eps_label,
           pos_x_mm, pos_y_mm, rotation_deg

Esc = (E_total - E_background) YA es diferencial (se resta dentro del .m,
ver linea 365: `Esc = (Ep_tot(...) - Ep_bg(...)) .* meas_mask;`), y ya viene
con la mascara fisica A<->B aplicada (solo 72/144 entradas != 0).
-> load_synthetic_domain() NO vuelve a restar nada, solo lee y filtra.

IMPORTANTE: a fecha de escritura de este modulo el .h5 esta INCOMPLETO
(el script de MATLAB genera hasta 1216 escenarios objetivo: 3 tipos de
target x posiciones x 4 rotaciones x 4 permitividades, pero puede seguir
corriendo en background). Este loader simplemente usa lo que exista en el
archivo en el momento de leerlo, e imprime cuantas muestras encontro.

============================================================================
DOMINIO B -- experimental / "corrupto" (mediciones reales con el robot)
============================================================================
Fuente   : Robot_Acquisition/measurements_database_robot_V2.h5
Contenido: grupos 'acq_XXXX' con:
    Sm_real, Sm_imag  [11,12,12] float64  (barrido VNA 0.8-1.8 GHz, 11 pts)
    freq              [1,11] float64
    attrs: meas_type ('healthy' | 'target'), target_type,
           target_epsilon_re, target_eps_label, pos_x_mm, pos_y_mm,
           rotation_deg, timestamp

A diferencia del dominio A, Sm_real/Sm_imag es CAMPO TOTAL crudo, no
diferencial. Las acquisitions con meas_type == 'healthy' son remediciones
periodicas del background (se remide cada cierto numero de targets para
corregir drift termico/temporal del VNA -- ver repetitividad_analysis.m).
-> load_experimental_domain() SI debe construir la diferencial:

       dS_target = S_target(f) - S_healthy_mas_cercana_anterior(f)

   usando la 'healthy' inmediatamente anterior en indice de acquisition
   (no la primera del archivo, no la mas cercana en eps): asi es como se
   midio fisicamente en el laboratorio (medir background, luego N targets,
   volver a medir background, ...).
"""

import re

import h5py
import numpy as np

# =============================================================================
# Mascara fisica A<->B
# =============================================================================
# Antenas 1-6 -> switch A (indices 0-5 en 0-based), antenas 7-12 -> switch B
# (indices 6-11). Los switches SP8T no permiten auto-conexion, por lo que
# los pares A-A y B-B NUNCA se miden fisicamente: solo existen los 72
# pares cruzados A<->B de los 144 posibles en una matriz 12x12.
# Esta mascara debe coincidir EXACTAMENTE con `meas_mask` en el script de
# MATLAB (main_generate_simulation_dataset.m, lineas 60-62):
#   meas_mask(1:nA, nA+1:N) = true;   ->  aqui: mask[:6, 6:] = True
#   meas_mask(nA+1:N, 1:nA) = true;   ->  aqui: mask[6:, :6] = True
N_ANTENNAS = 12
N_A = 6  # numero de antenas del switch A


def build_ab_mask(n_antennas=N_ANTENNAS, n_a=N_A):
    """Mascara booleana (n,n): True donde el par Tx-Rx es fisicamente medible."""
    mask = np.zeros((n_antennas, n_antennas), dtype=bool)
    mask[:n_a, n_a:] = True  # bloque A -> B
    mask[n_a:, :n_a] = True  # bloque B -> A
    return mask


AB_MASK = build_ab_mask()  # (12, 12) bool, 72 entradas True


# =============================================================================
# Compresion 12x12 -> 6x12 (elimina los bloques A-A y B-B, siempre invalidos)
# =============================================================================
# La matriz completa 12x12 tiene 4 bloques de 6x6:
#   [ A-A (invalido, =0) | A-B (valido) ]
#   [ B-A (valido)       | B-B (invalido, =0) ]
#
# Los bloques A-A y B-B nunca contienen informacion (los switches SP8T no
# permiten auto-conexion), asi que guardarlos es puro desperdicio: de 144
# entradas solo 72 son utiles. Se empaquetan las dos mitades utiles en una
# matriz compacta (6, 12) SIN ceros de relleno:
#
#   compressed[i, 0:6]  = S[A_i -> B_j]   para j = 0..5   (bloque A-B, fila i)
#   compressed[i, 6:12] = S[B_j -> A_i]   para j = 0..5   (bloque B-A, columna i,
#                                                           transpuesto para que
#                                                           la fila i siga
#                                                           correspondiendo
#                                                           siempre a la antena A_i)
#
# Es decir, cada fila i representa "todo lo relacionado con la antena A_i":
# las 6 mediciones en las que A_i transmite y B recibe, seguidas de las 6
# mediciones en las que B transmite y A_i recibe.
def compress_ab_pairs(mat_12x12, n_a=N_A):
    """
    mat_12x12 : np.ndarray (12,12), real o complejo (con o sin mascara
                aplicada, los bloques A-A/B-B se descartan igual)
    Devuelve  : np.ndarray (6,12), mismo dtype que la entrada
    """
    assert mat_12x12.shape == (2 * n_a, 2 * n_a), \
        f"se esperaba una matriz ({2*n_a},{2*n_a}), llego {mat_12x12.shape}"
    ab_block = mat_12x12[:n_a, n_a:]        # (6,6): S[A_i -> B_j]
    ba_block = mat_12x12[n_a:, :n_a]        # (6,6): S[B_j -> A_i], fila=B, col=A
    return np.concatenate([ab_block, ba_block.T], axis=1)  # (6,12)


def expand_ab_pairs(mat_6x12, n_a=N_A):
    """
    Inverso de compress_ab_pairs: reconstruye la matriz (12,12) completa,
    con los bloques A-A y B-B forzados a cero. Util para verificar la
    compresion o para reusar codigo/visualizaciones pensadas para 12x12.
    """
    assert mat_6x12.shape == (n_a, 2 * n_a), \
        f"se esperaba una matriz ({n_a},{2*n_a}), llego {mat_6x12.shape}"
    ab_block = mat_6x12[:, :n_a]      # (6,6): S[A_i -> B_j]
    ba_block_T = mat_6x12[:, n_a:]    # (6,6): S[B_j -> A_i], indexado [A_i, B_j]

    full = np.zeros((2 * n_a, 2 * n_a), dtype=mat_6x12.dtype)
    full[:n_a, n_a:] = ab_block
    full[n_a:, :n_a] = ba_block_T.T   # deshace la transposicion
    return full


def compress_ab_pairs_list(mat_list):
    """Aplica compress_ab_pairs a una lista de matrices (12,12) -> (6,12)."""
    return [compress_ab_pairs(m) for m in mat_list]


def _to_str(x):
    """h5py a veces devuelve bytes para strings escritos desde MATLAB."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _scalar_attr(attrs, key, default=np.nan):
    """Lee un atributo numerico h5 (viene como array de 1 elemento) como float."""
    if key not in attrs:
        return default
    return float(np.ravel(attrs[key])[0])


def _sorted_group_keys(h5file):
    """Ordena las claves 'acq_XXXX' / 'sim_XXXX' por su indice numerico final."""
    keys = list(h5file.keys())
    return sorted(keys, key=lambda k: int(re.search(r"(\d+)$", k).group(1)))


def _extract_common_meta(attrs):
    return {
        "target_type": _to_str(attrs.get("target_type", "NA")),
        "target_epsilon_re": _scalar_attr(attrs, "target_epsilon_re"),
        "target_eps_label": _to_str(attrs.get("target_eps_label", "NA")),
        "pos_x_mm": _scalar_attr(attrs, "pos_x_mm"),
        "pos_y_mm": _scalar_attr(attrs, "pos_y_mm"),
        "rotation_deg": _scalar_attr(attrs, "rotation_deg"),
    }


# =============================================================================
# DOMINIO A -- simulacion FEM (ya diferencial, ya enmascarada, 1.3 GHz unico)
# =============================================================================
def load_synthetic_domain(sim_h5_path, apply_mask=True, verbose=True):
    """
    Lee results/simulation_dataset.h5.

    Parametros
    ----------
    sim_h5_path : str
        Ruta al .h5 generado por main_generate_simulation_dataset.m.
    apply_mask : bool
        Si True, fuerza a cero las entradas fuera de la mascara A<->B (por
        seguridad; en teoria ya vienen en cero desde MATLAB).

    Devuelve
    --------
    A_list : list[np.ndarray complex128 (12,12)]
        Cada elemento es Esc_real + 1j*Esc_imag (ya diferencial).
    meta   : list[dict]
        Metadatos por muestra (target_type, target_epsilon_re, pos_x_mm,
        pos_y_mm, rotation_deg).
    """
    A_list, meta = [], []
    with h5py.File(sim_h5_path, "r") as f:
        for key in _sorted_group_keys(f):
            g = f[key]
            meas_type = _to_str(g.attrs.get("meas_type", "NA"))
            # El script de MATLAB no genera grupos 'healthy' (Esc=0 trivial,
            # ver comentario "healthy = Esc all-zeros, redundant" en el .m),
            # pero se filtra por seguridad si alguno llegara a existir.
            if meas_type != "target":
                continue

            Esc = g["Esc_real"][()] + 1j * g["Esc_imag"][()]
            if apply_mask:
                Esc = Esc * AB_MASK

            A_list.append(Esc)
            sample_meta = _extract_common_meta(g.attrs)
            # Correccion de un problema de CONVENCION en rotation_deg (NO en
            # los valores de Esc): confirmado por el usuario comparando
            # contra el robot, rotation_deg=0 en simulation_dataset_V2.h5
            # corresponde FISICAMENTE a 90 grados, y rotation_deg=90
            # corresponde a 0 grados (-45/45 SI coinciden entre dominios).
            # Consistente con que main_generate_simulation_dataset.m arma su
            # grid de rotaciones (rotations_deg = [-45,0,45,90], linea ~141)
            # copiando los 4 valores canonicos SIN aplicarles la correccion
            # de reflexion que si le hacia falta (la misma que csv_angle_to_
            # dataset le aplica a target_intersections.csv mas abajo en este
            # archivo) -- es decir, el rotation_deg guardado ahi quedo,
            # efectivamente, en el marco de positions/stroke_positions.m en
            # vez del marco real robot/mesh. Se reutiliza la MISMA funcion
            # para corregirlo aca. Esto NO afecta el pipeline target-
            # background (find_analogous_sample/find_analog_index nunca usan
            # rotation_deg para emparejar), solo importa para target-target.
            sample_meta["rotation_deg"] = csv_angle_to_dataset(sample_meta["rotation_deg"])
            meta.append(sample_meta)

    if verbose:
        print(f"[data_io] Dominio A (sintetico): {len(A_list)} muestras leidas de "
              f"'{sim_h5_path}'")
        if len(A_list) < 50:
            print("[data_io][WARN] Muy pocas muestras sinteticas todavia: "
                  "main_generate_simulation_dataset.m aun no ha terminado de "
                  "generar el dataset completo (hasta 1216 escenarios). El "
                  "entrenamiento del Cycle-GAN con tan pocas muestras de "
                  "dominio A no sera representativo, re-ejecuta este loader "
                  "cuando el .h5 este mas completo.")

    if len(A_list) == 0:
        raise RuntimeError(f"No se encontraron muestras 'target' en {sim_h5_path}")

    return A_list, meta


# =============================================================================
# Correccion de un error de ETIQUETADO conocido (NO de datos corruptos) en
# measurements_database_robot_V2.h5
# =============================================================================
# acq_818 a acq_843 (26 acquisitions, Target 2 + Isc2, rotation_deg=-45) y
# acq_870 a acq_895 (otras 26, misma combinacion Target 2 + Isc2,
# rotation_deg=-45 TAMBIEN) resultaron ser la MISMA secuencia de 26
# posiciones repetida dos veces (mismo orden exacto de pos_x_mm/pos_y_mm en
# ambas tandas, confirmado). La primera tanda (818-843) es genuinamente
# -45 grados. La segunda (870-895) esta MAL ETIQUETADA: en realidad
# corresponde a la tanda de +45 grados -- confirmado por el usuario, y
# consistente con que, sin esta correccion, Target2+Isc2+45deg tiene
# EXACTAMENTE 0 acquisitions en todo el archivo (el hueco que esta segunda
# tanda llena) mientras que Target2+Isc2+(-45deg) aparece con 52 (doble de
# lo esperado, 26+26).
#
# A diferencia de exclude_acq_numbers (datos invalidos, se descartan), esto
# es un dato VALIDO pero con un atributo mal escrito -- se corrige en
# memoria, nunca se toca el .h5 original.
MISLABELED_ROTATION_ACQ = {acq: 45.0 for acq in range(870, 896)}  # acq_870..acq_895


def _fix_known_mislabels(acq_num, meta):
    """Corrige rotation_deg para acquisitions con etiquetado incorrecto
    conocido en el .h5 crudo (ver MISLABELED_ROTATION_ACQ). Devuelve un
    dict nuevo, no modifica meta in-place."""
    if acq_num in MISLABELED_ROTATION_ACQ:
        meta = dict(meta)
        meta["rotation_deg"] = MISLABELED_ROTATION_ACQ[acq_num]
    return meta


# =============================================================================
# DOMINIO B -- mediciones robot (campo total crudo -> hay que restar background)
# =============================================================================
def load_experimental_domain(robot_h5_path, freq_hz=1.3e9, apply_mask=True,
                              verbose=True, exclude_acq_numbers=None):
    """
    Lee measurements_database_robot_V2.h5 y construye la matriz diferencial
    dS = S_target - S_healthy_previa para cada acquisition 'target', a la
    frecuencia mas cercana a freq_hz.

    Parametros
    ----------
    robot_h5_path : str
        Ruta a measurements_database_robot_V2.h5.
    freq_hz : float
        Frecuencia (Hz) a extraer del barrido VNA de 11 puntos. Por defecto
        1.3 GHz para alinear con la unica frecuencia resuelta en el dominio
        sintetico (ver sim_f en main_generate_simulation_dataset.m).
    apply_mask : bool
        Fuerza a cero las entradas fuera de la mascara A<->B. Es necesario
        aqui (a diferencia del dominio A) porque los datos crudos del VNA
        SI tienen valores no nulos en los bloques A-A/B-B (crosstalk /
        aislamiento de los switches, no mediciones fisicas validas).
    exclude_acq_numbers : coleccion de int, opcional
        Numeros de acquisition (el XXXX en 'acq_XXXX') a excluir POR
        COMPLETO -- ni como target ni como referencia de background para
        las targets siguientes -- por ejemplo por estar corrompidas. Se
        tratan como si no existieran en el archivo. Default: ninguna
        exclusion.

    Devuelve
    --------
    B_list : list[np.ndarray complex128 (12,12)]
        Cada elemento es dS = S_target(f) - S_healthy_previa(f).
    meta   : list[dict]
        Metadatos por muestra, incluye ademas 'timestamp'.
    """
    exclude_acq_numbers = set(exclude_acq_numbers) if exclude_acq_numbers else set()

    B_list, meta = [], []
    n_skipped_no_bg = 0
    n_excluded = 0

    with h5py.File(robot_h5_path, "r") as f:
        keys = _sorted_group_keys(f)

        # Todas las acquisitions comparten el mismo barrido de frecuencias,
        # asi que basta con leerlo una vez.
        freqs = f[keys[0]]["freq"][()].flatten()
        f_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        actual_freq = freqs[f_idx]
        if not np.isclose(actual_freq, freq_hz, rtol=1e-6):
            print(f"[data_io][WARN] freq_hz={freq_hz:.4e} Hz no esta exactamente "
                  f"en el barrido; usando {actual_freq:.4e} Hz (indice {f_idx}).")

        last_healthy_S = None
        for key in keys:
            acq_num = int(re.search(r"(\d+)$", key).group(1))
            if acq_num in exclude_acq_numbers:
                n_excluded += 1
                continue

            g = f[key]
            meas_type = _to_str(g.attrs.get("meas_type", "NA"))
            S = g["Sm_real"][f_idx] + 1j * g["Sm_imag"][f_idx]

            if meas_type == "healthy":
                # Se guarda como referencia de background para las targets
                # que sigan, hasta la proxima 'healthy'.
                last_healthy_S = S
                continue

            if meas_type != "target":
                continue

            if last_healthy_S is None:
                # No deberia pasar (la primera acquisition del archivo es
                # siempre 'healthy'), pero se descarta por seguridad.
                n_skipped_no_bg += 1
                continue

            dS = S - last_healthy_S
            if apply_mask:
                dS = dS * AB_MASK

            B_list.append(dS)
            sample_meta = _extract_common_meta(g.attrs)
            sample_meta["timestamp"] = _to_str(g.attrs.get("timestamp", "NA"))
            sample_meta = _fix_known_mislabels(acq_num, sample_meta)
            meta.append(sample_meta)

    if verbose:
        print(f"[data_io] Dominio B (experimental): {len(B_list)} muestras "
              f"diferenciales construidas de '{robot_h5_path}' "
              f"(f = {actual_freq/1e9:.3f} GHz)")
        if n_excluded:
            print(f"[data_io][WARN] {n_excluded} acquisitions excluidas "
                  f"explicitamente via exclude_acq_numbers.")
        if n_skipped_no_bg:
            print(f"[data_io][WARN] {n_skipped_no_bg} muestras descartadas por "
                  f"no tener background previo.")

    if len(B_list) == 0:
        raise RuntimeError(f"No se encontraron muestras 'target' en {robot_h5_path}")

    return B_list, meta


def get_experimental_sample_before_after(robot_h5_path, freq_hz, target_order_index):
    """
    Utilidad de depuracion/visualizacion (NO se usa para construir el
    dataset de entrenamiento, solo para inspeccionar el efecto de la resta).

    Para el `target_order_index`-esimo acquisition de tipo 'target' del
    archivo (mismo orden que produce load_experimental_domain, o sea
    target_order_index=i corresponde exactamente a B_list[i]/B_meta[i]),
    devuelve por separado el ANTES y el DESPUES de restar el background:

        S_target_raw  : campo total crudo de esa acquisition (12,12), SIN
                        enmascarar. Los bloques A-A/B-B tienen valores no
                        nulos (aislamiento/crosstalk de los switches, no
                        mediciones fisicas validas).
        S_healthy_raw : campo total crudo de la 'healthy' inmediatamente
                        anterior, usada como background (12,12), SIN
                        enmascarar.
        dS_unmasked   : S_target_raw - S_healthy_raw (12,12), diferencial
                        SIN enmascarar todavia.
        dS_masked     : dS_unmasked con la mascara A<->B aplicada -- esto es
                        EXACTAMENTE lo que devuelve load_experimental_domain
                        para esta misma muestra.
        meta          : metadatos de la acquisition target.
    """
    with h5py.File(robot_h5_path, "r") as f:
        keys = _sorted_group_keys(f)
        freqs = f[keys[0]]["freq"][()].flatten()
        f_idx = int(np.argmin(np.abs(freqs - freq_hz)))

        last_healthy_S = None
        target_counter = -1
        for key in keys:
            g = f[key]
            meas_type = _to_str(g.attrs.get("meas_type", "NA"))
            S = g["Sm_real"][f_idx] + 1j * g["Sm_imag"][f_idx]

            if meas_type == "healthy":
                last_healthy_S = S
                continue
            if meas_type != "target":
                continue
            if last_healthy_S is None:
                continue

            target_counter += 1
            if target_counter == target_order_index:
                dS_unmasked = S - last_healthy_S
                dS_masked = dS_unmasked * AB_MASK
                meta = _extract_common_meta(g.attrs)
                meta["timestamp"] = _to_str(g.attrs.get("timestamp", "NA"))
                return S, last_healthy_S, dS_unmasked, dS_masked, meta

    raise IndexError(f"No existe target_order_index={target_order_index} en {robot_h5_path}")


def load_experimental_raw_targets(robot_h5_path, freq_hz=1.3e9, apply_mask=True,
                                   verbose=True, exclude_acq_numbers=None):
    """
    A diferencia de load_experimental_domain (que resta el 'healthy' mas
    cercano para obtener dS = S_target - S_healthy_previa), esta funcion
    devuelve el campo TOTAL CRUDO S_target de cada acquisition 'target', SIN
    restar ningun background.

    Se usa para construir deltas TARGET-TARGET (S_target_A - S_target_B,
    ver seccion "Delta target-target" mas abajo): como ambas mediciones son
    campo crudo de la MISMA cadena de medicion, el delta target-target NO
    involucra ningun background de referencia. Si en cambio se restaran dos
    dS ya diferenciados (dS_A - dS_B), quedaria un residuo de background
    salvo que A y B hayan compartido exactamente el mismo 'healthy' previo
    -- cosa que no se puede asumir en general para pares arbitrarios.

    Parametros
    ----------
    (mismos que load_experimental_domain)

    Devuelve
    --------
    S_list : list[np.ndarray complex128 (12,12)]  -- crudo, SIN diferenciar
    meta   : list[dict]  -- igual que _extract_common_meta + 'acq_num' y
             'timestamp'
    """
    exclude_acq_numbers = set(exclude_acq_numbers) if exclude_acq_numbers else set()

    S_list, meta = [], []
    with h5py.File(robot_h5_path, "r") as f:
        keys = _sorted_group_keys(f)
        freqs = f[keys[0]]["freq"][()].flatten()
        f_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        actual_freq = freqs[f_idx]

        for key in keys:
            acq_num = int(re.search(r"(\d+)$", key).group(1))
            if acq_num in exclude_acq_numbers:
                continue

            g = f[key]
            meas_type = _to_str(g.attrs.get("meas_type", "NA"))
            if meas_type != "target":
                continue

            S = g["Sm_real"][f_idx] + 1j * g["Sm_imag"][f_idx]
            if apply_mask:
                S = S * AB_MASK

            m = _extract_common_meta(g.attrs)
            m["acq_num"] = acq_num
            m["timestamp"] = _to_str(g.attrs.get("timestamp", "NA"))
            m = _fix_known_mislabels(acq_num, m)
            S_list.append(S)
            meta.append(m)

    if verbose:
        print(f"[data_io] {len(S_list)} mediciones 'target' CRUDAS (sin diferenciar) "
              f"leidas de '{robot_h5_path}' (f={actual_freq/1e9:.3f} GHz)")

    if len(S_list) == 0:
        raise RuntimeError(f"No se encontraron muestras 'target' en {robot_h5_path}")

    return S_list, meta


# =============================================================================
# Delta target-target (aumento de datos): posiciones del CSV de
# intersecciones (positions/stroke_positions.m) -> marco del dataset
# =============================================================================
# target_intersections.csv (generado por positions/stroke_positions.m) lista
# pares de targets (de DISTINTO tamano, T1xT2 / T1xT3 / T2xT3 -- el script
# original solo calculo solapamientos entre tamanos distintos) cuyas elipses
# se solapan geometricamente, en el marco de coordenadas PROPIO de esa
# herramienta (basado en la imagen del contorno del phantom), no en el marco
# de pos_x_mm/pos_y_mm/rotation_deg que usan los .h5 del dataset.
#
# POSICION: verificado EMPIRICAMENTE comparando las 42+26+8 posiciones
# validas de valid_positions_T1/T2/T3.csv contra las posiciones REALES
# usadas en simulation_dataset_V2.h5 (pos_T1/T2/T3 en
# main_generate_simulation_dataset.m, "extraidas del HDF5 experimental"):
# de las 8 transformaciones candidatas (identidad, swap, negaciones
# combinadas), la UNICA que da 100% de coincidencia exacta en los TRES
# tamanos de target simultaneamente es el intercambio PURO de ejes (sin
# cambio de signo). Target 2 es el desempate: solo el swap puro da 26/26,
# las variantes con signo dan 24-25/26.
def csv_position_to_dataset(x_csv, y_csv):
    """Es su propia inversa (aplicar dos veces devuelve el original)."""
    return y_csv, x_csv


# ANGULO: la posicion es un intercambio PURO de ejes, es decir una REFLEXION
# (no una rotacion) -- por lo tanto el angulo de rotacion de la elipse
# TAMBIEN cambia, no se puede dejar igual. Derivacion: representando el
# contorno de la elipse como R(ang)*[rx*cos(t); ry*sin(t)] (misma convencion
# en ambos scripts .m: positivo = antihorario, rx a lo largo del eje x local
# en ang=0 -- confirmado leyendo tanto stroke_positions.m como
# main_generate_simulation_dataset.m) y aplicando la reflexion swap(x,y) a
# ese contorno, se obtiene algebraicamente:
#
#   ang_dataset = 90 - ang_csv   (modulo 180 grados, por la simetria natural
#                                 de una elipse: rotarla 180 grados de mas da
#                                 la MISMA elipse)
#
# Reducido al conjunto de angulos realmente usado {-45, 0, 45, 90}, esto da
# una permutacion CERRADA y auto-consistente dentro del mismo conjunto
# (fuerte indicio de que la formula es correcta, no un artefacto de calculo):
#   -45 -> -45   (135 grados equivale a -45 modulo 180)
#     0 ->  90
#    45 ->  45
#    90 ->   0
# Verificado ademas con un caso concreto: una elipse con ang_csv=0 (rx a lo
# largo del eje x local) tiene, tras la reflexion swap(x,y), su punto mas
# alejado sobre el eje Y del marco del dataset -- que es exactamente lo que
# da rotation_deg=90 en la parametrizacion de main_generate_simulation_
# dataset.m. NO verificado contra una medicion fisica real -- si algo se ve
# raro en la visualizacion de solapamiento del notebook, revisar esto primero.
ANGLE_CSV_TO_DATASET = {-45.0: -45.0, 0.0: 90.0, 45.0: 45.0, 90.0: 0.0}


def csv_angle_to_dataset(ang_csv_deg):
    """Ver ANGLE_CSV_TO_DATASET. Es su propia inversa."""
    key = float(ang_csv_deg)
    if key not in ANGLE_CSV_TO_DATASET:
        raise ValueError(
            f"Angulo {ang_csv_deg} no esta en el conjunto conocido {sorted(ANGLE_CSV_TO_DATASET)}")
    return ANGLE_CSV_TO_DATASET[key]


# =============================================================================
# Emparejamiento de muestras "analogas" entre dominio A y dominio B
# =============================================================================
# Para que los ejemplos usados en las visualizaciones del notebook sean
# comparables entre dominios, se busca una muestra de A y una de B que
# compartan: mismo tamano de target (Target 1/2/3), misma posicion (pos_x_mm,
# pos_y_mm) y mismo GRUPO de liquido (Hem1/Hem2/Isc1/Isc2). El grupo de
# liquido se determina a partir del TEXTO de target_eps_label, no del valor
# numerico de target_epsilon_re: la simulacion y el robot usan valores de
# permitividad ligeramente distintos para la "misma" etiqueta (p.ej. Hem 1 es
# eps=63.69 en la simulacion vs eps=52.0 en el robot), asi que comparar por
# numero daria falsos negativos.
def parse_target_size_id(target_type):
    """'Target 1 (30x20mm)' / 'Target 1 (30×20mm)' -> '1'. None si no matchea."""
    m = re.search(r"Target\s*(\d+)", str(target_type))
    return m.group(1) if m else None


def parse_eps_group(eps_label):
    """'Hem 1  (eps = 52.0)' -> 'Hem1'. 'Isc 2  (eps = 80.0)' -> 'Isc2'. None si no matchea."""
    m = re.search(r"(Hem|Isc)\s*(\d+)", str(eps_label), re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1).capitalize()}{m.group(2)}"


def _identity_position(x, y):
    """Transformacion de posicion por defecto: no hace nada."""
    return x, y


def experimental_to_synthetic_position(x, y):
    """
    [OBSOLETA desde simulation_dataset_V2.h5 -- ver conversacion] Transformacion
    de posicion descubierta EMPIRICAMENTE contra la version ORIGINAL del
    dataset sintetico (simulation_dataset.h5): comparando patrones ΔS/ΔE
    crudos -- SIN pasar por el Cycle-GAN -- entre ~1173 pares candidatos
    emparejados por tamano de target + liquido, negar AMBOS ejes de la
    posicion del dominio B (robot) antes de compararla con el dominio A daba
    ~2x mas similitud de patron que asumir que las posiciones ya estaban
    alineadas.

    Resulto ser un bug de signo en como main_generate_simulation_dataset.m
    ubicaba el target en la malla del FEM (no solo un problema de metadata:
    el campo Esc simulado en si mismo quedaba mal ubicado). Corregido en
    simulation_dataset_V2.h5 -- verificado empiricamente que con ese archivo
    la IDENTIDAD (sin transformar) es la que da mejor coincidencia (+0.049 de
    exceso vs +0.020 con esta transformacion, y encuentra match para las
    1239/1239 muestras de B en vez de 1173/1239).

    Se deja esta funcion solo por si hace falta reprocesar datos generados
    con el simulation_dataset.h5 viejo. Para el pipeline actual (V2) NO debe
    usarse -- dejar b_position_transform / query_position_transform en su
    default (identidad).
    """
    return -x, -y


def find_analogous_sample(A_meta, B_meta, pos_tol_mm=1.0, match_position=True,
                           b_position_transform=_identity_position):
    """
    Busca la primera pareja (idx_A, idx_B) cuyos metadatos coincidan en
    tamano de target + grupo de liquido (+ posicion, si match_position=True).

    b_position_transform : funcion (x_mm, y_mm) -> (x_mm, y_mm) aplicada a
        la posicion de CADA muestra de B_meta antes de compararla con
        A_meta (que se usa sin transformar). Default: identidad. Pasar
        `experimental_to_synthetic_position` para aplicar la correccion de
        signo descubierta empiricamente (ver docstring de esa funcion).

    Devuelve (idx_A, idx_B) o None si no se encuentra ninguna coincidencia.
    """
    def key_a(meta):
        size_id = parse_target_size_id(meta.get("target_type"))
        eps_group = parse_eps_group(meta.get("target_eps_label"))
        if size_id is None or eps_group is None:
            return None
        if match_position:
            px = round(meta.get("pos_x_mm", np.nan) / pos_tol_mm) * pos_tol_mm
            py = round(meta.get("pos_y_mm", np.nan) / pos_tol_mm) * pos_tol_mm
            return (size_id, eps_group, px, py)
        return (size_id, eps_group)

    def key_b(meta):
        size_id = parse_target_size_id(meta.get("target_type"))
        eps_group = parse_eps_group(meta.get("target_eps_label"))
        if size_id is None or eps_group is None:
            return None
        if match_position:
            x, y = b_position_transform(meta.get("pos_x_mm", np.nan), meta.get("pos_y_mm", np.nan))
            px = round(x / pos_tol_mm) * pos_tol_mm
            py = round(y / pos_tol_mm) * pos_tol_mm
            return (size_id, eps_group, px, py)
        return (size_id, eps_group)

    b_index = {}
    for j, m in enumerate(B_meta):
        key = key_b(m)
        if key is not None and key not in b_index:
            b_index[key] = j

    for i, m in enumerate(A_meta):
        key = key_a(m)
        if key is not None and key in b_index:
            return i, b_index[key]

    return None


def find_analog_index(query_meta, candidate_meta_list, pos_tol_mm=1.0, match_position=True,
                       query_position_transform=_identity_position):
    """
    Version "de un solo lado" de find_analogous_sample: dado UN dict de
    metadatos (p.ej. de una muestra especifica del dominio B), busca el
    primer indice en candidate_meta_list (p.ej. A_meta) cuyo tamano de
    target + grupo de liquido (+ posicion) coincidan. Util para notebooks de
    evaluacion donde ya se eligio una muestra puntual y solo hace falta su
    analoga en el otro dominio.

    query_position_transform : funcion (x_mm, y_mm) -> (x_mm, y_mm) aplicada
        a la posicion de `query_meta` antes de compararla con
        candidate_meta_list (que se usa sin transformar). Default:
        identidad. Pasar `experimental_to_synthetic_position` cuando
        `query_meta` pertenece al dominio B (experimental) y
        candidate_meta_list es A_meta.

    Devuelve el indice (int) o None si no hay coincidencia.
    """
    size_id = parse_target_size_id(query_meta.get("target_type"))
    eps_group = parse_eps_group(query_meta.get("target_eps_label"))
    if size_id is None or eps_group is None:
        return None

    qx, qy = query_position_transform(query_meta.get("pos_x_mm", np.nan),
                                       query_meta.get("pos_y_mm", np.nan))

    for j, m in enumerate(candidate_meta_list):
        if parse_target_size_id(m.get("target_type")) != size_id:
            continue
        if parse_eps_group(m.get("target_eps_label")) != eps_group:
            continue
        if match_position:
            cx, cy = m.get("pos_x_mm", np.nan), m.get("pos_y_mm", np.nan)
            if abs(qx - cx) > pos_tol_mm or abs(qy - cy) > pos_tol_mm:
                continue
        return j

    return None


# =============================================================================
# Formato comun: lista de matrices complejas (12,12) -> tensor real (N,12,12,2)
# =============================================================================
def complex_list_to_tensor(mat_list):
    """
    Convierte una lista de N matrices complejas (12,12) a un tensor real
    (N, 12, 12, 2):
        canal 0 = parte real
        canal 1 = parte imaginaria
    Este es el formato de entrada/salida de todas las redes del Cycle-GAN
    (mismo esquema que el codigo original de Jeffrey: complejo -> 2 canales).
    """
    n = len(mat_list)
    h, w = mat_list[0].shape
    tensor = np.zeros((n, h, w, 2), dtype=np.float64)
    for i, m in enumerate(mat_list):
        tensor[i, :, :, 0] = np.real(m)
        tensor[i, :, :, 1] = np.imag(m)
    return tensor


def tensor_to_complex(tensor):
    """Inverso de complex_list_to_tensor: (N,12,12,2) -> (N,12,12) complejo."""
    return tensor[..., 0] + 1j * tensor[..., 1]


# =============================================================================
# Normalizacion por dominio
# =============================================================================
# Domain A (campo E calibrado, ~1e-4 a 1e-3 V/m) y domain B (parametros S
# crudos, ~1e-3 a 1e-1, adimensional) viven en escalas MUY distintas porque
# son magnitudes fisicas distintas (no una version "corrupta" de la misma
# señal como en el caso sintetico original de Jeffrey). Sin normalizar, el
# entrenamiento del Cycle-GAN es inestable porque el termino de identidad y
# de ciclo (L1) queda dominado por el dominio de mayor magnitud.
#
# Se normaliza cada dominio de forma independiente (media/std calculados
# sobre partes real+imag juntas). Los generadores entonces trabajan en
# unidades normalizadas; `Normalizer.inverse()` permite recuperar las
# unidades fisicas originales (necesario si luego se quiere alimentar la
# salida calibrada del Cycle-GAN, GB2A(dS_experimental), al pipeline TSVD
# que espera E_cal en V/m reales).
class Normalizer:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, tensor, mask=None):
        """
        tensor : (N, H, W, 2)
        mask   : opcional, bool (H, W). Solo hace falta para tensores 12x12
                 SIN comprimir, donde las entradas fuera de la mascara A<->B
                 son siempre 0 y no deben contar en las estadisticas (pasar
                 AB_MASK en ese caso). Para tensores YA comprimidos (6x12,
                 ver compress_ab_pairs) no hace falta mascara: las 72
                 entradas son todas validas, por eso el default es None
                 (usa el tensor completo).
        """
        valid = tensor[:, mask, :] if mask is not None else tensor
        mean = np.mean(valid)
        std = np.std(valid)
        return cls(mean, std)

    def forward(self, tensor):
        return (tensor - self.mean) / self.std

    def inverse(self, tensor):
        return tensor * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def load_datasets(sim_h5_path, robot_h5_path, freq_hz=1.3e9):
    """
    [LEGACY] Atajo que carga los dominios directamente desde los .h5
    ORIGINALES (12x12, sin comprimir): simulation_dataset.h5 y
    measurements_database_robot_V2.h5. Util para depurar/reproducir el
    preprocesamiento completo en un solo paso, pero para entrenar el
    Cycle-GAN es preferible usar load_compressed_datasets() sobre los .h5
    ya preprocesados por 01_preprocesamiento.ipynb (mas rapido, no repite la
    resta de background en cada corrida).

    Devuelve
    --------
    A_tensor, B_tensor : np.ndarray float64 (N,12,12,2)  -- SIN normalizar
    A_meta, B_meta      : list[dict]
    norm_A, norm_B      : Normalizer  -- ajustados sobre los datos crudos
                           (solo dentro de la mascara AB_MASK)
    """
    A_list, A_meta = load_synthetic_domain(sim_h5_path)
    B_list, B_meta = load_experimental_domain(robot_h5_path, freq_hz=freq_hz)

    A_tensor = complex_list_to_tensor(A_list)
    B_tensor = complex_list_to_tensor(B_list)

    norm_A = Normalizer.fit(A_tensor, mask=AB_MASK)
    norm_B = Normalizer.fit(B_tensor, mask=AB_MASK)

    return A_tensor, B_tensor, A_meta, B_meta, norm_A, norm_B


def load_compressed_datasets(synthetic_h5_path, experimental_h5_path):
    """
    Atajo de conveniencia PARA ENTRENAMIENTO: carga los dos .h5 ya
    comprimidos a 6x12 (generados por 01_preprocesamiento.ipynb) y los
    devuelve en formato tensor (N,6,12,2), listos para normalizar y
    entrenar. No hace falta mascara aqui: al estar comprimidos, las 72
    entradas de cada muestra son todas fisicamente validas.

    Devuelve
    --------
    A_tensor, B_tensor : np.ndarray float64 (N,6,12,2)  -- SIN normalizar
    A_meta, B_meta      : list[dict]
    norm_A, norm_B      : Normalizer  -- ajustados sobre los datos crudos
    freq_A, freq_B       : float  -- frecuencia (Hz) guardada en cada .h5
                           (deberian coincidir, ambas ~1.3 GHz)
    """
    A_list, A_meta, freq_A = load_compressed_dataset_h5(synthetic_h5_path)
    B_list, B_meta, freq_B = load_compressed_dataset_h5(experimental_h5_path)

    A_tensor = complex_list_to_tensor(A_list)
    B_tensor = complex_list_to_tensor(B_list)

    norm_A = Normalizer.fit(A_tensor)
    norm_B = Normalizer.fit(B_tensor)

    return A_tensor, B_tensor, A_meta, B_meta, norm_A, norm_B, freq_A, freq_B


# =============================================================================
# Escritura / lectura de los datasets YA PREPROCESADOS (6x12) en HDF5 nuevos
# =============================================================================
# Estas funciones NUNCA tocan los .h5 originales (measurements_database_
# robot_V2.h5, simulation_dataset.h5): siempre crean un archivo de salida
# distinto. Se usan desde el notebook 01_preprocesamiento.ipynb para dejar
# el resultado del pipeline (diferencial + compresion 6x12) guardado en
# disco, y asi no tener que repetir la lectura/resta de background cada vez
# que se quiera entrenar o experimentar.
#
# Esquema del archivo de salida:
#   atributos raiz:
#       domain        : "synthetic" | "experimental"
#       source_path   : ruta del .h5 original del que se derivo
#       freq_hz       : frecuencia (Hz) a la que se extrajeron los datos
#   por cada muestra, grupo "sample_XXXX":
#       matrix_real, matrix_imag : [6,12] float64  (salida de compress_ab_pairs)
#       atributos: todos los campos del dict de metadatos de esa muestra
#                  (target_type, target_epsilon_re, target_eps_label,
#                  pos_x_mm, pos_y_mm, rotation_deg, [timestamp])
def save_compressed_dataset_h5(out_path, complex_6x12_list, meta_list,
                                freq_hz, domain_name, source_path):
    """
    complex_6x12_list : list[np.ndarray complex128 (6,12)]  (ya comprimidas,
                         ver compress_ab_pairs_list)
    meta_list          : list[dict], mismo largo que complex_6x12_list
    domain_name        : "synthetic" o "experimental" (solo informativo)
    source_path        : ruta del .h5 original, guardada como referencia
    """
    assert len(complex_6x12_list) == len(meta_list)
    with h5py.File(out_path, "w") as f:
        f.attrs["domain"] = domain_name
        f.attrs["source_path"] = str(source_path)
        f.attrs["freq_hz"] = float(freq_hz)
        f.attrs["n_samples"] = len(complex_6x12_list)

        for i, (mat, meta) in enumerate(zip(complex_6x12_list, meta_list)):
            g = f.create_group(f"sample_{i:04d}")
            g.create_dataset("matrix_real", data=np.real(mat))
            g.create_dataset("matrix_imag", data=np.imag(mat))
            for key, value in meta.items():
                # h5py no acepta None como atributo; se sustituye por "NA"
                g.attrs[key] = "NA" if value is None else value

    print(f"[data_io] Guardadas {len(complex_6x12_list)} muestras ({domain_name}) "
          f"en '{out_path}'")


def load_compressed_dataset_h5(path):
    """
    Lee un .h5 escrito por save_compressed_dataset_h5.

    Devuelve
    --------
    mat_list : list[np.ndarray complex128 (6,12)]
    meta     : list[dict]
    freq_hz  : float
    """
    mat_list, meta = [], []
    with h5py.File(path, "r") as f:
        freq_hz = float(f.attrs["freq_hz"])
        for key in _sorted_group_keys(f):
            g = f[key]
            mat = g["matrix_real"][()] + 1j * g["matrix_imag"][()]
            mat_list.append(mat)
            meta.append({k: _to_str(v) for k, v in g.attrs.items()})
    return mat_list, meta, freq_hz


# =============================================================================
# Metricas cuantitativas de calibracion (RMSE relativo, coseno, SSIM)
# =============================================================================
# Usadas por los scripts de evaluacion (evaluation/*.py) y de figuras.
from skimage.metrics import structural_similarity as _ssim


def relative_rmse_batch(a, b, mask=AB_MASK):
    a12 = np.stack([expand_ab_pairs(x) for x in a])
    b12 = np.stack([expand_ab_pairs(x) for x in b])
    diff = a12[:, mask] - b12[:, mask]
    num = np.sqrt(np.mean(np.abs(diff) ** 2, axis=1))
    den = np.sqrt(np.mean(np.abs(b12[:, mask]) ** 2, axis=1))
    return num / den


def cosine_sim_batch(a, b, mask=AB_MASK):
    a12 = np.stack([expand_ab_pairs(x) for x in a])
    b12 = np.stack([expand_ab_pairs(x) for x in b])
    av = np.concatenate([np.real(a12[:, mask]), np.imag(a12[:, mask])], axis=1)
    bv = np.concatenate([np.real(b12[:, mask]), np.imag(b12[:, mask])], axis=1)
    num = np.sum(av * bv, axis=1)
    den = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1) + 1e-12
    return num / den


def ssim_batch(a, b, ref, win_size=5):
    """a, b, ref : arrays complejos (N,6,12) -- matriz comprimida, sin
    expandir. `ref` define el data_range (min/max) usado para Re e Im por
    separado, por muestra (misma convencion en todos los scripts de evaluacion)."""
    n = a.shape[0]
    ssim_re = np.empty(n)
    ssim_im = np.empty(n)
    for i in range(n):
        ref_re, ref_im = np.real(ref[i]), np.imag(ref[i])
        dr_re = ref_re.max() - ref_re.min()
        dr_im = ref_im.max() - ref_im.min()
        dr_re = dr_re if dr_re > 0 else 1e-12
        dr_im = dr_im if dr_im > 0 else 1e-12
        ssim_re[i] = _ssim(np.real(a[i]), np.real(b[i]), data_range=dr_re, win_size=win_size)
        ssim_im[i] = _ssim(np.imag(a[i]), np.imag(b[i]), data_range=dr_im, win_size=win_size)
    return ssim_re, ssim_im
