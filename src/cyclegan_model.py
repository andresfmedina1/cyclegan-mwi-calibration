"""
cyclegan_model.py

Arquitectura del Cycle-GAN de calibracion, adaptada de Ian Jeffrey et al.
2023 ("Experimental Microwave Imaging System Calibration via Cycle-GAN",
IEEE TAP) y del codigo de referencia en "Bens Cycle Gan Code/Unet 1.py".

Se conserva SOLO la parte de traduccion de dominio (Cycle-GAN). La red de
inversion U-Net del paper original NO se incluye: en este proyecto la
inversion se hace por TSVD (ver OSMI---Open-Source-Microwave-Imaging-main),
por lo que el Cycle-GAN solo necesita producir dS/E calibrado, no una
imagen de permitividad.

----------------------------------------------------------------------------
Dominio A (sintetico, "limpio"): dS_A = campo diferencial calibrado del FEM
Dominio B (experimental, "corrupto"): dS_B = S-parametros diferenciales
                                        medidos con el robot
GA2B : A -> B   (limpio -> corrupto, solo se usa para consistencia de ciclo)
GB2A : B -> A   (corrupto -> limpio, ESTA es la funcion de calibracion util:
                 se le da dS_B real y devuelve una estimacion de dS_A)
----------------------------------------------------------------------------

Todos los hiperparametros de arquitectura y de perdida son argumentos de
__init__ (no estan quemados en el codigo) para poder ajustarlos facilmente
desde los scripts de training/ sin tocar este archivo.
"""

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers


# =============================================================================
# Funciones usadas dentro de capas Lambda -- DEBEN estar a nivel de modulo y
# registradas con @register_keras_serializable(), o Keras 3 no puede
# reconstruirlas al cargar un modelo guardado (.keras): una closure local
# definida dentro de build_generator() se guarda solo por nombre y falla al
# deserializar con "Could not locate function ...", incluso con
# safe_mode=False. Los argumentos extra (mask, half) se pasan via el
# parametro `arguments=` de Lambda, que si es serializable (numeros/listas).
# =============================================================================
@keras.saving.register_keras_serializable(package="cyclegan_calibration")
def _apply_ab_mask(t, mask):
    """Multiplica t (N,H,W,C) por una mascara (H,W) booleana/0-1, mask como lista anidada."""
    mask_t = tf.constant(mask, dtype=t.dtype)
    mask_t = mask_t[tf.newaxis, :, :, tf.newaxis]  # (1,H,W,1), broadcast sobre batch y canales
    return t * mask_t


@keras.saving.register_keras_serializable(package="cyclegan_calibration")
def _symmetrize_ab_ba(t, half):
    """
    Promedia las dos mitades de columnas de t (N,H,W,C): columnas [0:half] =
    bloque A->B, columnas [half:2*half] = bloque B->A (ver compress_ab_pairs
    en data_io.py). Fuerza reciprocidad exacta S[A_i->B_j] == S[B_j->A_i].
    """
    ab = t[:, :, :half, :]
    ba = t[:, :, half:, :]
    sym = (ab + ba) / 2.0
    return tf.concat([sym, sym], axis=2)


def _rms_per_sample(t):
    """RMS de cada muestra de un batch (N,H,W,C) -> (N,) -- reduce todos los ejes salvo el batch."""
    return tf.sqrt(tf.reduce_mean(tf.square(t), axis=[1, 2, 3]))


# =============================================================================
# GENERADOR: U-Net pequeño (encoder-decoder con skip connections)
# =============================================================================
# Arquitectura identica en espiritu a la usada por Jeffrey (clase
# Generator(UNET) en Unet.py): depth niveles de downsampling con
# MaxPooling2D, un nivel "cuello de botella" sin pooling, y depth-1 niveles
# de upsampling con Conv2DTranspose + concatenacion de skip connection.
#
# Parametros por defecto (dim=8, depth=3, kernel=3) reproducen la secuencia
# de filtros 8 -> 16 -> 32 -> 16 -> 8 documentada en el main4.py original
# para el caso sintetico. Para el caso experimental (mas dificil / mas
# ruidoso) Jeffrey usa mas epocas y batch mas grande, no necesariamente mas
# filtros; se deja dim como parametro por si se quiere probar una red mas
# grande.
#
# input_shape debe ser divisible por 2**(depth-1) en alto y ancho (aqui
# 12 / 2**(3-1) = 3, exacto) para que el downsampling/upsampling calce.
def build_generator(input_shape, dim=8, depth=3, kernel_size=3,
                     apply_output_mask=True, mask=None,
                     enforce_ab_ba_symmetry=False, name="generator",
                     kernel_initializer="he_uniform"):
    """
    input_shape : tuple (H, W, C) -- para este proyecto (12, 12, 2)
    dim         : numero de filtros en el primer nivel (se duplica en cada
                  nivel de downsampling: dim, 2*dim, 4*dim, ...)
    depth       : numero de niveles (incluye el cuello de botella). depth=3
                  -> 2 niveles de pooling + 1 bottleneck.
    kernel_size : tamaño de kernel de todas las convoluciones (int, cuadrado)
    apply_output_mask : si True, multiplica la salida por `mask` (mascara
                  fisica A<->B) mediante una capa no entrenable. Esto obliga
                  a la red a respetar la estructura fisica del problema
                  (las 72 entradas invalidas siempre en cero) en vez de
                  tener que aprenderla por si sola.
    mask        : np.ndarray bool (H, W), requerido si apply_output_mask=True
    enforce_ab_ba_symmetry : si True, promedia las dos mitades de columnas
                  de la salida (formato compress_ab_pairs: columnas 0:W/2 =
                  bloque A->B, columnas W/2:W = bloque B->A) mediante una
                  capa no entrenable, forzando reciprocidad exacta
                  (S[A_i->B_j] == S[B_j->A_i]) en la salida. Requiere W par.
                  Pensado para GB2A (dominio A = FEM, que es recíproco por
                  construccion -- se verifico empiricamente asimetria=0.0000
                  en los 1216 datos sinteticos); NO se recomienda para GA2B
                  (dominio B = mediciones reales, que tienen una asimetria
                  pequeña pero real de ~2-4% por diferencias de hardware
                  entre las cadenas del switch A y el switch B -- forzar
                  simetria ahi le daria al discriminador DB una señal trivial
                  para distinguir falso de real, empeorando el colapso
                  temprano de D que ya se observo en el entrenamiento).
    """
    h, w, c = input_shape
    n_pool_levels = depth - 1
    assert h % (2 ** n_pool_levels) == 0, (
        f"input_shape[0]={h} debe ser divisible por 2**(depth-1)={2**n_pool_levels}")
    assert w % (2 ** n_pool_levels) == 0, (
        f"input_shape[1]={w} debe ser divisible por 2**(depth-1)={2**n_pool_levels}")

    # HeUniform: mismo inicializador de pesos usado por Jeffrey et al. 2023
    # ("The weights of the models were initialized using the HeUniform
    # kernel initializer"), recomendado para capas con activacion ReLU.
    # Parametrizable (default he_uniform) para poder reproducir el baseline
    # original (Run 1), que usaba Glorot -- ver retrain_baseline.py.
    kernel_init = kernel_initializer

    inputs = K.Input(shape=input_shape, name=f"{name}_input")
    skips = []
    x = inputs

    # --- camino de bajada (encoder) ---
    for level in range(n_pool_levels):
        filters = dim * (2 ** level)
        x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                           kernel_initializer=kernel_init)(x)
        x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                           kernel_initializer=kernel_init)(x)
        skips.append(x)  # se guarda ANTES del pooling, para la skip connection
        x = layers.MaxPooling2D(pool_size=2)(x)

    # --- cuello de botella (sin pooling) ---
    bottleneck_filters = dim * (2 ** n_pool_levels)
    x = layers.Conv2D(bottleneck_filters, kernel_size, padding="same", activation="relu",
                       kernel_initializer=kernel_init)(x)
    x = layers.Conv2D(bottleneck_filters, kernel_size, padding="same", activation="relu",
                       kernel_initializer=kernel_init)(x)

    # --- camino de subida (decoder) ---
    for level in reversed(range(n_pool_levels)):
        filters = dim * (2 ** level)
        x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same",
                                    kernel_initializer=kernel_init)(x)
        x = layers.Concatenate()([x, skips[level]])
        x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                           kernel_initializer=kernel_init)(x)
        x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu",
                           kernel_initializer=kernel_init)(x)

    # Capa de salida: 1x1 conv lineal a `c` canales (real+imag), SIN
    # activacion acotada (tanh/sigmoid) porque los datos son valores
    # fisicos con signo y magnitud arbitraria (se normalizan aparte, ver
    # data_io.Normalizer), no imagenes en [0,1] o [-1,1].
    outputs = layers.Conv2D(c, 1, padding="same", activation="linear",
                             kernel_initializer=kernel_init)(x)

    if apply_output_mask:
        assert mask is not None, "apply_output_mask=True requiere pasar `mask`"
        outputs = layers.Lambda(
            _apply_ab_mask, arguments={"mask": mask.astype(np.float32).tolist()},
            name=f"{name}_ab_mask",
        )(outputs)

    if enforce_ab_ba_symmetry:
        assert w % 2 == 0, (
            f"enforce_ab_ba_symmetry=True requiere ancho par (formato "
            f"compress_ab_pairs: mitad A->B, mitad B->A); input_shape[1]={w}")
        half = w // 2
        outputs = layers.Lambda(
            _symmetrize_ab_ba, arguments={"half": half},
            name=f"{name}_ab_ba_symmetry",
        )(outputs)

    return K.Model(inputs=inputs, outputs=outputs, name=name)


# =============================================================================
# DISCRIMINADOR
# =============================================================================
# Clasificador convolucional simple: dim -> 2*dim -> 4*dim ... con stride 2
# (downsampling) y Leaky ReLU, terminando en Dense(1) + tanh. Mismo esquema
# que la clase Discriminator en Unet.py, mas la capa tanh final que Jeffrey
# et al. 2023 reportan como mejora de estabilidad:
#   "The appended extra tanh activation layer seemed to result in slightly
#    more stable training, allowing adversarial equilibrium for longer."
# La salida sigue usandose con MSE contra 1.0 (real) / 0.0 (fake) como en un
# LSGAN -- ambos valores caen dentro del rango de tanh, asi que el resto de
# la funcion de perdida (ver CycleGan._train_step) no cambia.
def build_discriminator(input_shape, dim=16, depth=3, kernel_size=5,
                         name="discriminator", kernel_initializer="he_uniform",
                         final_activation="tanh"):
    """
    dim         : filtros del primer nivel (se duplica hasta un maximo de
                  8*dim, igual que el original: `dim = min(dim*2, dim*8)`)
    depth       : numero de bloques con stride 2
    kernel_size : tamaño de kernel (Jeffrey usa 5x5 aqui, mas grande que en
                  el generador, para un campo receptivo mayor)
    kernel_initializer, final_activation : parametrizables (default
                  he_uniform/tanh) para poder reproducir el baseline
                  original (Run 1: Glorot + activacion lineal en D) -- ver
                  retrain_baseline.py.
    """
    # HeUniform: mismo inicializador de pesos usado por Jeffrey et al. 2023.
    kernel_init = kernel_initializer

    inputs = K.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(dim, kernel_size, strides=2, padding="same",
                       kernel_initializer=kernel_init)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    current_dim = dim
    for _ in range(depth - 1):
        current_dim = min(current_dim * 2, dim * 8)
        x = layers.Conv2D(current_dim, kernel_size, strides=2, padding="same",
                           use_bias=False, kernel_initializer=kernel_init)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    current_dim = min(current_dim * 2, dim * 8)
    x = layers.Conv2D(current_dim, kernel_size, strides=1, padding="same",
                       use_bias=False, kernel_initializer=kernel_init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(1, kernel_size, strides=1, padding="same",
                       kernel_initializer=kernel_init)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation=final_activation, kernel_initializer=kernel_init)(x)

    return K.Model(inputs=inputs, outputs=outputs, name=name)


# =============================================================================
# CYCLE-GAN
# =============================================================================
class CycleGan:
    """
    Encapsula los 4 sub-modelos (GA2B, GB2A, DA, DB) y el loop de
    entrenamiento con las 4 perdidas del Cycle-GAN:

        L_gan   = MSE(D(fake), 1)                    (adversarial, LSGAN)
        L_cycle = |A - GB2A(GA2B(A))| + |B - GA2B(GB2A(B))|   (L1)
        L_id    = |A - GB2A(A)| + |B - GA2B(B)|              (L1, identidad)
        L_scale = |RMS(GB2A(B)) - RMS(A)| + |RMS(GA2B(A)) - RMS(B)|  (ver abajo)

        L_G = L_gan + gamma_cycle*L_cycle + gamma_identity*L_id + gamma_scale*L_scale
        L_D = MSE(D(real), 1) + MSE(D(fake), 0)

    L_scale (nueva, ver conversacion): ni cycle ni identity ni la parte
    adversarial le exigen a GB2A que su salida tenga la MAGNITUD tipica del
    dominio A -- cycle solo exige reversibilidad del ciclo, identity solo
    restringe A->A y B->B (no B->A), y la adversarial resulto ser mas
    sensible al patron espacial que a la escala (evaluacion empirica: tras
    calibrar, la similitud coseno -- invariante a escala -- mejora en la
    mayoria de las muestras, pero el RMSE -- sensible a escala -- empeora en
    la mayoria). L_scale compara el RMS de cada salida generada contra el
    RMS tipico (promedio del batch) del dominio real correspondiente,
    dandole al generador una señal directa de "tu salida debe tener la
    magnitud correcta", independiente del patron espacial.

    Todos los hiperparametros son argumentos explicitos del constructor.
    """

    def __init__(self, img_shape, model_dir,
                 # --- arquitectura del generador ---
                 gen_dim=8, gen_depth=3, gen_kernel=3,
                 # --- arquitectura del discriminador ---
                 disc_dim=16, disc_depth=3, disc_kernel=5,
                 # --- pesos de las perdidas ---
                 gamma_cycle=100.0, gamma_identity=100.0, gamma_scale=0.0,
                 # gamma_supervised: peso de una perdida L1 DIRECTA entre
                 # GB2A(batch_B) y batch_A (y GA2B(batch_A) vs batch_B), SOLO
                 # tiene sentido si las muestras estan PAREADAS por indice
                 # (batch_A[i] es la contraparte fisica real de batch_B[i] --
                 # ver train(..., paired=True)). El dataset target-background
                 # es unpaired por diseño, asi que esto se deja en 0.0 por
                 # default (sin efecto, ni siquiera cambia el valor numerico
                 # de g_loss ya que se multiplica por 0) para no alterar ese
                 # entrenamiento. El dataset target-target SI quedo pareado
                 # por construccion (ver 01b_preprocesamiento_target_target.
                 # ipynb) -- ahi tiene sentido activarlo.
                 gamma_supervised=0.0,
                 # --- optimizacion ---
                 lr_generator=1e-4, lr_discriminator=1e-4,
                 # --- mascara fisica A<->B ---
                 apply_output_mask=True, ab_mask=None,
                 # --- reciprocidad A<->B forzada en la salida ---
                 enforce_symmetry_a2b=False, enforce_symmetry_b2a=False,
                 # --- inicializacion/activacion de D, parametrizables para
                 # poder reproducir el baseline original (Run 1: Glorot +
                 # activacion lineal en D) sin afectar el resto de configs,
                 # que siguen usando el default he_uniform/tanh de siempre ---
                 gen_kernel_init="he_uniform", disc_kernel_init="he_uniform",
                 disc_final_activation="tanh"):
        self.img_shape = img_shape
        self.model_dir = model_dir
        self.gamma_cycle = gamma_cycle
        self.gamma_identity = gamma_identity
        self.gamma_scale = gamma_scale
        self.gamma_supervised = gamma_supervised

        gen_kwargs = dict(dim=gen_dim, depth=gen_depth, kernel_size=gen_kernel,
                           apply_output_mask=apply_output_mask, mask=ab_mask,
                           kernel_initializer=gen_kernel_init)
        self.GA2B = build_generator(img_shape, name="GA2B",
                                     enforce_ab_ba_symmetry=enforce_symmetry_a2b,
                                     **gen_kwargs)
        self.GB2A = build_generator(img_shape, name="GB2A",
                                     enforce_ab_ba_symmetry=enforce_symmetry_b2a,
                                     **gen_kwargs)

        self.DA = build_discriminator(img_shape, dim=disc_dim, depth=disc_depth,
                                       kernel_size=disc_kernel, name="DA",
                                       kernel_initializer=disc_kernel_init,
                                       final_activation=disc_final_activation)
        self.DB = build_discriminator(img_shape, dim=disc_dim, depth=disc_depth,
                                       kernel_size=disc_kernel, name="DB",
                                       kernel_initializer=disc_kernel_init,
                                       final_activation=disc_final_activation)

        # Optimizador separado para generadores y discriminadores (practica
        # estandar en GANs: permite usar learning rates distintos si se
        # necesita estabilizar el entrenamiento).
        self.g_optimizer = K.optimizers.Adam(learning_rate=lr_generator, beta_1=0.5)
        self.d_optimizer = K.optimizers.Adam(learning_rate=lr_discriminator, beta_1=0.5)

        self.mse = K.losses.MeanSquaredError()
        self.mae = K.losses.MeanAbsoluteError()

        self.history = {"g_loss": [], "d_loss": [],
                         "g_adv": [], "g_cycle": [], "g_identity": [], "g_scale": [],
                         "g_supervised": [],
                         "val_cycle": [], "val_identity": [], "val_scale": [],
                         "val_supervised": []}

    # ------------------------------------------------------------------
    def save(self):
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        self.GA2B.save(os.path.join(self.model_dir, "GA2B.keras"))
        self.GB2A.save(os.path.join(self.model_dir, "GB2A.keras"))
        self.DA.save(os.path.join(self.model_dir, "DA.keras"))
        self.DB.save(os.path.join(self.model_dir, "DB.keras"))
        print(f"[CycleGan] Modelos guardados en {self.model_dir}")

    def load(self):
        import os
        # safe_mode=False: los generadores usan capas Lambda con funciones
        # locales (mascara A<->B, simetria A<->B) que Keras 3 no puede
        # reconstruir por nombre al deserializar (no estan registradas con
        # @keras.saving.register_keras_serializable); safe_mode=False le
        # permite reconstruirlas desde el bytecode guardado en el .keras.
        # Como estos modelos son generados por nosotros mismos (no se cargan
        # archivos de terceros no confiables), esto es seguro aqui.
        self.GA2B = K.models.load_model(os.path.join(self.model_dir, "GA2B.keras"),
                                         safe_mode=False)
        self.GB2A = K.models.load_model(os.path.join(self.model_dir, "GB2A.keras"),
                                         safe_mode=False)
        self.DA = K.models.load_model(os.path.join(self.model_dir, "DA.keras"),
                                       safe_mode=False)
        self.DB = K.models.load_model(os.path.join(self.model_dir, "DB.keras"),
                                       safe_mode=False)
        print(f"[CycleGan] Modelos cargados desde {self.model_dir}")

    # ------------------------------------------------------------------
    @tf.function
    def _train_step(self, batch_A, batch_B, supervised_weight=None):
        """
        supervised_weight : override opcional de self.gamma_supervised PARA
            ESTE PASO. Pensado para el modelo COMBINADO (target-background +
            target-target mezclados): el bloque target-background NO esta
            pareado por indice, asi que para esos batches hay que pasar
            supervised_weight=0.0 explicitamente (aunque el modelo se haya
            construido con gamma_supervised>0 para los batches target-target
            SI pareados). None (default) usa self.gamma_supervised, igual
            que antes -- no cambia nada para quien no pase este argumento.
        """
        sw = self.gamma_supervised if supervised_weight is None else supervised_weight
        with tf.GradientTape() as g_tape:
            fake_B = self.GA2B(batch_A, training=True)
            fake_A = self.GB2A(batch_B, training=True)

            cycled_A = self.GB2A(fake_B, training=True)
            cycled_B = self.GA2B(fake_A, training=True)

            same_A = self.GB2A(batch_A, training=True)  # identidad: A->A
            same_B = self.GA2B(batch_B, training=True)  # identidad: B->B

            disc_fake_B = self.DB(fake_B, training=True)
            disc_fake_A = self.DA(fake_A, training=True)

            adv_loss = (self.mse(tf.ones_like(disc_fake_B), disc_fake_B) +
                        self.mse(tf.ones_like(disc_fake_A), disc_fake_A))
            cycle_loss = self.mae(batch_A, cycled_A) + self.mae(batch_B, cycled_B)
            identity_loss = self.mae(batch_A, same_A) + self.mae(batch_B, same_B)

            # L_scale: el RMS de cada muestra generada (fake_A, fake_B) debe
            # acercarse al RMS TIPICO (promedio del batch) del dominio real
            # correspondiente. target_rms_* usa stop_gradient porque es el
            # "objetivo" (no se debe optimizar moviendo el batch real, solo
            # el generador).
            target_rms_A = tf.stop_gradient(tf.reduce_mean(_rms_per_sample(batch_A)))
            target_rms_B = tf.stop_gradient(tf.reduce_mean(_rms_per_sample(batch_B)))
            scale_loss = (tf.reduce_mean(tf.abs(_rms_per_sample(fake_A) - target_rms_A)) +
                          tf.reduce_mean(tf.abs(_rms_per_sample(fake_B) - target_rms_B)))

            # L_supervised: solo tiene sentido si batch_A[i]/batch_B[i] son
            # la MISMA muestra fisica en ambos dominios (ver train(...,
            # paired=True)) -- si no, es simplemente ruido (gamma_supervised
            # se deja en 0 en ese caso, ver __init__).
            supervised_loss = self.mae(batch_A, fake_A) + self.mae(batch_B, fake_B)

            g_loss = (adv_loss
                      + self.gamma_cycle * cycle_loss
                      + self.gamma_identity * identity_loss
                      + self.gamma_scale * scale_loss
                      + sw * supervised_loss)

        g_vars = self.GA2B.trainable_variables + self.GB2A.trainable_variables
        g_grads = g_tape.gradient(g_loss, g_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

        with tf.GradientTape() as d_tape:
            disc_real_A = self.DA(batch_A, training=True)
            disc_real_B = self.DB(batch_B, training=True)
            disc_fake_A = self.DA(fake_A, training=True)
            disc_fake_B = self.DB(fake_B, training=True)

            d_loss_A = (self.mse(tf.ones_like(disc_real_A), disc_real_A) +
                        self.mse(tf.zeros_like(disc_fake_A), disc_fake_A))
            d_loss_B = (self.mse(tf.ones_like(disc_real_B), disc_real_B) +
                        self.mse(tf.zeros_like(disc_fake_B), disc_fake_B))
            d_loss = d_loss_A + d_loss_B

        d_vars = self.DA.trainable_variables + self.DB.trainable_variables
        d_grads = d_tape.gradient(d_loss, d_vars)
        self.d_optimizer.apply_gradients(zip(d_grads, d_vars))

        return g_loss, d_loss, adv_loss, cycle_loss, identity_loss, scale_loss, supervised_loss

    # ------------------------------------------------------------------
    @tf.function
    def _eval_step(self, batch_A, batch_B):
        """
        Version "solo lectura" de _train_step: mismos forward passes de
        ciclo (A->B->A, B->A->B) e identidad (A->A, B->B), pero con
        training=False y SIN GradientTape (no se actualiza ningun peso).
        No incluye la perdida adversarial: el discriminador se sigue
        entrenando en cada epoca, asi que su "opinion" sobre el test set no
        es una senal de validacion estable (cambia aunque el generador no
        haya cambiado). Cycle + identity si son comparables epoca a epoca
        porque solo dependen de los generadores.
        """
        fake_B = self.GA2B(batch_A, training=False)
        fake_A = self.GB2A(batch_B, training=False)

        cycled_A = self.GB2A(fake_B, training=False)
        cycled_B = self.GA2B(fake_A, training=False)

        same_A = self.GB2A(batch_A, training=False)
        same_B = self.GA2B(batch_B, training=False)

        cycle_loss = self.mae(batch_A, cycled_A) + self.mae(batch_B, cycled_B)
        identity_loss = self.mae(batch_A, same_A) + self.mae(batch_B, same_B)

        target_rms_A = tf.reduce_mean(_rms_per_sample(batch_A))
        target_rms_B = tf.reduce_mean(_rms_per_sample(batch_B))
        scale_loss = (tf.reduce_mean(tf.abs(_rms_per_sample(fake_A) - target_rms_A)) +
                      tf.reduce_mean(tf.abs(_rms_per_sample(fake_B) - target_rms_B)))

        # Solo interpretable si batch_A/batch_B estan pareados (ver
        # train(..., paired=True)); si no, es ruido y simplemente se ignora
        # (gamma_supervised=0 en ese caso).
        supervised_loss = self.mae(batch_A, fake_A) + self.mae(batch_B, fake_B)

        return cycle_loss, identity_loss, scale_loss, supervised_loss

    # ------------------------------------------------------------------
    def train(self, train_A, train_B, epochs, batch_size, verbose_every=1,
              test_A=None, test_B=None, paired=False):
        """
        train_A, train_B : np.ndarray float64 (N,12,12,2) -- YA NORMALIZADOS
                            (ver data_io.Normalizer). No es necesario que
                            len(train_A) == len(train_B): el Cycle-GAN es
                            "unpaired" por diseño, cada epoca se recorre
                            min(nA, nB) // batch_size lotes emparejando
                            batches al azar de cada dominio.
        paired      : si True, usa la MISMA permutacion para indexar train_A
                      y train_B en cada batch, de forma que batch_A[i] y
                      batch_B[i] sean siempre la misma muestra fisica (ver
                      gamma_supervised en __init__). Requiere
                      len(train_A) == len(train_B). Default False preserva
                      el comportamiento unpaired original (indices
                      independientes para A y B).
        epochs      : numero de epocas. Jeffrey usa 50 para el caso
                      sintetico "limpio" y 1000 para el caso experimental
                      (datos mas ruidosos, requieren mas iteraciones).
        batch_size  : Jeffrey usa 16 para sintetico y 512 para experimental
                      (en nuestro caso los tamaños de dataset son mucho mas
                      chicos, hay que ajustar empiricamente; empezar con
                      algo pequeño, p.ej. 8-16).
        test_A, test_B : opcional, np.ndarray (N,H,W,2) YA NORMALIZADOS (el
                      subset de test separado antes del entrenamiento, nunca
                      visto por el modelo). Si se dan, AL FINAL DE CADA
                      EPOCA se calcula la perdida de ciclo + identidad
                      (ver _eval_step) sobre este conjunto completo, sin
                      gradientes. No hace falta que len(test_A)==len(test_B).
                      Es la señal de generalizacion/overfitting mas barata
                      que se puede obtener sin pares (A,B) reales del mismo
                      escenario fisico -- si `val_cycle`/`val_identity` se
                      despegan hacia arriba mientras `g_cycle`/`g_identity`
                      (entrenamiento) siguen bajando, es overfitting.
        """
        import time

        n_batches = min(len(train_A), len(train_B)) // batch_size
        assert n_batches > 0, (
            f"batch_size={batch_size} es mayor que el dataset mas chico "
            f"(A={len(train_A)}, B={len(train_B)}). Reduce batch_size.")
        if paired:
            assert len(train_A) == len(train_B), (
                f"paired=True requiere len(train_A) == len(train_B), "
                f"llego A={len(train_A)} B={len(train_B)}.")

        has_val = test_A is not None and test_B is not None
        if has_val:
            test_A_t = tf.convert_to_tensor(test_A, dtype=tf.float32)
            test_B_t = tf.convert_to_tensor(test_B, dtype=tf.float32)

        for epoch in range(epochs):
            t0 = time.time()

            idx_A = np.random.permutation(len(train_A))
            idx_B = idx_A if paired else np.random.permutation(len(train_B))

            epoch_losses = {"g_loss": [], "d_loss": [], "g_adv": [],
                             "g_cycle": [], "g_identity": [], "g_scale": [],
                             "g_supervised": []}

            for b in range(n_batches):
                a_idx = idx_A[b * batch_size:(b + 1) * batch_size]
                b_idx = idx_B[b * batch_size:(b + 1) * batch_size]
                batch_A = tf.convert_to_tensor(train_A[a_idx], dtype=tf.float32)
                batch_B = tf.convert_to_tensor(train_B[b_idx], dtype=tf.float32)

                g_loss, d_loss, adv, cyc, idn, scl, sup = self._train_step(batch_A, batch_B)
                epoch_losses["g_loss"].append(float(g_loss))
                epoch_losses["d_loss"].append(float(d_loss))
                epoch_losses["g_adv"].append(float(adv))
                epoch_losses["g_cycle"].append(float(cyc))
                epoch_losses["g_identity"].append(float(idn))
                epoch_losses["g_scale"].append(float(scl))
                epoch_losses["g_supervised"].append(float(sup))

            for k, v in epoch_losses.items():
                self.history[k].append(float(np.mean(v)))

            if has_val:
                val_cycle, val_identity, val_scale, val_supervised = self._eval_step(test_A_t, test_B_t)
                self.history["val_cycle"].append(float(val_cycle))
                self.history["val_identity"].append(float(val_identity))
                self.history["val_scale"].append(float(val_scale))
                self.history["val_supervised"].append(float(val_supervised))

            if (epoch + 1) % verbose_every == 0 or epoch == 0:
                dt = time.time() - t0
                msg = (f"[epoch {epoch+1:4d}/{epochs}] "
                       f"G={self.history['g_loss'][-1]:.4f} "
                       f"(adv={self.history['g_adv'][-1]:.4f}, "
                       f"cyc={self.history['g_cycle'][-1]:.4f}, "
                       f"id={self.history['g_identity'][-1]:.4f}, "
                       f"scl={self.history['g_scale'][-1]:.4f}, "
                       f"sup={self.history['g_supervised'][-1]:.4f}) "
                       f"D={self.history['d_loss'][-1]:.4f}")
                if has_val:
                    msg += (f"  | val_cyc={self.history['val_cycle'][-1]:.4f} "
                            f"val_id={self.history['val_identity'][-1]:.4f} "
                            f"val_scl={self.history['val_scale'][-1]:.4f} "
                            f"val_sup={self.history['val_supervised'][-1]:.4f}")
                msg += f"  [{dt:.1f}s]"
                print(msg)

    # ------------------------------------------------------------------
    def calibrate(self, dS_experimental_2ch):
        """
        Funcion de calibracion util para produccion: aplica GB2A a un batch
        de mediciones experimentales YA NORMALIZADAS con el Normalizer del
        dominio B, devuelve la estimacion (normalizada) del dominio A.
        Recuerda des-normalizar con Normalizer_A.inverse(...) antes de usar
        el resultado en el pipeline TSVD.
        """
        return self.GB2A(dS_experimental_2ch, training=False).numpy()
