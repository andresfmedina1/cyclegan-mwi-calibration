# Cycle-GAN calibration for microwave brain-stroke imaging

Calibration of a 12-antenna, 1.3 GHz microwave imaging system with a Cycle-GAN.
The network learns to translate raw **experimental** differential S-parameters
(measured with an XY robot, a VNA and a head phantom) into the **synthetic**
FEM field that a downstream TSVD reconstruction expects, without needing a
known target inside the chamber. Code and data of a POLITO Master's thesis.

Two imaging configurations are covered:

| Configuration | Differential signal | Samples |
|---|---|---|
| **Detection** | target − background (healthy) | 1,216 synthetic / 1,221 experimental, unpaired |
| **Monitoring** | target − target (stroke evolution) | 41,096 synthetic / 41,096 experimental, paired by index |

## Results

Mean over the **test split** of the 80/10/10 train/val/test protocol, comparing
the calibrated signal against the synthetic reference. Columns are
raw (no calibration) / traditional calibration / Cycle-GAN. Bold marks the better
of the two calibration methods.

| Configuration | Test set | Cosine similarity ↑ | Relative RMSE ↓ |
|---|---|---|---|
| Detection (final config) | 122 random | 0.068 / 0.112 / **0.160** | 1.311 / 2.794 / **1.692** |
| Monitoring, unsupervised | 4,110 random | 0.026 / **0.152** / 0.117 | 1.094 / **1.658** / 1.719 |
| Monitoring, supervised | 3,082 random | 0.026 / 0.189 / **0.696** | 1.110 / 1.727 / **0.738** |
| Monitoring, supervised | Isc2 holdout (10,274) | 0.041 / 0.019 / **0.243** | 1.037 / 1.472 / **1.016** |

- The supervised monitoring model (extra L1 loss, possible because the monitoring
  data are index-paired) is the only one whose RMSE falls below the raw signal,
  and it beats traditional calibration on all four metrics (RMSE, cosine, SSIM Re/Im).
- **Isc2** (pure alcohol, εr ≈ 15.1) is a whole liquid class excluded from
  training, validation and normalisation statistics. It measures generalisation to
  an unseen material and is clearly harder than the in-distribution test.
- The unsupervised configurations are mixed: Cycle-GAN is better in detection
  (cosine, RMSE), while traditional calibration is better in unsupervised
  monitoring (cosine, RMSE and SSIM).
- SSIM Re/Im, the share of samples that improve and the train-split numbers are in
  `results/eval_traditional_calibration_8010_results.json` and
  `results/eval_tt_sup_indist_test_8010_results.json`.

## Repository layout

```
data/
  detection/    synthetic_dataset_6x12.h5, experimental_dataset_6x12.h5
  monitoring/   synthetic_dataset_6x12_targettarget.h5, experimental_dataset_6x12_targettarget.h5
  raw/          measurements_database_robot_V2.h5, simulation_dataset_V2.h5   (inputs of the notebooks)
data_generation/
  experimental_robot/   MATLAB + Arduino code for the robot/VNA acquisition
  synthetic_fem/        subset of OSMI + main_generate_simulation_dataset.m (FEM)
src/            cyclegan_model.py, data_io.py (loading, metrics, Normalizer), splits.py
training/       retrain_*.py (one script per run), retrain_common.py (shared runner), extend_ttunsup_8010.py
evaluation/     eval_*.py: RMSE, cosine, SSIM against raw and traditional calibration
figures/        gen_*.py (loss curves and calibration examples), draw_*.py (architecture diagrams)
notebooks/      01_preprocesamiento*.ipynb (raw -> 6x12 datasets)
results/        eval_*_results.json (all computed on the 80/10/10 split)
```

## Data

Each `.h5` holds one group per sample, `sample_XXXX`, with `matrix_real` and
`matrix_imag` (6×12, float64) plus metadata attributes (target type, position,
rotation, permittivity label). The root attributes give the domain and the
frequency (1.3 GHz).

**The 6×12 format.** A 12×12 S-matrix has only 72 measurable entries: the two
SP8T switches (antennas A1–A6 and B1–B6) cannot connect A–A or B–B. `compress_ab_pairs`
in `src/data_io.py` packs them into a 6×12 matrix. Row *i* is antenna A*i*,
columns 0–5 are A*i*→B*j* and columns 6–11 are B*j*→A*i*. The networks take this
as a 6×12×2 tensor (real, imaginary).

| File | Domain | N | Notes |
|---|---|---|---|
| `synthetic_dataset_6x12.h5` | synthetic, detection | 1,216 | 3 target sizes × positions × 4 rotations × 4 permittivities |
| `experimental_dataset_6x12.h5` | experimental, detection | 1,221 | ΔS = S_target − previous healthy measurement |
| `synthetic_dataset_6x12_targettarget.h5` | synthetic, monitoring | 41,096 | sample *i* pairs with sample *i* of the experimental file |
| `experimental_dataset_6x12_targettarget.h5` | experimental, monitoring | 41,096 | attributes include the two source acquisitions `acq_num_A/B` |
| `raw/measurements_database_robot_V2.h5` | experimental, raw | 1,253 acquisitions | 12×12×11 frequencies, complex |
| `raw/simulation_dataset_V2.h5` | synthetic, raw | 1,216 FEM runs | 12×12, 1.3 GHz |

```
robot + VNA acquisition -> raw/measurements_database_robot_V2.h5 --\
  (data_generation/experimental_robot)                              >-- notebooks/01*.ipynb --> data/{detection,monitoring}/*.h5
FEM simulation          -> raw/simulation_dataset_V2.h5          --/
  (data_generation/synthetic_fem)
```

## Method

Four networks: generators `GA2B` (synthetic → experimental) and `GB2A`
(experimental → synthetic, the calibration function) and discriminators `DA`, `DB`.
Generator loss:

```
L_G = L_adv + γ_cyc·L_cycle + γ_id·L_identity + γ_scale·L_scale + γ_sup·L_supervised
```

- `L_adv`: least-squares GAN. `L_cycle`, `L_identity`: L1.
- `L_scale`: matches the RMS magnitude of the generated output to the target domain.
  Cycle, identity and adversarial terms fix the spatial pattern but not the gain.
- `L_supervised`: L1 between the generated and the true paired sample. Only used
  for the monitoring data, whose batches are index-aligned.
- Optional reciprocity layer on `GB2A` (`enforce_symmetry_b2a`) that forces
  S[A*i*→B*j*] = S[B*j*→A*i*], as the FEM data are exactly reciprocal.

| Run | Data | γ cyc / id / scale / sup | lr G / D | Batch | Epochs |
|---|---|---|---|---|---|
| Baseline (`retrain_baseline.py`) | detection | 100 / 100 / 0 / 0 | 1e-4 / 1e-4 | 8 | 100 |
| LR-only (`retrain_lronly.py`) | detection | 100 / 100 / 0 / 0 | 1e-4 / 2e-5 | 8 | 100 |
| Final detection (`retrain_run8.py`) | detection | 20 / 20 / 3 / 0 | 1e-4 / 2e-5 | 8 | 100 |
| Monitoring, unsupervised (`retrain_tt_unsupervised.py`) | monitoring | 20 / 20 / 3 / 0 | 1e-4 / 1e-5 | 32 | 20 |
| Monitoring, supervised (`retrain_tt_supervised.py`) | monitoring | 20 / 20 / 3 / 20 | 1e-4 / 1e-5 | 32 | 200 |

The baseline collapses: the discriminator loss falls to ~0 within the first ~20
epochs. Lowering the discriminator learning rate (LR-only) only delays this. The
final detection configuration (γ = 20, scale loss) reaches a stable discriminator
loss (~0.13). Unsupervised monitoring still collapses later, which is why its
epoch-20 checkpoint is the promoted one, and the supervised run does not collapse.
The baseline and LR-only runs also use Glorot initialisation and a linear
discriminator output; the rest use He-uniform and tanh.

**Split** (`src/splits.py`, scheme `8010_3way`, seed 42). 80/10/10 train/val/test.
The validation set drives the per-epoch curves and the test set is only used for
the final metrics, so the two never share samples. In detection, domains A and B
are split independently. In monitoring, one permutation is shared so that pairs
stay together. The input normaliser (scalar z-score) is fitted on the training
subset only. The supervised monitoring run removes Isc2 from the pool before
splitting.

## Quick start

Tested with Python 3.11.15, TensorFlow 2.21, Keras 3.15, NumPy 2.4, SciPy 1.17,
h5py 3.14, scikit-image 0.26, Matplotlib 3.11 and openpyxl 3.1, on CPU
(Apple M2; no GPU is needed).

```bash
pip install tensorflow numpy scipy h5py scikit-image matplotlib openpyxl
```

The scripts import each other as flat modules and read the data and checkpoints
relative to the current directory. Run them from a scratch directory with the
datasets linked in:

```bash
mkdir run && cd run
ln -s ../data/detection/*.h5 ../data/monitoring/*.h5 .
export PYTHONPATH=$PWD/../src:$PWD/../training:$PWD/../evaluation

python ../training/retrain_run8.py              # detection, final configuration
python ../evaluation/eval_train_test_full.py    # RMSE / cosine / SSIM, train and test
```

Each run writes `checkpoints/<Run>_8010_3way/` with the four `.keras` models,
`config.json` (hyperparameters, normaliser statistics, split sizes),
`split_indices.npz`, `history.json` and `training_history.{jpg,pdf}`. The
evaluation scripts expect the names `{Baseline,Run8,TTUnsup,TTSup}_8010_3way`
and skip any that are missing.

From this layout I ran a one-epoch Run8 training followed by the evaluation
function of `eval_train_test_full.py`. The remaining scripts follow the same
conventions but have not been re-run here.

Approximate CPU time per epoch: 2–3 s (detection), 30–40 s (unsupervised
monitoring), 23–28 s (supervised monitoring). The supervised monitoring run
(200 epochs) takes about two hours.

`extend_ttunsup_8010.py` continues the unsupervised monitoring run from epoch 20
to 100, only to plot the late discriminator collapse. It **overwrites the epoch-20
weights in place** (a copy is saved in `..._ep20`). Restore them before evaluating:
the promoted model is the epoch-20 one.

## Regenerating the raw data

**Experimental (`data_generation/experimental_robot/`).** `Robot_Acquisition/step1…step3`
calibrate, test and home the XY robot (Arduino firmware in `arduino/`).
`step4_acquisition_gui.m` runs the automatic acquisition: for each valid position
it measures the 12×12 matrix with the VNA and the two switches. In this copy it
writes `measurements_database_robot_V2.h5` (11 frequency points, 12 averages).
`step4_acquisition.m` is the command-line variant and still points at a test
database. The scripts expect `meas2x2_trigger_MGDR.m`, `setVNA_v3_2ports.m`,
`valid_positions.mat` and the `Meas_Switch_12x12/` output folder one level above
`Robot_Acquisition/`, so that layout is kept.

**Synthetic (`data_generation/synthetic_fem/`).** Run
`Examples/Head_imaging/main_generate_simulation_dataset.m` in MATLAB. It solves
1 background and 1,216 target FEM scenarios at 1.3 GHz with
`Forward_solvers/lib/fem2d.p` and writes `results/simulation_dataset_V2.h5`. The
relative layout of the OSMI repository is kept so its `addpath` calls resolve.

`notebooks/01_preprocesamiento.ipynb` and `01b_preprocesamiento_target_target.ipynb`
turn the two raw databases into the four 6×12 datasets.

## Known limitations

- **Hard-coded paths.** These files point to folders of the original machine and
  need editing: `notebooks/*.ipynb`, `evaluation/eval_traditional_calibration_v2.py`
  (imported by `eval_traditional_calibration_8010.py`) and
  `figures/draw_*_architecture.py`. Point the raw databases to `data/raw/`.
- **Not included.**
  - `Ep_healthy.mat`, the simulated healthy-background field used by the traditional
    calibration. The script that writes it is not in the repository.
  - `positions/target_intersections.csv`, needed by notebook 01b to build the monitoring pairs.
  - The Mini-Circuits switch library (`C:\Image_to_Image\mcl_SolidStateSwitch_NET45.dll`
    in the acquisition scripts): vendor software, lab Windows PC.
  - The trained checkpoints and the run logs (see `.gitignore`).
- **Lab-specific files.** `Robot_Acquisition/config/robot_config.mat` is the `step1`
  output for the lab setup (serial port, steps per mm) and has to be regenerated
  on another machine. `valid_positions.mat` is the copy that sat next to the
  acquisition scripts; a different, newer one exists elsewhere in the original folders.
- **File sizes.** The two monitoring datasets (159 and 164 MB) exceed GitHub's
  100 MB per-file limit, and the raw robot database (68 MB) triggers its 50 MB
  warning. Use Git LFS (`git lfs track "*.h5"`) or host the data externally.
