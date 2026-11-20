# KAN-POD-Attention-LSTM surrogate for BS 8414-1 façade fire test simulations

Code, trained weights and a Streamlit app for the surrogate model in:

> S. P. K. Balabomma, B. Matuszewski, E. Asimakopoulou. *A Deep-Learning-Based
> Surrogate Model for BS 8414 Façade Fire Test Simulations: A Proof-of-Concept
> Study.* Manuscript for *Engineering with Computers*.

The model predicts the thermocouple histories that an FDS simulation of a
BS 8414-1 rig would produce for a given façade build-up, without running the
simulation.

---

## What it predicts

| | |
|---|---|
| **Input** | 16 values describing the build-up: cladding system, insulation product and geometry variant (as learned embeddings), plus 13 material properties read from the FDS input file |
| **Output** | 16 external thermocouples (Levels 1 and 2, main face and return wing) and the 5-channel heat-release budget |
| **Time grid** | 0 to 1800 s at 10 s, 181 steps |
| **Fixed** | fire source (3.5 MW prescribed burner), mesh (0.10 m), ambient 18 °C |

Geometry variants are the eight combinations of three construction switches:
`noair` (ventilated cavity removed), `nogap` (open panel joints closed) and
`nocb` (cavity barriers removed).

## Corpus and split

199 FDS simulations covering 41 build-ups (11 cladding systems × 5 insulation
products, up to eight geometry variants each), split deterministically by hash:

| Partition | Simulations | Role |
|---|---|---|
| Training | 152 | fits the weights |
| Validation | 23 | never fitted; drives the learning-rate schedule and checkpoint selection |
| Test | 24 | never fitted and never used for any decision; final independent measure |

The FDS corpus itself is not included in this repository.

## Model

```
params ─ embeddings + 13 material ─ KAN encoder ─┐
time   ─ sinusoidal encoding ────────────────────┴─ input_proj ─ MultiScaleConv ─ BiLSTM(96)×2 ─ self-attention ─ features (B,181,192)

features ─ KAN decoder + param skip ─ a_tc(t)  (B,181,8) ─ Φ_tc  (16×8) + T̄ ─ 16 thermocouples ─ ambient clamp
         ─ KAN decoder + param skip ─ a_hrr(t) (B,181,4) ─ Φ_hrr ( 5×4) + H̄ ─  5 HRR channels  ─ HRR ≥ 0 clamp
```

- **Kolmogorov–Arnold (KAN) layers** with learnable B-spline edge functions
  condition the network on the build-up.
- **Attention-LSTM backbone**: multi-scale convolution, a two-layer
  bidirectional LSTM and self-attention over the time steps.
- **Proper orthogonal decomposition (POD) output**: instead of decoding every
  channel directly, the network predicts POD coefficients at each time step,
  and fixed sensor-wise modes fitted on the training data rebuild the channels,
  `q ≈ q̄ + Φ a(t)`. Rank 8 for the thermocouples, rank 4 for the heat-release
  budget.
- **Physical clamps** on the output: no temperature below ambient, no negative
  heat-release rate.
- **792,305 trainable parameters.** Each POD basis is stored inside its
  checkpoint, so a checkpoint always reconstructs with the basis it was
  trained with.

## Results

Regression metrics over all 16 thermocouples, every time step and every case
(Table 8 of the paper; mean ± SD over three seed bases):

| Metric | Training (152) | Validation (23) | Test (24) |
|---|---|---|---|
| R² | 0.955 ± 0.003 | 0.926 ± 0.005 | 0.927 ± 0.005 |
| MAE (°C) | 19.6 ± 0.8 | 24.5 ± 0.7 | 23.3 ± 0.9 |
| RMSE (°C) | 31.6 ± 1.1 | 42.1 ± 1.3 | 38.8 ± 1.3 |
| MAPE (%) | 10.5 ± 0.5 | 12.0 ± 0.5 | 12.2 ± 0.6 |
| MSRE | 0.032 ± 0.003 | 0.036 ± 0.001 | 0.036 ± 0.003 |

Across all five seed bases the test R² is 0.927 ± 0.005.

## Trained weights

Five seed bases, each a three-member ensemble whose members are seeded
`s`, `s+1`, `s+2`. A prediction is the mean of the three members.

| Seed base | Folder | Test R² |
|---|---|---|
| 42 | `models_part1_podattn_hash_s42_r1/` | 0.9234 |
| 45 | `models_part1_podattn_hash_s45_r1/` | 0.9326 |
| 48 | `models_part1_podattn_hash_s48_r1/` | 0.9257 |
| 51 | `models_r9/kanpod_hash_s51/` | 0.9315 |
| 54 | `models_r9/kanpod_hash_s54/` | 0.9232 |

Seed bases 42, 45 and 48 give Table 8. Seed base 48 is the one plotted in the
paper's figures. Each folder holds `part1_member{0,1,2}.pt` and the run's
recorded scores.

## Streamlit app

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements_lock_from_parent.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\run_app.ps1                    # http://localhost:8501
```

Choose a cladding, an insulation product and the geometry switches, and the
app predicts all 16 thermocouples and the heat-release budget in under a
second. It also gives the surrogate's reading of the BR 135 external
fire-spread criterion and exports the predicted histories as CSV.

- It serves the five seed bases above and nothing else. Each checkpoint is
  checked against the paper model (architecture, split, parameter count, POD
  rank) before it is loaded.
- Seed base 48 is the default, so the app reproduces the paper's figures.
- A build-up the corpus never contained is flagged as an extrapolation.
- The material table it reads, `app_assets/part1_materials.json`, describes
  the 199 simulations and is written by `export_app_assets_paper.py`.

The app runs on CPU if no CUDA device is available. Python 3.12,
torch 2.5.1 (CUDA 12.1 build), Streamlit 1.56.

`test_app_paper.py` checks the app against the paper. It needs the FDS corpus,
so it only runs where the corpus is available. Its checks confirm that each
seed base's test R² is reproduced through the app's own prediction path, and
that the input vector the app builds equals the training input for all 199
simulations.

## Reproducing training and evaluation

These need the FDS corpus. Point `PART1_SIMS_DIR` at it and
`PART1_CHID_ALLOWLIST` at a file listing the 199 simulation IDs, one per line,
so that later simulations in the same folder are not picked up.

```bash
export PART1_SPLIT=hash
export PART1_SIMS_DIR=/path/to/corpus
export PART1_CHID_ALLOWLIST=/path/to/frozen_199_chids.txt

python fit_pod_part1.py --kind spatial          # POD basis from the training frames
python train_part1.py --split hash --members 3 --seed 42 --model-dir models_part1_podattn_hash_s42_r1
python evaluate_part1.py --model-dir models_part1_podattn_hash_s42_r1
```

Training uses the GPU (`DEVICE = "cuda"` in `config_part1.py`).
`evaluate_part1.py` writes `evaluation_part1.json`, which holds the metrics
for each partition.

## Files

| File | Role |
|---|---|
| `model_part1.py` | the KAN-POD-Attention-LSTM (`Part1Surrogate`) |
| `layers_part1.py` | KAN layers, time encoding, multi-scale convolution |
| `pod_rom_part1.py` | POD basis and the reconstruction head with the physical clamps |
| `fit_pod_part1.py` | fits the POD basis on the training frames |
| `config_part1.py` | inputs, sensors, time grid, hyperparameters |
| `data_loader_part1.py` | reads FDS input files and outputs, builds the 16 inputs, hash split |
| `physics_part1.py` | physics constraints applied during training, each measured in the FDS data first |
| `train_part1.py` | trains an ensemble |
| `evaluate_part1.py` | scores an ensemble on all three partitions |
| `app_part1.py`, `app_common_part1.py`, `run_app.ps1` | the Streamlit app and its launcher |
| `export_app_assets_paper.py` | writes the app's material table from the 199 simulations |
| `test_app_paper.py` | checks the app against the paper's results |
| `pod_spatial_basis_part1_hash.pt` | POD basis the model is built with |
