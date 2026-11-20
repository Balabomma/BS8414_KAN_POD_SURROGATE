"""POD reduced basis for the Part1 thermocouple + HRR targets — the shared ROM layer.

Shared, byte-identical, by `bs8414_KAN_pod_surrogate` and `bs8414_MLP_pod_surrogate`
(checked by `verify_parity_pod.py`). Their `model_part1.py` files differ only in
the coefficient regressor (KANLinear vs MLPLinear blocks); everything that
defines the reduced space lives here.

THE REDUCED-ORDER MODEL
-----------------------
A non-intrusive parametric POD ROM (POD-NN; Hesthaven & Ubbiali, JCP 2018). One
snapshot is one simulation's whole standardised trajectory,

    y(mu) = vec(Y(t, channel; mu))         16 TC x 181 steps = 2896  (TC)
                                           5 HRR x 181 steps = 905   (HRR)

and the ROM is

    y(mu) ~ ybar + V_r^T (s_r * c_hat(mu)),      r << number of training cases

with V_r the leading r right-singular vectors of the centred train snapshot
matrix, s_r the train standard deviation of each coefficient, and c_hat a
network (or a GP) that maps the build-up mu to unit-variance coefficients.

Why intrusive POD-Galerkin is not an option here: a Galerkin ROM projects the
full-order operators onto the modes (Mendez, Tutorial 3), and FDS does not
expose its discrete operators. The CYPHER dynamic-ROM challenge hit the same
wall with a CFD solver and identified the reduced operators from data; here
the burner forcing is identical in every deck, so there is no input signal to
identify a forced reduced system against. What varies between runs is the
build-up, which makes this a parametric ROM rather than a dynamic one — the
map mu -> c is what has to be learned.

THE INNER PRODUCT
-----------------
Mendez, Tutorial 3: the POD must be orthonormal in the inner product the error
is measured in, or truncation is optimal for the wrong norm. The frozen Part1
training loss (`train_part1.masked_mse`) is an unweighted mean over timesteps and
channels in STANDARDISED space, so the snapshot matrix is built in that space
with identity weights. The basis is therefore L2-optimal for the loss the
networks are trained on, and a network trained through the reconstruction
(`PODReconstruction`) automatically weights each coefficient by its energy.

LEAKAGE AND CONSISTENCY GUARDS
------------------------------
  * The basis is fitted on TRAIN cases only (`fit_pod_part1.py`), in the space
    of scalers that `prepare_data_splits` fits on train rows only.
  * Cases with an incomplete time grid are excluded from the fit — a padded tail
    would put a held last value into the modes. They still train (masked loss).
  * The scalers used for the fit are stored with the basis, and
    `PODReconstruction.set_output_scaling` refuses to train against different
    ones. That catches a basis fitted on another split or an older corpus.
  * The basis is registered as buffers, so every checkpoint carries the exact
    basis it was trained with; re-fitting later cannot change an old result.
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

from config_part1 import (N_SENSORS, N_HRR_CHANNELS, N_TIMESTEPS, SPLIT_MODE,
                          T_AMBIENT)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Truncation. Chosen from the projection floor measured on 2026-09-14 by
# fit_pod_part1.py with the frozen evaluator's masked_r2 (hash split, 199 cases,
# test): TC r=16 R2 0.971, r=32 0.977, r=64 0.979, r=149 (all) 0.980; HRR r=16
# 0.992. r=32 / 16 sit where the floor has flattened, so the ROM's ceiling is not
# what limits it. Env-overridable for a truncation sweep; a checkpoint is tied
# to the r it was trained with (buffer shapes).
R_TC = int(os.environ.get("POD_R_TC", "32"))
R_HRR = int(os.environ.get("POD_R_HRR", "16"))

FORMAT_VERSION = 1


def basis_path(split=None):
    """Basis file for a split. One per split: the train rows differ."""
    return os.environ.get(
        "PART1_POD_BASIS",
        os.path.join(PROJECT_DIR, f"pod_basis_part1_{split or SPLIT_MODE}.pt"))


def array_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def complete_cases(mask, indices):
    """Indices whose whole time grid was reported."""
    return [int(i) for i in indices if bool(np.all(mask[i] > 0))]


def fit_pod(Y, keep):
    """Space-time POD of Y[keep].

    Y is (N, T, C) standardised; keep lists complete training cases. Returns the
    mean snapshot, the modes as ROWS (rank, T*C), singular values, and the train
    standard deviation of each coefficient. The SVD runs in float64: 151 x 2896
    is small, and double precision keeps the trailing modes orthonormal.
    """
    X = Y[keep].reshape(len(keep), -1).astype(np.float64)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)

    # Centring removes one rank, so the last singular value is numerically zero
    # and its vector is noise. Keep only directions the data actually spans.
    rank = int((S > S[0] * 1e-10).sum())
    S, Vt = S[:rank], Vt[:rank]

    # Singular vectors are defined up to sign; fix it (largest-magnitude entry
    # positive) so two fits of the same data are bitwise comparable.
    pivot = np.abs(Vt).argmax(axis=1)
    Vt = Vt * np.sign(Vt[np.arange(rank), pivot])[:, None]

    coef = (X - mean) @ Vt.T
    coef_std = coef.std(axis=0)
    coef_std[coef_std < 1e-12] = 1.0
    return {"mean": mean, "modes": Vt, "sigma": S, "coef_std": coef_std,
            "shape": tuple(Y.shape[1:])}


def modes_for_energy(sigma, levels=(0.9, 0.99, 0.999)):
    """Mendez, Tutorial 3: modes needed to reach each energy fraction."""
    energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    return {str(l): int(np.searchsorted(energy, l)) + 1 for l in levels}


def project(Y, mask, basis, r):
    """Coefficients (unit-variance units) of every case in Y, shape (N, r).

    Unreported steps are filled with the basis mean so they contribute nothing
    to the projection. For a complete case this is the exact orthogonal
    projection; for a truncated one it is the projection of the reported part.
    """
    mean = basis["mean"].reshape(basis["shape"])
    filled = np.where(mask[:, :, None] > 0, Y, mean[None])
    X = filled.reshape(len(Y), -1).astype(np.float64) - basis["mean"]
    return (X @ basis["modes"][:r].T) / basis["coef_std"][:r]


def reconstruct(coef_hat, basis, r):
    """Unit-variance coefficients (N, r) -> standardised trajectories (N, T, C)."""
    c = np.asarray(coef_hat, dtype=np.float64) * basis["coef_std"][:r]
    X = basis["mean"] + c @ basis["modes"][:r]
    return X.reshape((len(c),) + basis["shape"])


def projection_residual_var(Y, keep, basis, r):
    """Per-(t, channel) train variance NOT captured by r modes.

    Used to widen predictive intervals by the truncation error, which a
    coefficient-space model cannot see.
    """
    full = np.ones(Y[keep].shape[:2])
    rec = reconstruct(project(Y[keep], full, basis, r), basis, r)
    return ((Y[keep] - rec) ** 2).mean(axis=0)


def load_basis(path=None, split=None):
    path = path or basis_path(split)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"POD basis not found: {path}\n"
            f"  fit it first:  python fit_pod_part1.py --split {split or SPLIT_MODE}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"{path}: basis format {blob.get('format_version')} "
                           f"!= {FORMAT_VERSION}; re-fit it")
    return blob


class PODReconstruction(nn.Module):
    """SPACE-TIME variant (archived POD-NN, `model_part1_podnn.py`).

    Coefficients -> standardised TC and HRR trajectories, with physics floors.

    Holds the basis as buffers (so it is saved in every checkpoint) and applies
    the SAME hard output clamps as the parent Attention-LSTM models: ambient on
    the thermocouples and zero on total HRR, both expressed in standardised
    units once the scalers are known.
    """

    kind = "spacetime"

    def __init__(self, r_tc=R_TC, r_hrr=R_HRR, path=None):
        super().__init__()
        blob = load_basis(path)
        self.r_tc, self.r_hrr = r_tc, r_hrr
        for name, r, n_ch in (("tc", r_tc, N_SENSORS), ("hrr", r_hrr, N_HRR_CHANNELS)):
            b = blob[name]
            if tuple(b["shape"]) != (N_TIMESTEPS, n_ch):
                raise RuntimeError(f"{name} basis shape {b['shape']} does not match "
                                   f"({N_TIMESTEPS}, {n_ch})")
            if r > len(b["sigma"]):
                raise RuntimeError(f"{name}: r={r} exceeds the basis rank "
                                   f"{len(b['sigma'])}")
            self.register_buffer(f"{name}_mean",
                                 torch.as_tensor(b["mean"], dtype=torch.float32))
            self.register_buffer(f"{name}_modes",
                                 torch.as_tensor(b["modes"][:r], dtype=torch.float32))
            self.register_buffer(f"{name}_coef_std",
                                 torch.as_tensor(b["coef_std"][:r], dtype=torch.float32))

        self._fit_scalers = {"tc": blob["tc_scaler"], "hrr": blob["hrr_scaler"]}
        self.basis_digest = blob["corpus_digest"]

        self.register_buffer("ambient_scaled", torch.full((N_SENSORS,), -1e9))
        self.register_buffer("hrr_floor_scaled", torch.full((N_HRR_CHANNELS,), -1e9))

    def set_output_scaling(self, tc_scaler, hrr_scaler, hrr_nonnegative_idx=(0,),
                           t_ambient=T_AMBIENT):
        for name, scaler in (("tc", tc_scaler), ("hrr", hrr_scaler)):
            ref = self._fit_scalers[name]
            same = (np.allclose(np.asarray(scaler.mean), np.asarray(ref["mean"]),
                                rtol=0, atol=1e-5)
                    and np.allclose(np.asarray(scaler.scale), np.asarray(ref["scale"]),
                                    rtol=0, atol=1e-5))
            if not same:
                raise RuntimeError(
                    f"the {name} scaler differs from the one the POD basis was fitted "
                    f"with — the basis belongs to another split or an older corpus. "
                    f"Re-run fit_pod_part1.py for this split before training.")

        amb = (t_ambient - torch.as_tensor(tc_scaler.mean, dtype=torch.float32)) \
            / torch.as_tensor(tc_scaler.scale, dtype=torch.float32)
        self.ambient_scaled.copy_(amb.to(self.ambient_scaled.device))

        floor = torch.full((N_HRR_CHANNELS,), -1e9)
        mean = torch.as_tensor(hrr_scaler.mean, dtype=torch.float32)
        scale = torch.as_tensor(hrr_scaler.scale, dtype=torch.float32)
        for i in hrr_nonnegative_idx:
            floor[i] = (0.0 - mean[i]) / scale[i]
        self.hrr_floor_scaled.copy_(floor.to(self.hrr_floor_scaled.device))

    def expand(self, name, coef_hat):
        """(B, r) unit-variance coefficients -> (B, T, C) standardised, unclamped."""
        mean = getattr(self, f"{name}_mean")
        modes = getattr(self, f"{name}_modes")
        std = getattr(self, f"{name}_coef_std")
        n_ch = N_SENSORS if name == "tc" else N_HRR_CHANNELS
        flat = mean + (coef_hat * std) @ modes
        return flat.view(coef_hat.shape[0], N_TIMESTEPS, n_ch)

    def forward(self, tc_coef_hat, hrr_coef_hat):
        tc = torch.maximum(self.expand("tc", tc_coef_hat), self.ambient_scaled)
        hrr = torch.maximum(self.expand("hrr", hrr_coef_hat), self.hrr_floor_scaled)
        return tc, hrr


# ══════════════════════════════════════════════════════════════════════════
#  SPATIAL (sensor-wise) POD — the head of the POD-Attention-LSTM
# ══════════════════════════════════════════════════════════════════════════
# The q ~ qbar + Phi a(t) form of the CYPHER POD-Galerkin submission, taken
# across the sensor axis: fixed modes over the 16 thermocouples (and over the 5
# HRR channels), with the coefficients a(t) produced at EVERY timestep by the
# parent's unchanged Attention-LSTM backbone — a POD-LSTM reduced-order model.
# Snapshots are single time frames of the training cases, reported steps only,
# so a padded tail never enters the fit and no case needs excluding.
#
# Floors measured 2026-09-14 (hash split, 199 cases, frozen masked_r2, valid/test):
#   TC    r=6 0.974/0.972   r=8 0.982/0.981   r=10 0.991/0.991   r=16 exact
#   HRR   r=2 0.977/0.980   r=3 0.999/0.999   r=4 1.0000/1.0000 (4 d.p.)
# The HRR budget is effectively rank 3-4: Q_TOTAL is the sum of the other four
# channels to within ~8 kW, and the radiative/convective split tracks HRR. A
# 4-mode head therefore builds the energy-closure structure into the output
# space, where the direct heads (Q_TOTAL R2 0.09-0.39 in every architecture) have
# nothing that ties the channels together.
R_TC_SPATIAL = int(os.environ.get("POD_R_TC_SPATIAL", "8"))
R_HRR_SPATIAL = int(os.environ.get("POD_R_HRR_SPATIAL", "4"))


def spatial_basis_path(split=None):
    return os.environ.get(
        "PART1_POD_SPATIAL_BASIS",
        os.path.join(PROJECT_DIR, f"pod_spatial_basis_part1_{split or SPLIT_MODE}.pt"))


def fit_spatial_pod(Y, mask, train_idx):
    """Sensor-wise POD over every reported training frame.

    Y is (N, T, C) standardised. Returns the mean frame (C,), the modes as ROWS
    (C, C), singular values, and the train std of each coefficient.
    """
    F = Y[train_idx][mask[train_idx] > 0].astype(np.float64)        # (frames, C)
    mean = F.mean(axis=0)
    _, S, Vt = np.linalg.svd(F - mean, full_matrices=False)
    pivot = np.abs(Vt).argmax(axis=1)
    Vt = Vt * np.sign(Vt[np.arange(len(Vt)), pivot])[:, None]
    coef = (F - mean) @ Vt.T
    coef_std = coef.std(axis=0)
    coef_std[coef_std < 1e-12] = 1.0
    return {"kind": "spatial", "mean": mean, "modes": Vt, "sigma": S,
            "coef_std": coef_std, "n_frames": int(len(F)), "shape": (Y.shape[2],)}


def project_spatial(Y, basis, r):
    """(N, T, C) standardised -> (N, T, r) unit-variance coefficients."""
    return ((np.asarray(Y, dtype=np.float64) - basis["mean"]) @ basis["modes"][:r].T) \
        / basis["coef_std"][:r]


def reconstruct_spatial(A, basis, r):
    """(N, T, r) unit-variance coefficients -> (N, T, C) standardised."""
    return basis["mean"] + (np.asarray(A, dtype=np.float64) * basis["coef_std"][:r]) \
        @ basis["modes"][:r]


def load_spatial_basis(path=None, split=None):
    path = path or spatial_basis_path(split)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"spatial POD basis not found: {path}\n  fit it first:  "
            f"python fit_pod_part1.py --kind spatial --split {split or SPLIT_MODE}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format_version") != FORMAT_VERSION or blob.get("kind") != "spatial":
        raise RuntimeError(f"{path}: not a format-{FORMAT_VERSION} spatial basis; re-fit it")
    return blob


class SpatialPODReconstruction(nn.Module):
    """Per-step coefficients a(t) -> standardised TC and HRR, with physics floors.

    Same buffers-in-the-checkpoint and scaler guard as the space-time version,
    and the same hard ambient / zero-HRR clamps as the parent models.
    """

    kind = "spatial"
    # The scaler guard and floor installation are identical to the space-time
    # head's; they only touch _fit_scalers and the two floor buffers.
    set_output_scaling = PODReconstruction.set_output_scaling

    def __init__(self, r_tc=R_TC_SPATIAL, r_hrr=R_HRR_SPATIAL, path=None):
        super().__init__()
        blob = load_spatial_basis(path)
        self.r_tc, self.r_hrr = r_tc, r_hrr
        for name, r, n_ch in (("tc", r_tc, N_SENSORS), ("hrr", r_hrr, N_HRR_CHANNELS)):
            b = blob[name]
            if tuple(b["shape"]) != (n_ch,):
                raise RuntimeError(f"{name} spatial basis shape {b['shape']} != ({n_ch},)")
            if r > len(b["sigma"]):
                raise RuntimeError(f"{name}: r={r} exceeds {len(b['sigma'])} channels")
            self.register_buffer(f"{name}_mean",
                                 torch.as_tensor(b["mean"], dtype=torch.float32))
            self.register_buffer(f"{name}_modes",
                                 torch.as_tensor(b["modes"][:r], dtype=torch.float32))
            self.register_buffer(f"{name}_coef_std",
                                 torch.as_tensor(b["coef_std"][:r], dtype=torch.float32))

        self._fit_scalers = {"tc": blob["tc_scaler"], "hrr": blob["hrr_scaler"]}
        self.basis_digest = blob["corpus_digest"]
        self.register_buffer("ambient_scaled", torch.full((N_SENSORS,), -1e9))
        self.register_buffer("hrr_floor_scaled", torch.full((N_HRR_CHANNELS,), -1e9))

    def expand(self, name, a):
        """(B, T, r) unit-variance coefficients -> (B, T, C) standardised, unclamped."""
        return (getattr(self, f"{name}_mean")
                + (a * getattr(self, f"{name}_coef_std")) @ getattr(self, f"{name}_modes"))

    def forward(self, a_tc, a_hrr):
        tc = torch.maximum(self.expand("tc", a_tc), self.ambient_scaled)
        hrr = torch.maximum(self.expand("hrr", a_hrr), self.hrr_floor_scaled)
        return tc, hrr
