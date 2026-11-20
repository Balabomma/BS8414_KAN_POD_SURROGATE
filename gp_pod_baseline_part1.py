"""Gaussian-process POD ROM: the probabilistic baseline for the Part1 TC ROMs.

    python gp_pod_baseline_part1.py --out-dir models_gp_pod_hash
    python gp_pod_baseline_part1.py --split system --out-dir models_gp_pod_system

Source of the method: the CYPHER Training School PML/PEML session (Cicirello,
`L2_fileB_gaussian_processes_intro.ipynb`, `L2_hands_on_1_...sysID...ipynb`) —
GP regression with an ARD Matern kernel, hyperparameters by maximising the
marginal likelihood. Applied here in the POD-GPR form common in parametric ROMs:
one independent GP per unit-variance POD coefficient, over the SAME basis the
neural ROMs use.

Why it is in the project: with ~150 training cases the question is whether a
network is needed at all. A GP is the strongest simple answer to that question,
and unlike the networks it states its own uncertainty.

Inputs (38-d, fixed and fitted on train only):
    one-hot cladding (12) + one-hot insulation (5) + one-hot geometry (8)
    + the 13 material features, standardised on the training rows
One-hot rather than embeddings because a GP has no learned layers; one-hot
geometry keeps the 8-way (non-additive) treatment the neural models use.

Predictive uncertainty is propagated through the basis exactly (coefficients are
independent GPs, the reconstruction is linear):

    Var y(t, ch) = sum_k (s_k sd_k)^2 V_k(t, ch)^2

That variance cannot see truncation error, so coverage is reported twice: GP
only, and GP plus the train-set projection residual variance. Expect the first to
under-cover; the second is the honest interval.

Scored with the frozen contract's own functions (`evaluate_part1.masked_r2`,
`masked_rmse`, `physics_gates`) and written to <out-dir>/evaluation_gp.json —
not evaluation_part1.json, so checkpoint pickers never mistake it for a network
run.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from config_part1 import (N_CLADDING, N_INSULATION, N_GEOMETRY, COL_CLADDING,
                          COL_INSULATION, COL_GEOM, SPLIT_MODE, SENSOR_GROUPS,
                          GROUP_SIZES, HRR_CHANNELS, GEOMETRY_NAMES, T_AMBIENT)
from data_loader_part1 import build_dataset, prepare_data_splits, ChannelScaler
from evaluate_part1 import masked_r2, masked_rmse, physics_gates
import pod_rom_part1 as pod

Z95 = 1.959964


def features(params, mat_mean=None, mat_std=None):
    n = len(params)
    oh = []
    for col, size in ((COL_CLADDING, N_CLADDING), (COL_INSULATION, N_INSULATION),
                      (COL_GEOM, N_GEOMETRY)):
        m = np.zeros((n, size))
        m[np.arange(n), params[:, col].astype(int)] = 1.0
        oh.append(m)
    mat = params[:, 3:].astype(np.float64)
    if mat_mean is None:
        mat_mean, mat_std = mat.mean(0), mat.std(0)
        mat_std[mat_std < 1e-12] = 1.0
    return np.hstack(oh + [(mat - mat_mean) / mat_std]), mat_mean, mat_std


def fit_gps(X, C, restarts, seed):
    """One GP per coefficient column of C."""
    gps, notes = [], []
    d = X.shape[1]
    for k in range(C.shape[1]):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3),
                           nu=2.5)
                  + WhiteKernel(1e-2, (1e-6, 1e1)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                      n_restarts_optimizer=restarts,
                                      random_state=seed + k)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(X, C[:, k])
        notes.append({"mode": k, "log_marginal_likelihood":
                      float(gp.log_marginal_likelihood_value_),
                      "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning)
                                                  for w in caught),
                      "kernel": str(gp.kernel_)})
        gps.append(gp)
    return gps, notes


def predict(gps, X):
    mu, sd = zip(*(gp.predict(X, return_std=True) for gp in gps))
    return np.stack(mu, 1), np.stack(sd, 1)


def field_variance(sd_hat, basis, r):
    """Var of the standardised reconstruction, (N, T, C)."""
    w = (sd_hat * basis["coef_std"][:r]) ** 2               # (N, r)
    v = w @ (basis["modes"][:r] ** 2)                        # (N, T*C)
    return v.reshape((len(sd_hat),) + basis["shape"])


def coverage(pred, true, var_phys, mask):
    keep = (mask > 0)[..., None] & np.ones_like(true, dtype=bool)
    half = Z95 * np.sqrt(np.maximum(var_phys, 0.0))
    inside = np.abs(true - pred) <= half
    return {"coverage95": float(inside[keep].mean()),
            "mean_halfwidth": float(half[keep].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT_MODE, choices=("hash", "system"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--r-tc", type=int, default=pod.R_TC)
    ap.add_argument("--r-hrr", type=int, default=pod.R_HRR)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    eval_path = os.path.join(args.out_dir, "evaluation_gp.json")
    if os.path.isfile(eval_path):
        raise SystemExit(f"{eval_path} exists — choose a new --out-dir")
    os.makedirs(args.out_dir, exist_ok=True)

    params, tc, hrr, mask, meta, _ = build_dataset(verbose=False)
    datasets, tc_scaler, hrr_scaler, info, _ = prepare_data_splits(
        params, tc, hrr, mask, meta, mode=args.split, verbose=True)
    idx = info["indices"]

    blob = pod.load_basis(split=args.split)
    for name, sc in (("tc", tc_scaler), ("hrr", hrr_scaler)):
        if not np.allclose(sc.mean, blob[f"{name}_scaler"]["mean"], atol=1e-5):
            raise SystemExit(f"{name} scaler differs from the basis's — re-fit the basis")

    keep = pod.complete_cases(mask, idx["train"])
    X_tr, m_mean, m_std = features(params[keep])
    ambient_s = (T_AMBIENT - tc_scaler.mean) / tc_scaler.scale
    hrr_floor_s = np.full(len(HRR_CHANNELS), -np.inf)
    hrr_floor_s[0] = (0.0 - hrr_scaler.mean[0]) / hrr_scaler.scale[0]

    parts = {}
    for name, raw, scaler, r in (("tc", tc, tc_scaler, args.r_tc),
                                 ("hrr", hrr, hrr_scaler, args.r_hrr)):
        basis = blob[name]
        Y_s = scaler.transform(raw).astype(np.float64)
        C_tr = pod.project(Y_s[keep], mask[keep], basis, r)
        print(f"\n  fitting {r} GPs on {name.upper()} coefficients "
              f"({len(keep)} cases, {X_tr.shape[1]}-d inputs) ...", flush=True)
        gps, notes = fit_gps(X_tr, C_tr, args.restarts, args.seed)
        resid_var = pod.projection_residual_var(Y_s, keep, basis, r)
        parts[name] = dict(basis=basis, gps=gps, notes=notes, r=r, scaler=scaler,
                           raw=raw, resid_var=resid_var)

    results = {"method": "POD-GPR (ARD Matern 2.5 + white), one GP per coefficient",
               "split_mode": args.split, "r_tc": args.r_tc, "r_hrr": args.r_hrr,
               "basis_corpus_digest": blob["corpus_digest"],
               "n_train_fit": len(keep),
               "restarts": args.restarts, "seed": args.seed,
               "chid_allowlist": __import__("os").environ.get("PART1_CHID_ALLOWLIST")}
    for split in ("train", "valid", "test"):
        ids = idx[split]
        if len(ids) == 0:
            continue
        X, _, _ = features(params[ids], m_mean, m_std)
        out = {}
        for name, P in parts.items():
            mu_c, sd_c = predict(P["gps"], X)
            rec = pod.reconstruct(mu_c, P["basis"], P["r"])
            floor = ambient_s if name == "tc" else hrr_floor_s
            rec = np.maximum(rec, floor)
            pred = P["scaler"].inverse(rec.astype(np.float32))
            var_s = field_variance(sd_c, P["basis"], P["r"])
            scale2 = np.asarray(P["scaler"].scale) ** 2
            out[name] = dict(pred=pred, var_gp=var_s * scale2,
                             var_all=(var_s + P["resid_var"][None]) * scale2)
        tc_pred, hrr_pred = out["tc"]["pred"], out["hrr"]["pred"]
        m = mask[ids]
        row = {"n": len(ids),
               "tc_r2": masked_r2(tc_pred, tc[ids], m),
               "tc_rmse": masked_rmse(tc_pred, tc[ids], m),
               "groups": {}, "hrr": {}, "geometry": {},
               "tc_uq_gp_only": coverage(tc_pred, tc[ids], out["tc"]["var_gp"], m),
               "tc_uq_with_truncation": coverage(tc_pred, tc[ids], out["tc"]["var_all"], m),
               "hrr_uq_with_truncation": coverage(hrr_pred, hrr[ids], out["hrr"]["var_all"], m),
               "physics": physics_gates(tc_pred, tc[ids], m)}
        start = 0
        for g, size in zip(SENSOR_GROUPS, GROUP_SIZES):
            sl = slice(start, start + size)
            row["groups"][g] = {"r2": masked_r2(tc_pred[..., sl], tc[ids][..., sl], m),
                                "rmse": masked_rmse(tc_pred[..., sl], tc[ids][..., sl], m)}
            start += size
        for i, ch in enumerate(HRR_CHANNELS):
            sl = slice(i, i + 1)
            row["hrr"][ch] = {"r2": masked_r2(hrr_pred[..., sl], hrr[ids][..., sl], m),
                              "rmse": masked_rmse(hrr_pred[..., sl], hrr[ids][..., sl], m)}
        gids = np.array([meta[i]["geom_id"] for i in ids])
        for gid in sorted(set(gids.tolist())):
            sel = gids == gid
            row["geometry"][GEOMETRY_NAMES[gid]] = {
                "n": int(sel.sum()), "r2": masked_r2(tc_pred[sel], tc[ids][sel], m[sel]),
                "rmse": masked_rmse(tc_pred[sel], tc[ids][sel], m[sel])}
        results[split] = row
        print(f"\n  [{split}] {len(ids)} sims   TC R2 {row['tc_r2']:.4f}  "
              f"RMSE {row['tc_rmse']:.2f} degC   HRR R2 {row['hrr']['HRR']['r2']:.4f}")
        print(f"    95% coverage TC: GP only {row['tc_uq_gp_only']['coverage95']:.3f}, "
              f"with truncation {row['tc_uq_with_truncation']['coverage95']:.3f} "
              f"(half-width {row['tc_uq_with_truncation']['mean_halfwidth']:.1f} degC)")

    if "valid" in results and "test" in results:
        results["combined_tc_r2"] = 0.5 * (results["valid"]["tc_r2"]
                                           + results["test"]["tc_r2"])
        print(f"\n  combined valid+test TC R2: {results['combined_tc_r2']:.4f}")
    results["gp_fit"] = {k: P["notes"] for k, P in parts.items()}
    results["minutes"] = round((time.time() - t0) / 60, 1)

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"  written: {eval_path}  ({results['minutes']} min)")


if __name__ == "__main__":
    main()
