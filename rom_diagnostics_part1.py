"""ROM error decomposition for a trained Part1 POD ROM: basis vs coefficient map.

    python rom_diagnostics_part1.py --model-dir models_part1_podattn_hash_s42_r1

The diagnostic at the centre of Mendez's Tutorial 3 ("the projection error and
the POD-Galerkin error are not the same curve") and of the CYPHER ROM challenge
(`local_test/diagnose.py`). With true coefficients a and predicted a_hat, both
reconstructed through the same orthonormal modes Phi_r,

    y - Phi_r a_hat  =  (y - Phi_r a)   +   Phi_r (a - a_hat)
                        truncation         coefficient error

and the two terms are orthogonal, so their squared errors ADD. A large
truncation share means grow r. A large coefficient share means the network
producing the coefficients is the bottleneck, and a bigger basis would buy
nothing.

Works for both heads, read from the checkpoint's own buffers:
  spatial    (POD-Attention-LSTM) — split per reported time FRAME
  spacetime  (archived POD-NN)    — split per complete CASE
The split is exact in standardised space before the physics clamp; the clamped
degC R2 the frozen evaluator reports is printed beside the floor. Also printed:
per-mode coefficient R2 for the leading modes.

Writes rom_diagnostics.json into the model directory.
"""
import argparse
import json
import os

import numpy as np
import torch

from config_part1 import DEVICE, GEOMETRY_NAMES, N_TIMESTEPS
from data_loader_part1 import build_dataset, prepare_data_splits
from evaluate_part1 import load_ensemble, masked_r2
import pod_rom_part1 as pod

N_MODES_REPORTED = 8


@torch.no_grad()
def ensemble_coefficients(models, params, time_array, device):
    """Mean coefficients over members (the reconstruction is affine in them)."""
    p = torch.as_tensor(params, dtype=torch.float32, device=device)
    t = torch.as_tensor(time_array, dtype=torch.float32, device=device)
    tc, hrr = [], []
    for m in models:
        out = m.coefficients(p, t)
        tc.append(out[0])
        hrr.append(out[1])
    return (torch.stack(tc).mean(0).cpu().numpy().astype(np.float64),
            torch.stack(hrr).mean(0).cpu().numpy().astype(np.float64))


def basis_from_model(model, name):
    """The basis the checkpoint was TRAINED with, read back from its buffers."""
    rom = model.rom
    modes = getattr(rom, f"{name}_modes").double().cpu().numpy()
    n_ch = modes.shape[1] if rom.kind == "spatial" else modes.shape[1] // N_TIMESTEPS
    shape = (n_ch,) if rom.kind == "spatial" else (N_TIMESTEPS, n_ch)
    return {"mean": getattr(rom, f"{name}_mean").double().cpu().numpy(), "modes": modes,
            "coef_std": getattr(rom, f"{name}_coef_std").double().cpu().numpy(),
            "shape": shape}


def rebuild(kind, Y, mask, basis):
    """(true coefficients, their reconstruction) for every case."""
    r = len(basis["coef_std"])
    if kind == "spatial":
        a = pod.project_spatial(Y, basis, r)
        return a, pod.reconstruct_spatial(a, basis, r)
    a = pod.project(Y, mask, basis, r)
    return a, pod.reconstruct(a, basis, r)


def decompose(kind, Y, mask, a_hat, basis):
    r = len(basis["coef_std"])
    if kind == "spatial":
        sel = mask > 0                                           # (N, T) reported frames
        unit = "frames"
    else:
        sel = np.all(mask > 0, axis=1)                           # (N,) complete cases
        unit = "complete cases"
    if not sel.any():
        return None
    a_true, rec_true = rebuild(kind, Y, mask, basis)
    rec_hat = (pod.reconstruct_spatial(a_hat, basis, r) if kind == "spatial"
               else pod.reconstruct(a_hat, basis, r))

    total = float(((Y - rec_hat)[sel] ** 2).sum())
    trunc = float(((Y - rec_true)[sel] ** 2).sum())
    coef = float(((rec_true - rec_hat)[sel] ** 2).sum())
    per_mode = []
    for k in range(min(N_MODES_REPORTED, r)):
        t, p = a_true[..., k][sel], a_hat[..., k][sel]
        ss = ((t - t.mean()) ** 2).sum()
        per_mode.append(float(1 - ((t - p) ** 2).sum() / ss) if ss > 0 else float("nan"))
    return {"unit": unit, "n": int(sel.sum()),
            "sq_error_total": total, "sq_error_truncation": trunc,
            "sq_error_coefficients": coef,
            "truncation_share": trunc / total if total > 0 else float("nan"),
            "coefficient_share": coef / total if total > 0 else float("nan"),
            "pythagoras_residual": abs(total - trunc - coef) / max(total, 1e-12),
            "per_mode_coef_r2": per_mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = load_ensemble(args.model_dir, device)
    split_mode = ckpt.get("split_mode")
    kind = getattr(models[0].rom, "kind", "spacetime")

    params, tc, hrr, mask, meta, _ = build_dataset(verbose=False)
    _, _, _, info, time_array = prepare_data_splits(params, tc, hrr, mask, meta,
                                                    mode=split_mode, verbose=False)
    tc_s = tc_scaler.transform(tc).astype(np.float64)
    hrr_s = hrr_scaler.transform(hrr).astype(np.float64)
    bases = {n: basis_from_model(models[0], n) for n in ("tc", "hrr")}
    ambient = models[0].rom.ambient_scaled.cpu().numpy()

    print("=" * 78)
    print(f"  POD-ROM error decomposition — {args.model_dir}")
    print(f"  {kind} head, {len(models)} member(s), split {split_mode}, "
          f"r_tc={len(bases['tc']['coef_std'])} r_hrr={len(bases['hrr']['coef_std'])}")
    print("=" * 78)

    report = {"model_dir": args.model_dir, "split_mode": split_mode, "kind": kind,
              "r_tc": len(bases["tc"]["coef_std"]), "r_hrr": len(bases["hrr"]["coef_std"])}
    for split in ("train", "valid", "test"):
        ids = info["indices"][split]
        if len(ids) == 0:
            continue
        a_tc, a_hrr = ensemble_coefficients(models, params[ids], time_array, device)
        row = {}
        for name, Y_s, a_hat in (("tc", tc_s, a_tc), ("hrr", hrr_s, a_hrr)):
            d = decompose(kind, Y_s[ids], mask[ids], a_hat, bases[name])
            row[name] = d
            if d is None:
                continue
            print(f"\n  [{split}] {name.upper()}  ({d['n']} {d['unit']})")
            print(f"    error share  truncation {d['truncation_share']:6.1%}   "
                  f"coefficients {d['coefficient_share']:6.1%}   "
                  f"(orthogonality residual {d['pythagoras_residual']:.1e})")
            print("    coefficient R2, leading modes: "
                  + " ".join(f"{v:6.3f}" for v in d["per_mode_coef_r2"]))

        basis = bases["tc"]
        r = len(basis["coef_std"])
        _, rec_true = rebuild(kind, tc_s[ids], mask[ids], basis)
        rec_hat = (pod.reconstruct_spatial(a_tc, basis, r) if kind == "spatial"
                   else pod.reconstruct(a_tc, basis, r))
        floor = tc_scaler.inverse(rec_true.astype(np.float32))
        rom = tc_scaler.inverse(np.maximum(rec_hat, ambient).astype(np.float32))
        row["tc_r2_rom_degC"] = masked_r2(rom, tc[ids], mask[ids])
        row["tc_r2_floor_degC"] = masked_r2(floor, tc[ids], mask[ids])
        print(f"    TC R2 (degC)  ROM {row['tc_r2_rom_degC']:.4f}   "
              f"projection floor {row['tc_r2_floor_degC']:.4f}   "
              f"gap {row['tc_r2_floor_degC'] - row['tc_r2_rom_degC']:.4f}")

        geo = {}
        gids = np.array([meta[i]["geom_id"] for i in ids])
        for gid in sorted(set(gids.tolist())):
            sel = gids == gid
            geo[GEOMETRY_NAMES[gid]] = {
                "n": int(sel.sum()),
                "rom_r2": masked_r2(rom[sel], tc[ids][sel], mask[ids][sel]),
                "floor_r2": masked_r2(floor[sel], tc[ids][sel], mask[ids][sel])}
        row["geometry"] = geo
        report[split] = row

    out = os.path.join(args.model_dir, "rom_diagnostics.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
