"""Offline stage of the Part1 POD ROMs: fit a basis and measure its floor.

    python fit_pod_part1.py                                   # spatial basis, hash split
    python fit_pod_part1.py --kind spatial --copy-to ..\\bs8414_MLP_pod_surrogate
    python fit_pod_part1.py --kind spacetime                   # GP baseline / archived POD-NN

Two kinds of basis, both fitted on TRAIN rows only, in the train-only scalers:

  spatial    modes over the 16 TCs / 5 HRR channels, snapshots = training frames.
             The head of the POD-Attention-LSTM (`model_part1.py`).
             -> pod_spatial_basis_part1_<split>.pt, pod_spatial_floor_part1_<split>.json
  spacetime  modes over whole 181-step trajectories, snapshots = training cases.
             Used by the GP POD-ROM baseline and the archived POD-NN
             (`model_part1_podnn.py`).
             -> pod_basis_part1_<split>.pt, pod_floor_part1_<split>.json

The projection floor is what a perfect coefficient predictor would score: the
first diagnostic of Mendez's Tutorial 3 and of the CYPHER ROM challenge's
`diagnose.py`. It is scored with `evaluate_part1.masked_r2` / `masked_rmse`, the
frozen contract's own functions, so floor and model numbers are comparable.

Re-running on an unchanged corpus never re-fits: it re-scores the existing basis,
and exits non-zero if the corpus changed (pass --force to re-fit). The twin ROM
projects must share one basis file byte-for-byte; fit once and `--copy-to` the
twin. `verify_parity_pod.py` checks the hashes.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from config_part1 import SPLIT_MODE, HRR_CHANNELS
from data_loader_part1 import build_dataset, prepare_data_splits
from evaluate_part1 import masked_r2, masked_rmse
import pod_rom_part1 as pod

KINDS = {
    "spatial": {
        "path": pod.spatial_basis_path,
        "floor_name": "pod_spatial_floor_part1_{}.json",
        "fit": lambda Y, mask, idx: pod.fit_spatial_pod(Y, mask, idx["train"]),
        "rebuild": lambda Y, mask, b, r: pod.reconstruct_spatial(
            pod.project_spatial(Y, b, r), b, r),
        "ranks": {"tc": (2, 4, 6, 8, 10, 12, 16), "hrr": (1, 2, 3, 4, 5)},
        "r_default": {"tc": pod.R_TC_SPATIAL, "hrr": pod.R_HRR_SPATIAL},
    },
    "spacetime": {
        "path": pod.basis_path,
        "floor_name": "pod_floor_part1_{}.json",
        "fit": lambda Y, mask, idx: pod.fit_pod(Y, pod.complete_cases(mask, idx["train"])),
        "rebuild": lambda Y, mask, b, r: pod.reconstruct(pod.project(Y, mask, b, r), b, r),
        "ranks": {"tc": (2, 4, 8, 16, 32, 64), "hrr": (2, 4, 8, 16, 32, 64)},
        "r_default": {"tc": pod.R_TC, "hrr": pod.R_HRR},
    },
}


def floor_table(spec, Y_s, raw, scaler, mask, idx, basis, ranks):
    out = {}
    for split in ("train", "valid", "test"):
        ids = idx[split]
        if len(ids) == 0:
            continue
        rows = {}
        for r in ranks:
            rec = spec["rebuild"](Y_s[ids], mask[ids], basis, r)
            pred = scaler.inverse(rec.astype(np.float32))
            rows[str(r)] = {"r2": masked_r2(pred, raw[ids], mask[ids]),
                            "rmse": masked_rmse(pred, raw[ids], mask[ids])}
        out[split] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spatial", choices=tuple(KINDS))
    ap.add_argument("--split", default=SPLIT_MODE, choices=("hash", "system"))
    ap.add_argument("--force", action="store_true", help="re-fit even though a basis exists")
    ap.add_argument("--copy-to", action="append", default=[],
                    help="twin ROM project dir(s) to receive a byte-identical copy")
    args = ap.parse_args()
    spec = KINDS[args.kind]

    t0 = time.time()
    params, tc, hrr, mask, meta, sensor_names = build_dataset(verbose=False)
    _, tc_scaler, hrr_scaler, info, _ = prepare_data_splits(
        params, tc, hrr, mask, meta, mode=args.split, verbose=True)
    idx = info["indices"]
    digest = pod.array_digest(params, tc, hrr, mask)
    tc_s = tc_scaler.transform(tc).astype(np.float64)
    hrr_s = hrr_scaler.transform(hrr).astype(np.float64)

    path = spec["path"](args.split)
    existing = (torch.load(path, map_location="cpu", weights_only=False)
                if os.path.isfile(path) else None)
    if existing is not None and not args.force:
        if existing.get("corpus_digest") != digest:
            raise SystemExit(
                f"{path} was fitted on corpus {existing.get('corpus_digest')}, the "
                f"current one is {digest}. Checkpoints keep their own copy of the basis "
                f"in their buffers, so re-fitting is safe — pass --force.")
        print(f"\n  {path} already fitted on this corpus ({digest}) — re-scoring only")
        tc_basis, hrr_basis = existing["tc"], existing["hrr"]
    else:
        tc_basis = spec["fit"](tc_s, mask, idx)
        hrr_basis = spec["fit"](hrr_s, mask, idx)
        keep = (pod.complete_cases(mask, idx["train"]) if args.kind == "spacetime"
                else [int(i) for i in idx["train"]])
        torch.save({
            "format_version": pod.FORMAT_VERSION, "kind": args.kind, "split": args.split,
            "corpus_digest": digest, "n_cases": int(len(params)),
            "train_chids": [meta[i]["chid"] for i in keep],
            "excluded_from_fit": [meta[i]["chid"] for i in idx["train"] if i not in keep],
            "tc": tc_basis, "hrr": hrr_basis,
            "tc_scaler": tc_scaler.state_dict(), "hrr_scaler": hrr_scaler.state_dict(),
            "sensor_names": sensor_names, "hrr_channels": HRR_CHANNELS,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, path)
        print(f"\n  {args.kind} basis fitted on {len(keep)} training cases -> {path}")

    report = {"kind": args.kind, "split": args.split, "corpus_digest": digest,
              "split_sizes": {k: len(v) for k, v in idx.items()},
              "r_default": spec["r_default"]}
    for name, basis, Y_s, raw, scaler, unit in (
            ("tc", tc_basis, tc_s, tc, tc_scaler, "degC"),
            ("hrr", hrr_basis, hrr_s, hrr, hrr_scaler, "kW")):
        r_def = spec["r_default"][name]
        ranks = sorted(set(spec["ranks"][name] + (r_def, len(basis["sigma"]))))
        table = floor_table(spec, Y_s, raw, scaler, mask, idx, basis, ranks)
        energy = pod.modes_for_energy(np.asarray(basis["sigma"]))
        report[name] = {"rank": len(basis["sigma"]), "modes_for_energy": energy,
                        "sigma": np.asarray(basis["sigma"]).tolist(), "floor": table}
        print(f"\n  {name.upper()} {args.kind} POD  rank {len(basis['sigma'])}  "
              f"modes for 90/99/99.9% energy: {list(energy.values())}")
        print(f"    projection floor, R2 / RMSE {unit}")
        print("    " + f"{'r':>5}" + "".join(f"{s:>21}" for s in table))
        for r in ranks:
            row = f"    {r:>5}"
            for s in table:
                v = table[s][str(r)]
                row += f"{v['r2']:>12.4f} {v['rmse']:>8.2f}"
            print(row + ("   <- default r" if r == r_def else ""))

    r = str(spec["r_default"]["tc"])
    floors = report["tc"]["floor"]
    if "valid" in floors and "test" in floors:
        report["combined_tc_r2_ceiling"] = 0.5 * (floors["valid"][r]["r2"]
                                                  + floors["test"][r]["r2"])
        print(f"\n  combined valid+test TC R2 ceiling at r={r}: "
              f"{report['combined_tc_r2_ceiling']:.4f}")

    floor_path = os.path.join(pod.PROJECT_DIR, spec["floor_name"].format(args.split))
    with open(floor_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"  written: {floor_path}   ({time.time() - t0:.0f} s)")

    for target in args.copy_to:
        for src in (path, floor_path):
            shutil.copyfile(src, os.path.join(target, os.path.basename(src)))
        print(f"  copied to {target}")


if __name__ == "__main__":
    main()
