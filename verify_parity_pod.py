"""Prove the four POD-ROM projects differ from their parents and twins only where
they claim to.

    python verify_parity_pod.py               # files + bases + data
    python verify_parity_pod.py --files-only  # fast, no subprocesses

Byte-identical in all four ROM projects. Checks, each a named claim in the
projects' READMEs:

  1. INHERITED files are byte-identical to the parent's copy (so the data layer,
     trainer, evaluator, conditioner and DiT blocks are provably the parent's).
  2. ROM-SHARED files are byte-identical between the twins (so the reduced space
     and its training/scoring code are one implementation, not two).
  3. POD BASES are byte-identical between the twins wherever both hold one for
     the same corpus — the twins must regress onto the same space, or the
     KAN-vs-MLP / FunDiff-vs-FunDiff-KAN comparison gains a second variable.
  4. DATA: the corpus arrays built inside each ROM project's own venv hash the
     same as inside its parent's venv.

Exit code is non-zero on any mismatch. Run before any comparison claim.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_DIR)

TC_ROMS = {"bs8414_KAN_pod_surrogate": "bs8414_KAN_surrogate",
           "bs8414_MLP_pod_surrogate": "bs8414_MLP_surrogate"}
SLICE_ROMS = {"bs8414_fundiff_pod_surrogate": "bs8414_fundiff_surrogate",
              "bs8414_fundiff_kan_pod_surrogate": "bs8414_fundiff_kan_surrogate"}

TC_INHERITED = ["config_part1.py", "data_loader_part1.py", "physics_part1.py",
                "train_part1.py", "evaluate_part1.py", "layers_part1.py"]
SLICE_INHERITED = ["config.py", "config_part1.py", "data_loader_part1.py",
                   "slice_loader_part1.py", "dataset_part1.py",
                   "part1_conditioning.py", "physics_part1.py",
                   "model/__init__.py", "model/dit.py"]
KAN_SLICE_INHERITED = {"bs8414_fundiff_kan_pod_surrogate": ["kan_layers_part1.py"]}

TC_SHARED = ["pod_rom_part1.py", "fit_pod_part1.py", "rom_diagnostics_part1.py",
             "gp_pod_baseline_part1.py", "run_rom_campaign.sh", "README.md",
             "requirements.txt"]
SLICE_SHARED = ["pod_slices_part1.py", "fit_pod_slices.py", "train_dit_pod.py",
                "evaluate_pod.py", "gp_pod_baseline_slices.py", "run_rom_campaign.sh",
                "README.md", "requirements.txt"]

VARIABLE = "model_part1.py"


def sha(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line(ok, label, detail=""):
    print(f"    {'OK      ' if ok else 'MISMATCH'} {label:<44} {detail}")
    return ok


def check_inherited(roms, files, extra=None):
    ok = True
    for rom, parent in roms.items():
        for name in files + (extra or {}).get(rom, []):
            a, b = sha(os.path.join(ROOT, rom, name)), sha(os.path.join(ROOT, parent, name))
            good = a is not None and a == b
            ok &= line(good, f"{rom.replace('bs8414_', '')}/{name}",
                       f"== {parent}" if good else
                       f"rom {str(a)[:12]} parent {str(b)[:12]}")
    return ok


def check_twins(roms, files):
    ok = True
    names = list(roms)
    for name in files:
        digests = {r: sha(os.path.join(ROOT, r, name)) for r in names}
        good = None not in digests.values() and len(set(digests.values())) == 1
        ok &= line(good, name, f"{next(iter(digests.values()))[:16]}" if good else
                   "  ".join(f"{r}:{str(d)[:10]}" for r, d in digests.items()))
    return ok


def check_bases(roms, pattern, key_fn):
    """Group basis files by corpus key; every group present in both twins must agree."""
    ok = True
    seen = {}
    for rom in roms:
        for path in glob.glob(os.path.join(ROOT, rom, pattern)):
            seen.setdefault(key_fn(path), {}).setdefault(rom, set()).add(sha(path))
    if not seen:
        print("    (no POD basis fitted yet)")
    for key, by_rom in sorted(seen.items()):
        digests = set().union(*by_rom.values())
        both = len(by_rom) == len(roms)
        good = len(digests) == 1
        ok &= line(good, f"basis {key}",
                   (f"{next(iter(digests))[:16]} in " + ", ".join(sorted(by_rom)))
                   + ("" if both else "   (only one twin holds it yet)"))
    return ok


def tc_basis_key(path):
    return os.path.basename(path)


def slice_basis_key(path):
    try:
        import torch
        return "corpus " + torch.load(path, map_location="cpu",
                                      weights_only=False)["corpus_digest"]
    except Exception:
        # No torch in the interpreter running the check: fall back to the name.
        return os.path.relpath(path, os.path.dirname(os.path.dirname(path)))


SENSOR_FP = r"""
import hashlib, json, numpy as np
from data_loader_part1 import build_dataset, split_indices
params, tc, hrr, mask, meta, names = build_dataset(verbose=False)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]
idx = split_indices(meta)
print("FINGERPRINT " + json.dumps({"n": int(len(params)), "params": h(params), "tc": h(tc),
      "hrr": h(hrr), "mask": h(mask), "split": {k: len(v) for k, v in idx.items()}}))
"""

SLICE_FP = r"""
import hashlib, json, numpy as np
from slice_loader_part1 import load_corpus
from data_loader_part1 import assign_split
params, fields, time_mask, plane_mask, meta = load_corpus(verbose=False)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
c = {"train": 0, "valid": 0, "test": 0}
for m in meta: c[assign_split(m)] += 1
print("FINGERPRINT " + json.dumps({"n": int(len(params)), "params": h(params),
      "fields": h(fields), "plane_mask": h(plane_mask), "split": c}))
"""


def fingerprint(project, snippet):
    py = os.path.join(ROOT, project, "venv", "Scripts", "python.exe")
    if not os.path.isfile(py):
        return None, "no venv"
    p = subprocess.run([py, "-c", snippet], cwd=os.path.join(ROOT, project),
                       capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if ln.startswith("FINGERPRINT "):
            return json.loads(ln[12:]), None
    tail = p.stderr.strip().splitlines()
    return None, tail[-1] if tail else "no fingerprint"


def check_data(roms, snippet):
    ok = True
    projects = list(roms) + sorted(set(roms.values()))
    fps = {}
    for proj in projects:
        fp, err = fingerprint(proj, snippet)
        fps[proj] = fp
        print(f"    {proj:<36} " + (json.dumps(fp) if fp else f"ERROR {err}"))
    good = [f for f in fps.values() if f]
    if len(good) != len(projects) or any(f != good[0] for f in good):
        print("    MISMATCH — ROM and parent venvs do not build the same corpus")
        return False
    print(f"    all {len(good)} agree")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-only", action="store_true")
    args = ap.parse_args()
    print("=" * 78 + "\n  POD-ROM parity\n" + "=" * 78)
    ok = True
    print("\n  1. inherited files == parent")
    ok &= check_inherited(TC_ROMS, TC_INHERITED)
    ok &= check_inherited(SLICE_ROMS, SLICE_INHERITED, KAN_SLICE_INHERITED)
    print("\n  2. ROM-shared files identical between twins")
    ok &= check_twins(TC_ROMS, TC_SHARED)
    ok &= check_twins(SLICE_ROMS, SLICE_SHARED)
    ok &= check_twins({**TC_ROMS, **SLICE_ROMS}, ["verify_parity_pod.py"])
    print("\n  variable file (expected to differ)")
    for rom in list(TC_ROMS) + list(SLICE_ROMS):
        print(f"    {str(sha(os.path.join(ROOT, rom, VARIABLE)))[:16]}  {rom}/{VARIABLE}")
    print("\n  3. POD bases identical between twins")
    ok &= check_bases(TC_ROMS, "pod_basis_part1_*.pt", tc_basis_key)
    ok &= check_bases(TC_ROMS, "pod_spatial_basis_part1_*.pt", tc_basis_key)
    ok &= check_bases(SLICE_ROMS, "models_*/pod_basis.pt", slice_basis_key)
    if not args.files_only:
        print("\n  4. corpus arrays, built in each project's own venv")
        ok &= check_data(TC_ROMS, SENSOR_FP)
        ok &= check_data(SLICE_ROMS, SLICE_FP)
    print("\n  " + ("PARITY HOLDS" if ok else "PARITY BROKEN — do not compare results"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
