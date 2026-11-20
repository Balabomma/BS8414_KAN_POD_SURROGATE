r"""Material table for THIS project's Streamlit app, pinned to the paper's corpus.

    venv\Scripts\activate
    python export_app_assets_paper.py

Writes `app_assets/part1_materials.json` in this project ONLY.

Why not `..\export_app_assets.py`: that script loads the corpus through the KAN
project's data layer with no CHID allowlist, so it reads whatever has finished
in `D:\Bs8414_05052026\Part1\_completed` today (227 folders on 2026-11-20, not
the paper's 199), and it writes the table into all four sibling app projects.
Those are parallel experiments with their own tables; this app must not
rewrite them, and its table must describe the corpus the paper's model was
trained and scored on: the frozen 199 in `..\frozen_199_chids.txt`.

The output schema is identical to the shared exporter's
(`bs8414-part1-app-assets/1`), so `app_common_part1.py` reads it unchanged.
The same refusal applies: if a cladding id maps to more than one core+reaction
block, or an insulation id to more than one insulation block, nothing is
written, because the table would average two physically different systems
under one name.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(os.path.dirname(HERE), "frozen_199_chids.txt")

# Must be set before data_loader_part1 is imported: it reads the allowlist at
# import time.
os.environ["PART1_CHID_ALLOWLIST"] = FROZEN
sys.path.insert(0, HERE)

from config_part1 import (  # noqa: E402
    SIMS_DIR, CLADDING_SYSTEMS, INSULATION_SYSTEMS, MATERIAL_FEATURES,
    GEOMETRY_NAMES, N_GEOMETRY, T_AMBIENT, N_TIMESTEPS, T_END,
)
from data_loader_part1 import build_dataset  # noqa: E402

# Which of the 13 features each axis owns, in MATERIAL_FEATURES order
# (identical to ..\export_app_assets.py).
CORE_IDX = [0, 1, 2, 3, 4, 5]
INS_IDX = [6, 7, 8, 9, 10]
REAC_IDX = [11, 12]

PAPER_N_SIMS = 199
PAPER_N_BUILDUPS = 41


def _named(raw, idx):
    return {MATERIAL_FEATURES[i]: float(raw[i]) for i in idx}


def main():
    with open(FROZEN) as f:
        frozen = {line.strip() for line in f if line.strip()}
    print(f"corpus   : {SIMS_DIR}")
    print(f"allowlist: {FROZEN} ({len(frozen)} CHIDs)")

    _params, _tc, _hrr, _mask, meta, _names = build_dataset(verbose=False)
    loaded = {m["chid"] for m in meta}
    if loaded != frozen:
        raise SystemExit(
            f"REFUSING TO WRITE: loaded {len(loaded)} simulations, which is not "
            f"the frozen set ({len(loaded - frozen)} extra, "
            f"{len(frozen - loaded)} missing: {sorted(frozen - loaded)[:5]})")
    assert len(meta) == PAPER_N_SIMS, len(meta)

    # An id must fix its material block, or the table would mix two systems.
    blocks = {"cladding": defaultdict(set), "insulation": defaultdict(set)}
    for m in meta:
        raw = m["material_raw"]
        blocks["cladding"][m["cladding"]].add(
            tuple(float(raw[i]) for i in CORE_IDX + REAC_IDX))
        blocks["insulation"][m["insulation"]].add(
            tuple(float(raw[i]) for i in INS_IDX))
    bad = [(axis, name, len(v)) for axis, d in blocks.items()
           for name, v in d.items() if len(v) > 1]
    if bad:
        raise SystemExit(f"REFUSING TO WRITE: ids with more than one material "
                         f"vector: {bad}")
    print("every id maps to exactly one material vector")

    clad_rows, ins_rows = {}, {}
    clad_decks, ins_decks = defaultdict(int), defaultdict(int)
    pairs = defaultdict(set)
    for m in meta:
        raw, clad, ins = m["material_raw"], m["cladding"], m["insulation"]
        clad_decks[clad] += 1
        ins_decks[ins] += 1
        pairs[f"{clad}|{ins}"].add(int(m["geom_id"]))
        clad_rows.setdefault(clad, {
            "id": CLADDING_SYSTEMS[clad], "core_matl": m["core_matl"],
            "core": _named(raw, CORE_IDX), "reac": _named(raw, REAC_IDX)})
        ins_rows.setdefault(ins, {
            "id": INSULATION_SYSTEMS[ins], "ins_matl": m["ins_matl"],
            "ins": _named(raw, INS_IDX)})
    for name, row in clad_rows.items():
        row["n_decks"] = clad_decks[name]
    for name, row in ins_rows.items():
        row["n_decks"] = ins_decks[name]

    assert len(pairs) == PAPER_N_BUILDUPS, (
        f"{len(pairs)} build-ups, paper states {PAPER_N_BUILDUPS}")

    asset = {
        "schema": "bs8414-part1-app-assets/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": "frozen_199_chids.txt (the paper's 199 simulations)",
        "n_simulations": len(meta),
        "material_features": MATERIAL_FEATURES,
        "core_idx": CORE_IDX, "ins_idx": INS_IDX, "reac_idx": REAC_IDX,
        "t_ambient_c": T_AMBIENT,
        "t_end_s": T_END,
        "n_timesteps": N_TIMESTEPS,
        "geometry_names": {str(k): v for k, v in GEOMETRY_NAMES.items()},
        "n_geometry": N_GEOMETRY,
        "cladding": dict(sorted(clad_rows.items(), key=lambda kv: kv[1]["id"])),
        "insulation": dict(sorted(ins_rows.items(), key=lambda kv: kv[1]["id"])),
        "observed": {k: sorted(v) for k, v in sorted(pairs.items())},
    }
    out_dir = os.path.join(HERE, "app_assets")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "part1_materials.json")
    with open(out, "w") as f:
        f.write(json.dumps(asset, indent=1))
    print(f"wrote {out}: {len(meta)} simulations, {len(pairs)} build-ups, "
          f"{len(clad_rows)} claddings, {len(ins_rows)} insulants")


if __name__ == "__main__":
    main()
