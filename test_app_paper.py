r"""Checks that app_part1.py serves the paper's model, and serves it correctly.

    venv\Scripts\activate
    python test_app_paper.py

Four independent checks, each of which can fail on its own:

 1. SCOPE    exactly the five paper seed bases are offered, and the r16, podnn
             and system-split controls in this project are refused by what
             their checkpoints contain, not by their folder names.
 2. RENDER   every seed base renders through Streamlit's headless AppTest with
             no exception and physically sane outputs.
 3. INPUTS   the 16-d vector the app builds from its material table equals
             the training loader's vector for all 199 simulations.
 4. OUTPUTS  the app's own load + predict path reproduces each seed base's
             RECORDED test R2 (evaluation_part1.json / r9_result.json), and
             matches the seed-48 predictions behind the paper's Figs 7-12.

Reads the FDS corpus (for check 3 and the test truth), so it runs on this
workstation, not on a clone.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PART1_CHID_ALLOWLIST"] = os.path.join(os.path.dirname(HERE),
                                                  "frozen_199_chids.txt")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

FIG_NPZ = r"D:\Code_space\Firesnc\figures\kanpod_test_predictions.npz"
# Each control must be refused on what its checkpoint contains. The first
# guard to fire is reported; r16 trips the parameter count before the rank
# check, so the rank guard is also exercised on its own below.
CONTROLS = {
    "models_part1_podattn_r16_hash_s42_r1": ("parameters", "rom.tc_modes"),
    "models_part1_podnn_hash_s42_r1": ("model_name",),
    "models_part1_podattn_system_s42_r1": ("split_mode",),
}
FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    import app_part1 as app
    from app_common_part1 import build_params, load_assets
    from data_loader_part1 import build_dataset, split_indices
    from evaluate_part1 import masked_r2

    # ── 1. scope ─────────────────────────────────────────────────────────
    print("1. SCOPE")
    runs, rejected = app.discover_runs()
    names = [r["name"] for r in runs]
    check(names == [rel for _, rel in app.PAPER_RUNS],
          f"offers exactly the 5 paper seed bases in order: {names}")
    check(not rejected, f"no paper seed base rejected: {rejected}")
    for ctrl, keys in CONTROLS.items():
        ck = torch.load(sorted(glob.glob(os.path.join(HERE, ctrl, "*member*.pt")))[0],
                        map_location="cpu", weights_only=False)
        ok, why = app._paper_signature(ck)
        check(not ok and any(k in why for k in keys),
              f"control {ctrl} refused by content: {why!r}")
        if ctrl.startswith("models_part1_podattn_r16"):
            # Rank guard on its own: pretend the parameter count matched.
            ck["n_params"] = app.PAPER_N_PARAMS
            ok2, why2 = app._paper_signature(ck)
            check(not ok2 and "rom.tc_modes" in why2,
                  f"rank guard alone also refuses r16: {why2!r}")

    # ── 3. inputs (before 2, it needs the corpus anyway) ────────────────
    print("3. INPUTS")
    assets = load_assets()
    params, tc, _hrr, mask, meta, _ = build_dataset(verbose=False)
    check(len(meta) == 199, f"training loader sees {len(meta)} simulations")
    worst = 0.0
    for p, m in zip(params, meta):
        raw = [float(v) for v in m["material_raw"]]
        vec = build_params(assets, m["cladding"], m["insulation"],
                           int(m["geom_id"]), raw)
        # the app prefills material from its table; check the table agrees too
        from app_common_part1 import default_material
        table = default_material(assets, m["cladding"], m["insulation"])
        worst = max(worst, float(np.abs(np.asarray(vec) - np.asarray(p)).max()),
                    float(np.abs(np.asarray(table) - np.asarray(raw)).max()))
    check(worst < 1e-6, f"app input vector == training vector on all 199 "
                        f"(max |diff| {worst:.2e})")

    # ── 4. outputs ───────────────────────────────────────────────────────
    print("4. OUTPUTS")
    idx = split_indices(meta, mode="hash")
    test_i = idx["test"] if isinstance(idx, dict) else idx[2]
    test_i = list(test_i)
    check(len(test_i) == 24, f"hash test partition has {len(test_i)} cases")
    t_true = np.asarray(tc)[test_i]
    t_mask = np.asarray(mask)[test_i]
    test_chids = [meta[i]["chid"] for i in test_i]

    z = np.load(FIG_NPZ, allow_pickle=True)
    order = [list(z["chid"]).index(c) for c in test_chids]
    check(sorted(z["chid"]) == sorted(test_chids),
          "figure file holds the same 24 test cases")

    for r in runs:
        models, tcs, hrs, ck, dev = app.load_ensemble(r["path"])
        preds = []
        for i in test_i:
            m = meta[i]
            vec = build_params(assets, m["cladding"], m["insulation"],
                               int(m["geom_id"]),
                               [float(v) for v in m["material_raw"]])
            tc_m, _ = app.predict(models, vec, tcs, hrs, dev)
            preds.append(tc_m.mean(axis=0))
        preds = np.stack(preds)
        # true and mask from the loader are in physical units already?
        r2 = masked_r2(preds, z["actual"][order], z["mask"][order])
        check(abs(r2 - r["test_r2"]) < 5e-4,
              f"{r['label']}: app-path test R2 {r2:.4f} vs recorded "
              f"{r['test_r2']:.4f}")
        if r["label"] == "seed base 48":
            d = float(np.abs(preds - z["pred_s48"][order]).max())
            check(d < 0.05, f"seed base 48 matches the paper-figure predictions "
                            f"(max |diff| {d:.4f} degC over 69,504 values)")
        sub = float((preds < app.T_AMBIENT - 0.5).mean())
        check(sub == 0.0, f"{r['label']}: no sub-ambient prediction "
                          f"({sub:.4%} of values)")

    # ── 2. render ────────────────────────────────────────────────────────
    print("2. RENDER")
    from streamlit.testing.v1 import AppTest
    for r in runs:
        at = AppTest.from_file(os.path.join(HERE, "app_part1.py"),
                               default_timeout=180)
        at.run()
        at.sidebar.selectbox[0].select(r["name"]).run()
        exc = [e.value for e in at.exception]
        metrics = {m.label: m.value for m in at.metric}
        check(not exc, f"{r['label']}: renders with no exception {exc[:1]}")
        try:
            lv1 = float(metrics["Peak External LV1"].split()[0])
            lv2 = float(metrics["Peak External LV2"].split()[0])
            hrr = float(metrics["Peak HRR"].split()[0])
            sane = 18 < lv1 < 1400 and 18 < lv2 < 1400 and 1000 < hrr < 20000
        except (KeyError, ValueError):
            sane, lv1, lv2, hrr = False, None, None, None
        check(sane, f"{r['label']}: peaks LV1 {lv1} / LV2 {lv2} degC, "
                    f"HRR {hrr} kW, screen {metrics.get('BR 135 external screen')}")
    at = AppTest.from_file(os.path.join(HERE, "app_part1.py"), default_timeout=180)
    at.run()
    check(at.sidebar.selectbox[0].value == "models_part1_podattn_hash_s48_r1",
          f"default seed base is 48: {at.sidebar.selectbox[0].value}")

    print()
    print(f"{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
