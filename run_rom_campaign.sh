#!/usr/bin/env bash
# POD-Attention-LSTM campaign for a Part1 thermocouple ROM project (KAN or MLP twin).
# Byte-identical in bs8414_KAN_pod_surrogate and bs8414_MLP_pod_surrogate.
#
#   ./run_rom_campaign.sh                     # hash split, seeds 42 45 48, 3 members
#   SEEDS="52" SPLIT=system ./run_rom_campaign.sh
#
# 1. Both POD bases — spatial (the model's head) and space-time (the GP baseline) —
#    reused if present, copied from the twin if it has them, fitted (then copied
#    to the twin) otherwise: the twins share ONE file of each. fit_pod_part1.py
#    then re-scores them and exits non-zero if the corpus changed.
# 2. Per seed: train_part1 -> evaluate_part1 -> rom_diagnostics_part1, into
#    models_part1_podattn_<split>_s<seed>_<tag>; a dir holding checkpoints is skipped.
# 3. The GP POD-ROM baseline, once per split.
# Bash, not PowerShell: Windows PowerShell 5.1 turns any python stderr line into a
# terminating error under ErrorActionPreference=Stop, and writes UTF-16 logs.
set -u
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
SEEDS=${SEEDS:-"42 45 48"}
SPLIT=${SPLIT:-hash}
MEMBERS=${MEMBERS:-3}
TAG=${TAG:-r1}
export PART1_SPLIT=$SPLIT PYTHONIOENCODING=utf-8
mkdir -p logs

here=$(basename "$(pwd)")
case $here in
  bs8414_KAN_pod_surrogate) twin=../bs8414_MLP_pod_surrogate ;;
  bs8414_MLP_pod_surrogate) twin=../bs8414_KAN_pod_surrogate ;;
  *) echo "unknown project $here"; exit 2 ;;
esac

step () {                                   # $1 = log name, rest = python args
  local name=$1; shift
  echo "  [$(date +%H:%M)] $name"
  if ! $PY -u "$@" > "logs/$name.log" 2> "logs/$name.err.log"; then
    echo "  FAILED: $name — see logs/$name.err.log"; tail -5 "logs/$name.err.log"; exit 1
  fi
}

for kind in spatial spacetime; do
  if [ $kind = spatial ]; then basis=pod_spatial_basis_part1_$SPLIT.pt; floor=pod_spatial_floor_part1_$SPLIT.json
  else basis=pod_basis_part1_$SPLIT.pt; floor=pod_floor_part1_$SPLIT.json; fi
  if [ ! -f "$basis" ]; then
    if [ -f "$twin/$basis" ]; then
      cp "$twin/$basis" . && cp "$twin/$floor" . 2>/dev/null
      echo "  $kind basis copied from $twin"
    else
      step "fit_pod_${kind}_$SPLIT" fit_pod_part1.py --kind $kind --split "$SPLIT" --copy-to "$twin"
    fi
  fi
  step "check_pod_${kind}_$SPLIT" fit_pod_part1.py --kind $kind --split "$SPLIT"
done

# Leak guard. VRAM_LIMIT_MIB may be raised ONLY for a deliberate, known co-tenant
# (e.g. a FunDiff DiT at ~3 GB); combined use must stay well under 24 GB.
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$vram" -gt "${VRAM_LIMIT_MIB:-2000}" ]; then
  echo "  GPU already holds $vram MiB (limit ${VRAM_LIMIT_MIB:-2000}) — clear it before training (WDDM leak rule)"; exit 1
fi

for s in $SEEDS; do
  d="models_part1_podattn_${SPLIT}_s${s}_$TAG"
  if ls "$d"/*.pt >/dev/null 2>&1; then echo "  SKIP $d (holds checkpoints)"; continue; fi
  step "train_part1_podattn_${SPLIT}_s${s}_$TAG" train_part1.py --split "$SPLIT" \
       --members "$MEMBERS" --seed "$s" --model-dir "$d"
  step "evaluate_part1_podattn_${SPLIT}_s${s}_$TAG" evaluate_part1.py --model-dir "$d"
  step "rom_diagnostics_podattn_${SPLIT}_s${s}_$TAG" rom_diagnostics_part1.py --model-dir "$d"
done

g=models_gp_pod_$SPLIT
if [ ! -f "$g/evaluation_gp.json" ]; then
  step "gp_pod_baseline_$SPLIT" gp_pod_baseline_part1.py --split "$SPLIT" --out-dir "$g"
fi
echo "  CAMPAIGN COMPLETE $(date +%H:%M)"
