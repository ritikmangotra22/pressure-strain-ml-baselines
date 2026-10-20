#!/usr/bin/env bash
# Check 3, stage "retune": independent 100-trial nested re-tuning of the given folds, in parallel.
#   bash scripts/paper2/run_check3_retune.sh LM_Channel_0180 LM_Channel_0550
set -u
cd "$(dirname "$0")/../.."
mkdir -p results/paper2/check3
for f in "$@"; do
  python3 scripts/paper2/check3_r0.py --stage retune --fold "$f" > "results/paper2/check3/log_retune_$f.txt" 2>&1 &
done
wait
