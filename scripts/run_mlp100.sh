#!/bin/bash
# Two parallel lanes on the 2-core machine; each job logs to results/log_mlp100_*.txt
cd "$(dirname "$0")/.."
lane() { for f in "$@"; do
  if [ "$f" = e2 ]; then python scripts/run_e2.py --config configs/e2_mlp100.json --models mlp --out results/e2_parts_mlp100/mlp > results/log_mlp100_e2.txt 2>&1
  else python scripts/run_e1.py --config configs/e1_mlp100.json --models mlp --folds $f --out results/e1_F1_parts_mlp100/mlp_$f > results/log_mlp100_$f.txt 2>&1; fi
  echo "$f exit $?" >> results/log_mlp100_status.txt; done; }
lane 0180 0550 1000 &
lane 2000 5200 e2 &
wait
echo ALL_DONE >> results/log_mlp100_status.txt
