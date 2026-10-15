#!/bin/bash
# 100-trial MLP retuning (E1 five folds + E2) on a MacBook Air M4: 4 parallel lanes = 4 performance cores.
# CPU only, one thread per job. Logs: results/log_mlp100_*.txt ; progress: results/log_mlp100_status.txt
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p results/e1_F1_parts_mlp100 results/e2_parts_mlp100
rm -f results/log_mlp100_status.txt
python -c "import platform,sys,numpy,sklearn,torch,optuna,lightgbm,pandas;print('python',sys.version.split()[0]);print('platform',platform.platform(),platform.processor());[print(m.__name__,m.__version__) for m in (numpy,pandas,sklearn,lightgbm,optuna,torch)]" > results/environment_mac.txt
pip freeze > results/pip_freeze_mac.txt
job() {
  if [ "$1" = e2 ]; then python scripts/run_e2.py --config configs/e2_mlp100.json --models mlp --out results/e2_parts_mlp100/mlp > results/log_mlp100_e2.txt 2>&1
  else python scripts/run_e1.py --config configs/e1_mlp100.json --models mlp --folds $1 --out results/e1_F1_parts_mlp100/mlp_$1 > results/log_mlp100_$1.txt 2>&1; fi
  echo "$(date '+%H:%M:%S') $1 finished, exit code $?" >> results/log_mlp100_status.txt
}
( job 0180; job 0550 ) &
( job 1000; job 2000 ) &
( job 5200 ) &
( job e2 ) &
wait
echo "$(date '+%H:%M:%S') ALL_DONE" >> results/log_mlp100_status.txt
