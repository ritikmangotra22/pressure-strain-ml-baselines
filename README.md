# Machine-learning baselines for the pressure–strain tensor in turbulent channel flow

Code, configuration files and results for an a priori benchmark of machine-learning and classical
closures for the four non-zero components of Π_ij/ε in turbulent channel flow.

- **Training data:** Lee & Moser (2015) DNS budgets, Re_τ ≈ 180–5200 (experiment E1: nested leave-one-Re_τ-out).
- **External test data:** Hoyas & Jiménez (2008) budgets, Re_τ ≈ 186–2003 (experiment E2).
- **Models:** ridge regression, random forest, LightGBM, MLP (PyTorch); Rotta + SSG and Rotta + LRR-IP
  (published and refitted coefficients; LRR-IP also with the Gibson–Launder wall-reflection term).

Paper: *[to be added on publication]*.

## 1. Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
The reported results were produced with Python 3.13.16 on Linux (CPU only, one thread per job).
`results/environment.txt` and `results/pip_freeze.txt` record the exact environment.

## 2. Get the DNS data (not redistributed here)

```bash
python scripts/download_lm.py --out data/raw/lee_moser_2015   # 35 files from the Oden Institute server
cd data/raw/lee_moser_2015 && sha256sum -c SHA256SUMS.txt && cd -   # macOS: shasum -a 256 -c
bash scripts/download_hj.sh                                     # 67 files from the UPM server, checksummed
```
Sources: https://turbulence.oden.utexas.edu/channel2015/data/ and
https://torroja.dmt.upm.es/ftp/channels/data/statistics/. Please cite the original data papers.

## 3. Test

```bash
python -m pytest -q tests        # 20 tests: tensor algebra, closures, metrics, leakage, data checks
```

## 4. Reproduce

```bash
# E1 (all models; split into jobs as in results/log_*.txt)
python scripts/run_e1.py --config configs/e1_full.json --models ssg_published,ssg_refit,lrr_published,lrr_refit,lrr_gl_published,lrr_gl_refit,ridge,rf --out results/e1_F1_parts/classical_ridge_rf
python scripts/run_e1.py --config configs/e1_full.json --models lgbm --out results/e1_F1_parts/lgbm
for f in 0180 0550 1000 2000 5200; do
  python scripts/run_e1.py --config configs/e1_mlp100.json --models mlp --folds $f --out results/e1_F1_parts/mlp_$f
done
# E2
python scripts/run_e2.py --config configs/e2.json --models ssg_published,ssg_refit,lrr_published,lrr_refit,lrr_gl_published,lrr_gl_refit,ridge,rf,lgbm --out results/e2_parts/classical_ridge_rf_lgbm
python scripts/run_e2.py --config configs/e2_mlp100.json --models mlp --out results/e2_parts/mlp
# Ablations A1 (F2, F3) and A3
python scripts/run_e1.py --config configs/a1_F2.json --models ridge,rf,lgbm --out results/ablations/a1_F2
python scripts/run_e1.py --config configs/a1_F3.json --models ridge,rf,lgbm --out results/ablations/a1_F3
python scripts/run_e1.py --config configs/a3_uniform.json --models ridge,rf,lgbm --out results/ablations/a3_uniform
# Tables, figures and analyses
python scripts/summarise.py --parts results/e1_F1_parts --out results/e1_F1
python scripts/summarise_e2.py --parts results/e2_parts --out results/e2
python scripts/train_error_and_repro.py --results results/e1_F1     # training errors + exact-refit check
python scripts/ablation_trace.py --results results/e1_F1            # A2 (trace)
python scripts/summarise_ablations.py
python scripts/posthoc_stats.py                                      # unweighted, block length, paired bootstrap
python scripts/tuning_convergence.py
python -I scripts/hj_std_analysis.py
python scripts/compare_mlp_budgets.py
python scripts/make_figures.py --results results/e1_F1 --best mlp,lgbm
python scripts/doc_tables.py
```
Tuning a model on one fold takes minutes (ridge) to about 20–40 minutes (MLP) on a 2-core CPU.

## 5. What is in `results/`

| Path | Contents |
| --- | --- |
| `e1_F1/` | E1 metrics for every model × fold × seed (`metrics_long.csv`), tables, selected hyperparameters, test predictions, figures |
| `e2/` | E2 metrics, tables and predictions on the Hoyas & Jiménez cases |
| `e1_F1_parts/`, `e2_parts/` | The per-job outputs that the summaries merge (predictions omitted; identical copies are in `e1_F1/` and `e2/`) |
| `ablations/` | A1 (F2, F3) and A3 metrics and tables |
| `posthoc/` | Unweighted errors, bootstrap block-length sensitivity, paired bootstrap comparisons |
| `mlp40_original/` | The first MLP run with 40 tuning trials (superseded by the 100-trial run; kept for comparison) |
| `mac_mlp100/` | Replicate of the 100-trial MLP tuning on a MacBook Air M4 (not used for reported results) |
| `superseded_wallpoint_bug/` | Results from before the wall-point correction (do not use; kept for transparency) |
| `PROTOCOL_AND_DEVIATIONS.md` | Registered protocol, deviations and corrections |
| `verification_classical.md`, `trace_investigation.md`, `hj_std_trace.md` | Verification notes |
| `log_*.txt` | Job logs, including every tuning trial |

## License and citation

Code: MIT (see `LICENSE`). DNS data belong to their original providers. Citation metadata: `CITATION.cff`.
