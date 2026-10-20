# Paper 2 — Section H checks 1–3 (pre-training verification)

These outputs verify the Paper 2 design before any of M1–M4 is trained. Nothing in the Paper 1
release was modified: Paper 2 code lives in `src/ps2/`, `scripts/paper2/`, `configs/paper2/`
and `tests/paper2/`, and reads the processed DNS tables that Paper 1 released.

**Data substitution.** The raw DNS files could not be downloaded in the environment where the
checks were run (both data servers were refused by the network policy), so 9 of Paper 1's 20
data tests could not run there. Paper 2 reads `results/e1_F1/processed_table.csv` (Lee & Moser,
1691 points) and `results/e2_parts/classical_ridge_rf_lgbm/processed_table_hj.csv`
(Hoyas & Jiménez, 684 points), which Paper 1 built from checksum-verified raw files. Their
SHA-256 hashes are fixed in `src/ps2/data.py`, and loading fails if they change.

## Reproduce

```bash
pip install -r requirements.txt                      # Python 3.13.16; see environment.txt
python -m pytest -q tests/paper2                     # 38 tests, under 1 min

# Check 1: architectural properties of untrained M1-M4 on real DNS states (about 1 min)
python scripts/paper2/check1_architecture.py --config configs/paper2/check1.json

# Check 2: basis adequacy, conditioning and invariant uniqueness (about 3 min)
python scripts/paper2/check2_basis_invariants.py --config configs/paper2/check2.json

# Check 3: reproduction R0 of Paper 1's MLP through the Paper 2 trainer
python scripts/paper2/check3_r0.py --stage env        # Paper 1 code, this environment
python scripts/paper2/check3_r0.py --stage p2_f32     # Paper 2 trainer, float32
python scripts/paper2/check3_r0.py --stage p2_f64     # float64, layers built in float64
python scripts/paper2/check3_r0.py --stage p2_f64c    # float64, layers built in float32 and cast
python scripts/paper2/check3_precision_crosscheck.py  # Paper 1 code switched to float64
bash scripts/paper2/run_check3_retune.sh LM_Channel_0180 LM_Channel_0550   # ~30-60 min per pair
bash scripts/paper2/run_check3_retune.sh LM_Channel_1000 LM_Channel_2000
bash scripts/paper2/run_check3_retune.sh LM_Channel_5200
python scripts/paper2/check3_r0.py --stage summarise
python scripts/paper2/check3_selected_config_crosscheck.py   # Paper 1 code on the re-tuned 5200 config
```

## Outcome (17 October 2026)

- Check 1: pass (8/8 registered criteria).
- Check 2: all Section E.2 numbers reproduced; the E.3 rank claim fails at the 4 HJ centreline points
  (rank 2, exact representation); the E.4 invariant set is unique on all DNS states but not globally.
- Check 3: the code path reproduces Paper 1 (float32 within 0.0014; float64 with cast initialisation
  within 0.002). Registered 0.02 criterion fails at Re_tau = 5200 for native-float64 initialisation
  (0.0208, initial-weight draw) and for the independent re-tuning (0.064, different selected
  configuration; reproduced exactly by Paper 1's code).

## Design decisions (protocol v1, no training)

```bash
python scripts/paper2/check4_invariant_set.py --config configs/paper2/check4.json   # decision 1, ~7 min
python scripts/paper2/decision2_precision.py --config configs/paper2/decision2.json # decisions 2 and 4
```

The frozen protocol is `configs/paper2/protocol_v1.json` (checked by `tests/paper2/test_protocol_v1.py`).

## Files

| Path | Contents |
| --- | --- |
| `environment.txt`, `pip_freeze.txt` | Software and hardware used for these checks |
| `log_check1.txt`, `check1/check1_architecture.csv`, `check1/check1_summary.json` | Check 1 per-run diagnostics and verdict |
| `log_check2.txt`, `check2/check2_adequacy_summary.csv` | Check 2 Part A by database, time scale, C_T and region |
| `check2/check2_adequacy_points.csv` | Check 2 Part A per point (rank, coefficient norm, residuals) |
| `check2/check2_results.json` | Check 2 Part B (swap identity, twin search, degenerate set) and spanning subsets |
| `check3/r0_*.csv`, `check3/log_*.txt` | Check 3 per-run results for every stage |
| `check3/r0_retune_*_params.json` | Configurations selected by the independent re-tuning |
| `check3/check3_summary.json`, `check3/check3_fold_means.csv` | Check 3 verdicts, fold means and seed-mean tests |
| `decision1/check4_invariant_set.json`, `log_decision1.txt` | Decision 1: invariant definitions, units, transformations, uniqueness, conditioning |
| `decision2/*.csv`, `decision2/decision2_summary.json`, `log_decision2.txt` | Decisions 2 and 4: initialisation correspondence, precision of property tests, per-epoch timings (forward + backward only) |

## Protocol v1.0 execution (Stages A–E): reproduction commands

Run from the repository root on branch `paper2-section-h-checks`. Each worker is single-threaded; two workers were used.

```bash
python3 -m pytest tests/paper2 -q                                   # 44 tests (readiness gate)
python3 scripts/paper2/run_protocol.py --pilot                       # Stage B pilot -> results/paper2/pilot
python3 scripts/paper2/run_protocol.py --drain main --max-min 95     # Stage C main, 72 jobs (start one per worker)
python3 scripts/paper2/run_protocol.py --drain ablations --max-min 95  # Stage C ablations, 85 jobs
python3 scripts/paper2/run_protocol.py --list main                   # progress: 72/72
python3 scripts/paper2/run_protocol.py --list ablations              # progress: 85/85
python3 scripts/paper2/run_protocol.py --job M4__run2__LM_Channel_1000   # rerun a single job
python3 scripts/paper2/analyse_protocol.py                           # Stage D/E -> results/paper2/analysis
python3 scripts/paper2/check_code_version_reproduction.py            # refit seed 0 of jobs across runner hashes
```

Retry policy (model-independent): a job that is killed or interrupted is moved to `results/paper2/failed_runs/` and rerun from scratch with identical settings; a refit that returns non-finite values is retried once; a tuning trial with non-finite loss is scored +inf.

Outputs in `results/paper2/analysis/`: `run_manifest.csv`, `verification.csv`, `model_by_fold.csv`, `effects.csv`, `consistency.csv`, `ablations.csv`, `regions_and_diagnostics.csv`, `structural_max_by_model.csv`, `code_version_reproduction.json`, `summary.json`.

## Final audit (18 Oct 2026): commands and dependencies

The Paper 2 archives are not self-contained. They must be unpacked on top of a checkout of this
repository at commit `5b2752d5ccbc7e714eee9b79531ddd4a33513196` (Paper 1 release). That commit supplies:
`src/ps` (imported by `src/ps2/pipeline.py` for Paper 1's metric code), `results/e1_F1/processed_table.csv`
(SHA-256 aa7e316d…) and `results/e2_parts/classical_ridge_rf_lgbm/processed_table_hj.csv` (SHA-256 dc97f335…).
Unpack both archives (results and checkpoints) at the repository root.

```bash
python3 scripts/paper2/audit_code_versions.py                     # rebuilds the three runner hashes byte for byte
python3 scripts/paper2/analyse_protocol.py --out /tmp/reanalysis  # must equal results/paper2/analysis byte for byte
python3 scripts/paper2/audit_independent_checks.py --root . --data-root . --out results/paper2/audit/independent
python3 scripts/paper2/audit_supplementary.py --root . --out results/paper2/audit/supplementary
python3 scripts/paper2/make_tables_figures.py --root . --out results/paper2/manuscript_assets
python3 scripts/paper2/run_protocol.py --job M4__run1__LM_Channel_5200   # in a scratch copy: full-job replay
```

Audit outputs: `results/paper2/audit/` (code_versions/, independent/, supplementary/, bootstrap_seed_sensitivity.csv,
full_job_replay.json). Manuscript tables and figures: `results/paper2/manuscript_assets/`.
