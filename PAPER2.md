# Paper 2: tensor invariance and physical constraints in learned pressure–strain closures

Code, frozen protocol, results and audit for the manuscript *"Separating the Effects of Tensor Invariance and Physical
Constraints in Learned Pressure–Strain Closures: A Controlled A Priori Study in Turbulent Channel Flow"*
(Ritik and Sukanta Ghosh, Department of Computer Applications, Lovely Professional University, Phagwara, India).

Paper 2 builds on Paper 1 (this repository's release v1.0.0, commit `5b2752d`), which supplies the data pipeline,
the processed DNS tables and the metric code. Nothing of Paper 1 is changed.

## What is here

| Path | Contents |
|---|---|
| `configs/paper2/protocol_v1.json` | The protocol, frozen on 17 Oct 2026 before any model was trained |
| `src/ps2/` | Tensors, basis, invariants, models M1–M4 and ablation variants, training and evaluation pipeline |
| `scripts/paper2/run_protocol.py` | Runs the 72 main jobs, the 85 ablations and the pilot |
| `scripts/paper2/analyse_protocol.py` | Verification, statistics (paired hierarchical bootstrap, decision rules) and summary tables |
| `scripts/paper2/make_tables_figures.py` | Regenerates every manuscript table and figure from the result files |
| `scripts/paper2/audit_*.py` | Final audit: code-version reconstruction, independent recomputation, checkpoint reload |
| `tests/paper2/` | 44 tests: architecture, basis, invariants, protocol settings, leakage, determinism |
| `results/paper2/analysis/` | Run manifest, model-by-fold table, effects, consistency, ablations, diagnostics |
| `results/paper2/audit/` | Audit outputs (reproducibility checks, bootstrap-seed sensitivity, full-job replay) |
| `results/paper2/manuscript_assets/` | Tables 1–9 (CSV, Markdown) and Figures 1–5 (PNG, PDF) |
| `results/paper2/raw_data_check/` | Raw-DNS re-verification of 20 Oct 2026 (add this folder from the machine that ran it) |
| `results/paper2/main/`, `ablations/` | Per-job metrics, selected hyperparameters, tuning trials, logs. Predictions and checkpoints are in the Zenodo record |

## Data

The DNS data are not redistributed. Paper 2 reads Paper 1's processed tables, checked on load against SHA-256:

- `results/e1_F1/processed_table.csv` (Lee & Moser, 1691 points): `aa7e316d68aacf5906a8045f627f2e6705c0c89b19fd242155064b3bf6780f11`
- `results/e2_parts/classical_ridge_rf_lgbm/processed_table_hj.csv` (Hoyas & Jiménez, 684 points): `dc97f3350f4fe13a4cdbe437e5e1d3bab767826e75892bd8865dfdf9c54f30ed`

On 20 Oct 2026 both tables were rebuilt from freshly downloaded raw files on a second machine (Apple Silicon, macOS 26.2,
Python 3.13.16). All 20 Paper 1 tests passed; every column Paper 2 uses is identical; only the unused column `III_b`
differs in its last digit (at most 1.1e-16), so the byte hashes of the rebuilt files differ.

## Reproduce

```bash
python3.13 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python -m pytest -q tests/paper2                                     # 44 tests
python scripts/paper2/run_protocol.py --pilot                        # Stage B pilot
python scripts/paper2/run_protocol.py --drain main --max-min 95      # 72 main jobs (about 33 core-hours)
python scripts/paper2/run_protocol.py --drain ablations --max-min 95 # 85 ablation jobs (about 2 core-hours)
python scripts/paper2/analyse_protocol.py                            # statistics -> results/paper2/analysis
python scripts/paper2/make_tables_figures.py --root . --out results/paper2/manuscript_assets
```

Statistics, tables and figures only (no training): unpack the Zenodo archives at the repository root and run the last
two commands. Results are bit-reproducible on the same platform and software versions; on another processor the last
digits of floating-point results can differ. The test `test_paper1_code_reproduces_recorded_run` therefore requires
1e-12 agreement with Paper 1's recorded run only on Linux x86-64 (where it was recorded) and 1e-6 elsewhere; set
`PS2_EXPECT_BITWISE=1` to force the strict check. On an Apple M-series Mac (macOS 26.2) the difference is 2.4e-8.

## AI-assisted preparation

The code, analysis and manuscript draft were prepared with substantial assistance from an AI system (Claude, Anthropic),
under the authors' direction. The authors are responsible for all claims.

## License

Code: MIT (see `LICENSE`). Result files: as stated in the Zenodo record.
