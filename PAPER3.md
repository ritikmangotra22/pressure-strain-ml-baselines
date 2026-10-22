# Paper 3: multi-objective selection of learned pressure–strain closures

Working title: *Selecting Learned Pressure–Strain Closures under Tuning Uncertainty: A Multi-Objective
Comparison of Accuracy, Physical Consistency, Reynolds-Number Transfer and Cost in Turbulent Channel Flow*
(Ritik and Sukanta Ghosh). PhD Research Objective 3.

Paper 3 builds on Paper 1 (`v1.0.0`) and Paper 2 (`paper2-v1.0`). Their code in `src/ps` and
`src/ps2` is imported and never modified; Paper 3 code is in `src/ps3`, `scripts/paper3`, `tests/paper3`.

## Frozen configuration

`configs/paper3/protocol_v1.json`, frozen 21 Oct 2026 before any main search run. It records the data
and hashes, search space, objectives, strategies, selection rule, statistics, execution rules and
approvals. Any later change is added to its `deviations` list with date and reason.

| Item | Setting |
|---|---|
| Training and tuning data | Lee & Moser Re_τ = 180, 550, 1000, 2000, 5200 (Paper 1 table, SHA-256 checked) |
| Comparability test | Hoyas & Jiménez (previously inspected in Paper 2) |
| Confirmatory test | Moser, Kim & Mansour Re_τ = 395, 590; hash-frozen; read only by `confirmatory_test.py` (`tests/paper3/test_mkm_guard.py`) |
| Arms | A: M4 only; B: M1–M4 searched |
| Strategies | random, single-objective TPE (Arm A), NSGA-II (population 40), qNEHVI (qLogNEHVI, 20 random start-up points shared with random search and NSGA-II, deviation 3; reduced acquisition budget, deviation 1) |
| Budget | 200 evaluations per job, 5 optimizer seeds (3001–3005), 35 primary jobs |
| Objectives | f_err (mean inner E_Π), f_gen (max inner E_Π relative to M0), f_phys (Arm A: return-to-isotropy and coefficient norm vs M0; Arm B: trace + rotated-frame excess), f_cost (FLOPs) |
| Reference M0 | Paper 2 run-1 M4 (E2 tuning); seed 0 reproduces Paper 2's inner error exactly |
| Seeds per candidate | 1 (pilot: seed SD is 1–3 % of the spread between configurations) |

## Deviations

1. 21 Oct 2026 — qNEHVI acquisition-optimisation budget reduced (5 restarts, 256 raw samples, 100 iterations) because its overhead grew to about 219 s per proposal by trial 65 (projected 20 min by trial 200). The started job is rerun; its 69 trials are kept in `results/paper3/search_superseded/` as RQ4 evidence. Details in `protocol_v1.json` → `deviations`.

2. 22 Oct 2026 — hypervolume normalisation and selection tie tolerance defined before any strategy comparison (clarification).
3. 22 Oct 2026 — qNEHVI start-up trials are uniform random with the job seed, not Sobol as written (an Optuna fallback); kept, so random search, NSGA-II and qNEHVI share a common 20-point initial design.

## Approvals

- Author (Ritik), 21 Oct 2026: all settings, including the scope exclusions, feasibility thresholds,
  qNEHVI reference point and return-to-isotropy floor (10⁻⁴).
- Supervisor (Sukanta Ghosh): **pending**. To be recorded in `protocol_v1.json` → `approvals.supervisor`
  with the date. Scope items needing sign-off: no geometry or flow-regime transfer; CFD-level objectives
  J_U and J_R only.

## Reproduce

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt botorch==0.18.1 optuna-integration==5.0.0
# raw data: Lee & Moser and Hoyas & Jimenez as in README.md; Moser-Kim-Mansour:
curl -L -o data/raw/moser_kim_mansour_1999/chandata.tar.gz https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz
(cd data/raw/moser_kim_mansour_1999 && tar xzf chandata.tar.gz README chan{180,395,590}/{balances,profiles} && sha256sum -c SHA256SUMS.txt)
python -m pytest -q tests                                  # Papers 1-3
python scripts/paper3/build_mkm_tables.py                  # rebuild and validate MKM tables
python scripts/paper3/pilot.py --part 0 --nparts 2 &        # timing pilot (Stage 0)
python scripts/paper3/pilot.py --part 1 --nparts 2; wait
python scripts/paper3/reference_m0.py                      # M0 reference
python scripts/paper3/run_queue.py --workers 2 --session-minutes 100   # repeat until 35/35 complete
python scripts/paper3/run_queue.py --list
```

Searches run in sessions of at most 100 minutes; each job resumes from its Optuna SQLite study
(`results/paper3/search/<job>/study.db`). Resume points and sampler seeds are logged in
`sessions.jsonl`; trials interrupted by a killed session are excluded from the budget and listed in
`status.json`.

## Status

| Stage | Status |
|---|---|
| 0. Setup, pilot, M0, MKM data | done (20–21 Oct 2026) |
| 1. Primary searches (35 jobs) | running |
| 2. Secondary E1 folds, ablations A2–A4, classical references | not started |
| 3. Refit and single confirmatory test | not started |
| 4. A posteriori 1D RSM solve; solver-in-the-loop NSGA-II | not started |

## AI-assisted preparation

The code and analysis are prepared with substantial assistance from an AI system (Claude, Anthropic),
under the authors' direction. The authors are responsible for all claims.
