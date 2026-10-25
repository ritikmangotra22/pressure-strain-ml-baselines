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
4. 23 Oct 2026 — the secondary repeat on Paper 2's interpolation folds is dropped. Three ablations are added on Arm A (5 seeds × 200 evaluations each, run after the primary jobs): NSGA-II without f_gen (A2), NSGA-II without f_phys (A4), and an equal-weight sum of the M0-normalised objectives optimised by TPE (A3).
5. 23 Oct 2026 — Stage 2/3 redesign. The frozen 1D Reynolds-stress solve is verified by an anchored oracle (all five Lee & Moser cases, stresses within 1.0–3.5 % of the DNS), but it is ill-conditioned for learned closures: the first Newton step amplifies the most accurate candidate's a priori error about 15,000-fold, against 1–3 for the published classical closures. Solver-in-the-loop NSGA-II on J_U and J_R is replaced by NSGA-II with a fifth objective, f_cond (log₁₀ amplification on the held-out inner case); a posteriori solves are attempted for every selected model.
6. 23 Oct 2026 — Stage 3 implementation fixed before any selected model was refitted or assessed: refit and classical-reference scripts; a posteriori solves on refit seeds 0–2 (≤ 200 iterations), amplification on all 10 seeds; frozen-solve outcomes are interpreted on a case only if the anchored oracle converges there (it does not on MKM 180, whose DNS budgets close only to 13.6 % of ε); solver grids ending on the centreline (MKM); guards on the single confirmatory run.
7. 25 Oct 2026 — execution only: the a posteriori stage on LM moved to the author's MacBook; a cross-machine recheck found one failing solve labelled differently (diverged vs not converged), so the whole stage was rerun on the MacBook alone; the mixed-host file is kept as superseded. Analyses report converged against failed.
8. 25 Oct 2026 — execution only: the confirmatory test runs in parallel on the MacBook (identical results to the serial version in a dry run).

## Approvals

- Author (Ritik), 21 Oct 2026: all settings, including the scope exclusions, feasibility thresholds,
  qNEHVI reference point and return-to-isotropy floor (10⁻⁴).
- Supervisor (Sukanta Ghosh): **approved 25 Oct 2026** (scope, solver-level measures, deviations 4–7), as reported by the author; recorded in `protocol_v1.json`.
- Previously: pending. To be recorded in `protocol_v1.json` → `approvals.supervisor`
  with the date. Scope items needing sign-off: no geometry or flow-regime transfer; CFD-level objectives
  J_U and J_R only; deviations 4–6.

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
python scripts/paper3/analyse_search.py                    # inner objectives only
python scripts/paper3/figures_search.py                    # hypervolume, selected models, RQ4 cost table
python scripts/paper3/classical_refs.py                    # SSG / LRR-IP / LRR-IP+GL, published and refitted
python scripts/paper3/refit_selected.py --workers 2        # 56 models (55 jobs + M0) x 10 seeds on all LM
python scripts/paper3/aposteriori.py --workers 2           # conditioning and frozen solves on LM
git tag paper3-selection-v1                                # after supervisor approval is recorded
python scripts/paper3/confirmatory_test.py --check         # preconditions only
python scripts/paper3/confirmatory_test.py                 # ONCE
python scripts/paper3/analyse_confirmatory.py              # bootstrap comparisons, consistency rule
```

Searches run in sessions of at most 100 minutes; each job resumes from its Optuna SQLite study
(`results/paper3/search/<job>/study.db`). Resume points and sampler seeds are logged in
`sessions.jsonl`; trials interrupted by a killed session are excluded from the budget and listed in
`status.json`.

## Status

| Stage | Status |
|---|---|
| 0. Setup, pilot, M0, MKM data | done (20–21 Oct 2026) |
| 1. Primary searches (35 jobs) | done (25 Oct 2026) |
| 2. Ablations A2–A4 (15 jobs); classical references | done (25 Oct 2026) |
| 3. Refit, classical references, a posteriori, single confirmatory test | refits (560) and a posteriori on LM done (25 Oct; a posteriori on one machine, deviation 7); confirmatory test waits for supervisor approval and the paper3-selection-v1 tag |
| 4. Stage 2 solver and conditioning diagnostic | done (verified); conditioning-aware NSGA-II (5 jobs) done |

## AI-assisted preparation

The code and analysis are prepared with substantial assistance from an AI system (Claude, Anthropic),
under the authors' direction. The authors are responsible for all claims.
