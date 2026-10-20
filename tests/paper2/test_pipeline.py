"""Readiness-gate tests for the protocol v1 training pipeline (leakage, settings, determinism)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
from ps2 import models, pipeline as PL  # noqa: E402
import run_protocol as RP  # noqa: E402

PROTO = json.loads((ROOT / "configs/paper2/protocol_v1.json").read_text())


@pytest.fixture(scope="module")
def S():
    return RP.splits()


def test_runner_settings_match_protocol():
    assert RP.TUNER_SEEDS == PROTO["tuning"]["tuner_sampler_seeds"]
    assert RP.SEEDS == PROTO["training"]["seeds_final_refit"]
    assert RP.N_TRIALS == PROTO["tuning"]["trials_per_run"] == 100
    assert RP.E1 == PROTO["folds"]["E1"]
    assert (PL.MAX_EPOCHS, PL.PATIENCE) == (PROTO["training"]["early_stopping_inner"]["max_epochs"],
                                            PROTO["training"]["early_stopping_inner"]["patience"])
    assert len(RP.main_jobs()) == 4 * 3 * 6
    assert len(RP.ablation_jobs()) == 5 * (6 + 4 + 1 + 4 + 1 + 1)


def test_search_space_is_common_to_all_models():
    import optuna
    spaces = []
    for _ in range(2):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        t = study.ask()
        PL.suggest(t)
        spaces.append(t.distributions)
    assert spaces[0] == spaces[1]
    p = PROTO["tuning"]["search_space"]
    d = spaces[0]
    assert (d["n_layers"].low, d["n_layers"].high) == tuple(p["n_layers"])
    assert list(d["width"].choices) == p["width"] and list(d["activation"].choices) == p["activation"]
    assert (d["lr"].low, d["lr"].high, d["lr"].log) == (p["lr"][0], p["lr"][1], True)


def test_tuning_never_sees_the_held_out_case(S, monkeypatch):
    """Spy on every fit inside tune(): the outer held-out case must never be trained or validated on."""
    seen = []
    real_fit = PL.fit

    def spy(name, train, params, seed, val=None, **kw):
        seen.append((set(train.df.case), set(val.df.case) if val is not None else set()))
        return real_fit(name, train, params, seed, val=val, **kw)

    monkeypatch.setattr(PL, "fit", spy)
    monkeypatch.setattr(PL, "MAX_EPOCHS", 5)
    outer = "LM_Channel_0550"
    train_cases = {c: S[c] for c in RP.E1 if c != outer}
    PL.tune("M4", train_cases, 2026, n_trials=1)
    assert len(seen) == 4
    for tr, va in seen:
        assert outer not in tr and outer not in va and not (tr & va)


def test_scalers_and_target_scale_use_training_data_only(S):
    train = PL.concat([S[c] for c in RP.E1 if c != "LM_Channel_5200"])
    m = models.build_any("M1", train.b, train.G, train.re, seed=0, init="float32_cast")
    x = models.cartesian_inputs(train.b, train.G, train.re).numpy()
    assert np.allclose(m.scaler.mu.numpy(), x.mean(0))
    allc = PL.concat([S[c] for c in RP.E1])
    assert PL.target_scale(train) != PL.target_scale(allc)
    assert "LM_Channel_5200" not in set(train.df.case)


def test_fit_is_deterministic_and_respects_precision(S, monkeypatch):
    monkeypatch.setattr(PL, "MAX_EPOCHS", 30)
    train = PL.concat([S[c] for c in RP.E1 if c != "LM_Channel_1000"])
    p = {"n_layers": 2, "width": 32, "activation": "silu", "lr": 1e-3, "weight_decay": 1e-5}
    outs = []
    for _ in range(2):
        f, info, _ = PL.fit("M4", train, p, seed=3, epochs=30)
        assert next(f.model.parameters()).dtype == torch.float32 and info["status"] == "ok"
        outs.append(f.predict(S["LM_Channel_1000"]))
    assert np.array_equal(outs[0], outs[1])


def test_evaluate_reports_exact_properties_for_m4(S, monkeypatch):
    train = PL.concat([S[c] for c in RP.E1 if c != "LM_Channel_2000"])
    p = {"n_layers": 2, "width": 32, "activation": "tanh", "lr": 1e-3, "weight_decay": 1e-5}
    f, _, _ = PL.fit("M4", train, p, seed=0, epochs=20)
    met, _ = PL.evaluate(f, S["LM_Channel_2000"])
    tol = PROTO["tolerances_exact"]
    assert met["eq_max"] <= tol["equivariance_rel"] and met["trace_rel_max"] <= tol["trace_rel"]
    assert met["iso_limit_max"] <= tol["isotropic_limit_abs"] and met["asym_max"] <= tol["symmetry_abs"]
    assert met["f32_f64_pred_rel_max"] <= 1e-4
    assert abs(met["E_Pi"] - met["p1_E_Pi"]) < 1e-12        # M4 predicts no 13/23 components in channel flow
