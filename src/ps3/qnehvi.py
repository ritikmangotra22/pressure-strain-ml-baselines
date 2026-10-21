"""qNEHVI candidates function for Paper 3 (protocol_v1 deviation 1, 21 Oct 2026).

Identical to optuna_integration 5.0.0 `qnehvi_candidates_func` (no constraints branch is needed:
failed fits are FAIL trials and are not passed to the sampler) except for the acquisition-
optimisation budget, reduced because the exact 4-objective computation made each proposal take
about 219 s by trial 65 and was projected to reach about 20 min by trial 200:

    num_restarts 20 -> 5, raw_samples 1024 -> 256, maxiter 200 -> 100

As in Optuna, the acquisition is BoTorch's qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI,
Ament et al. 2023), the numerically stable form of qNEHVI (Daulton et al. 2021); exact box
decomposition (alpha = 0) for 4 objectives; prune_baseline=True; 256 Sobol QMC samples;
reference point = observed worst value of each objective minus 1e-8 (Optuna's default).
"""
from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

ACQ_OPT = {"num_restarts": 5, "raw_samples": 256, "maxiter": 100, "batch_limit": 5}


def candidates_func(train_x, train_obj, train_con, bounds, pending_x):
    if train_con is not None:
        raise ValueError("constraints are not used in Paper 3")
    torch.set_num_threads(1)
    n_obj = train_obj.size(-1)
    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=n_obj))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    alpha = 10 ** (-8 + n_obj) if n_obj > 4 else 0.0
    ref_point = (train_obj.min(dim=0).values - 1e-8).tolist()
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=ref_point, X_baseline=train_x, alpha=alpha, prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256])), X_pending=pending_x)
    std_bounds = torch.zeros_like(bounds)
    std_bounds[1] = 1
    cand, _ = optimize_acqf(acq_function=acqf, bounds=std_bounds, q=1,
                            num_restarts=ACQ_OPT["num_restarts"], raw_samples=ACQ_OPT["raw_samples"],
                            options={"batch_limit": ACQ_OPT["batch_limit"], "maxiter": ACQ_OPT["maxiter"],
                                     "nonnegative": True}, sequential=True)
    return unnormalize(cand.detach(), bounds=bounds)
