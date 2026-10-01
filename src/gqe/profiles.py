"""Compute-budget profiles for the Ising comparison.

Keep ``n_iters × group_size × shots`` well below ``2^n``: once the shot budget
covers the state space, any sampler (including uniform random) finds the
ground state and the comparison says nothing.
"""

from __future__ import annotations

from typing import Any

_SHARED: dict[str, Any] = {
    "seq_len": 20,
    "d_latent": 12,
    "cvar_alpha": 0.1,
    "weight_decay": 0.01,
    "clip_eps": 0.2,
    "temperature": 1.0,
    "repetition_penalty": 1.2,
    "lr": 2e-3,
    "qaoa_p": 2,
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        **_SHARED,
        "n": 6,
        "n_instances": 1,
        "n_seeds": 1,
        "n_iters": 2,
        "group_size": 2,
        "shots": 16,
        "d_model": 32,
        "d_ff": 128,
        "n_layers": 2,
        "n_heads": 4,
        "policy_updates": 2,
        "qaoa_maxiter": 30,
        "sa_sweeps": 50,
    },
    "laptop": {
        **_SHARED,
        "n": 14,
        "n_instances": 3,
        "n_seeds": 2,
        "n_iters": 25,
        "group_size": 8,
        "shots": 32,
        "d_model": 32,
        "d_ff": 128,
        "n_layers": 2,
        "n_heads": 4,
        "policy_updates": 10,
        "qaoa_maxiter": 200,
        "sa_sweeps": 500,
    },
    "nano": {
        **_SHARED,
        "n": 16,
        "n_instances": 5,
        "n_seeds": 3,
        "n_iters": 60,
        "group_size": 8,
        "shots": 64,
        "d_model": 64,
        "d_ff": 256,
        "n_layers": 4,
        "n_heads": 4,
        "policy_updates": 10,
        "qaoa_maxiter": 400,
        "sa_sweeps": 1000,
    },
}


def get_profile(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Choose one of: {', '.join(sorted(PROFILES))}")
    return dict(PROFILES[key])


def train_config(profile: dict[str, Any], **overrides):
    from gqe.train import TrainConfig

    fields = TrainConfig.__dataclass_fields__
    kwargs = {k: v for k, v in profile.items() if k in fields}
    kwargs.update(overrides)
    return TrainConfig(**kwargs)
