"""Compute-budget profiles: laptop demo, Sol-friendly nano, paper-adjacent stretch.

Physical setup is identical across profiles (linear H₄, 6-31G, (4e,4o), L=20).
Only model size, shots, iterations, and seed count change.
"""

from __future__ import annotations

from typing import Any

# Shared chemistry / QSCI knobs (paper §V-A).
_CHEMISTRY: dict[str, Any] = {
    "basis": "6-31g",
    "seq_len": 20,
    "d_max": 2000,
    "d_latent": 12,
    "r_min": 0.8,
    "r_max": 2.2,
    "op_angle": 0.5 * 3.141592653589793,
    "weight_decay": 0.01,
    "clip_eps": 0.2,
    "temperature": 1.0,
    "repetition_penalty": 1.2,
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        **_CHEMISTRY,
        "n_iters": 2,
        "group_size": 2,
        "shots": 128,
        "d_model": 32,
        "d_ff": 128,
        "n_layers": 2,
        "n_heads": 4,
        "lr": 5e-5,
        "policy_updates": 2,
        "n_points": 2,
        "n_seeds": 1,
        "vqe_maxiter": 80,
    },
    "laptop": {
        **_CHEMISTRY,
        "n_iters": 20,
        "group_size": 4,
        "shots": 512,
        "d_model": 32,
        "d_ff": 128,
        "n_layers": 2,
        "n_heads": 4,
        "lr": 5e-5,
        "policy_updates": 10,
        "n_points": 5,
        "n_seeds": 1,
        "vqe_maxiter": 300,
    },
    "nano": {
        **_CHEMISTRY,
        "n_iters": 70,
        "group_size": 8,
        "shots": 8192,
        "d_model": 64,
        "d_ff": 256,
        "n_layers": 4,
        "n_heads": 4,
        "lr": 5e-5,
        "policy_updates": 10,
        "n_points": 7,
        "n_seeds": 3,
        "vqe_maxiter": 1000,
    },
    "paperish": {
        **_CHEMISTRY,
        "n_iters": 100,
        "group_size": 10,
        "shots": 50_000,
        "d_model": 64,
        "d_ff": 256,
        "n_layers": 4,
        "n_heads": 4,
        "lr": 5e-6,
        "policy_updates": 30,
        "n_points": 9,
        "n_seeds": 3,
        "vqe_maxiter": 2000,
    },
}


def get_profile(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown profile {name!r}. Choose one of: {known}")
    return dict(PROFILES[key])
