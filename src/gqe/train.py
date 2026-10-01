"""GRPO training loop for one H₄ geometry (GQKAE / QSCI)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from gqe.grpo import grpo_loss, standardize_rewards, token_log_probs
from gqe.model import (
    count_params,
    init_transformer,
    sample_sequences,
    transformer_logits,
)
from gqe.molecule import H4System
from gqe.operators import build_uccsd_pool
from gqe.qsci import build_full_cas_hamiltonian, evaluate_sequence


BOS_ID = 0  # identity token doubles as BOS context starter in the pool


@dataclass
class TrainConfig:
    """GRPO training knobs for one geometry.

    Paper (H₄, arXiv:2605.04604 §V-A) used GPT-2-scale HQKANsformer, L=20,
    M=10, 1e5 shots, 100 iters, 5 seeds, AdamW 5e-6 / 30 policy updates,
    CUDA-Q. This demo keeps the same chemistry and loop, with much smaller
    models and shot budgets (see ``gqe.profiles``).
    """

    bond_length: float = 1.0
    basis: str = "6-31g"
    n_iters: int = 40
    group_size: int = 8
    seq_len: int = 20  # paper L=20 for H4
    shots: int = 2048
    d_max: int = 2000
    d_model: int = 32
    d_latent: int = 12
    d_ff: int = 128
    n_layers: int = 2
    n_heads: int = 4
    backbone: Literal["gqkae", "gqe"] = "gqkae"
    lr: float = 5e-5
    weight_decay: float = 0.01
    policy_updates: int = 10
    clip_eps: float = 0.2
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    seed: int = 0
    out_dir: str = "runs/h4"
    # pool angle (fixed); π/2 is a common discrete GQE choice
    op_angle: float = 0.5 * float(np.pi)


@dataclass
class TrainResult:
    best_energy: float
    casci_energy: float
    hf_energy: float
    history: list[dict] = field(default_factory=list)
    best_tokens: list[int] = field(default_factory=list)
    n_params: int = 0
    vocab_size: int = 0
    backbone: str = "gqkae"


def train_geometry(cfg: TrainConfig | None = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    system = H4System(bond_length=cfg.bond_length, basis=cfg.basis)
    pool = build_uccsd_pool(system, angle=cfg.op_angle, include_identity=True)
    vocab_size = len(pool)
    H_full, na, nb = build_full_cas_hamiltonian(system.hamiltonian)
    casci = system.casci_energy()
    hf = system.hf_energy

    key = jax.random.PRNGKey(cfg.seed)
    key, k_init = jax.random.split(key)
    params = init_transformer(
        k_init,
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.seq_len,
        d_latent=cfg.d_latent,
        backbone=cfg.backbone,
        d_ff=cfg.d_ff,
    )
    n_params = count_params(params)
    optimizer = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(params)

    best_energy = float("inf")
    best_tokens: list[int] = []
    history: list[dict] = []

    def loss_fn(p, tokens, old_lp, advantages):
        logits = transformer_logits(tokens, p)
        new_lp = token_log_probs(logits, tokens, temperature=cfg.temperature)
        return grpo_loss(new_lp, old_lp, advantages, clip_eps=cfg.clip_eps)

    @jax.jit
    def old_log_probs(p, tokens):
        return token_log_probs(transformer_logits(tokens, p), tokens, temperature=cfg.temperature)

    @jax.jit
    def update_step(p, opt_st, tokens, old_lp, advantages):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, tokens, old_lp, advantages
        )
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss, metrics

    for it in trange(cfg.n_iters, desc=f"H4 {cfg.backbone} R={cfg.bond_length:.2f}"):
        key, k_sample = jax.random.split(key)
        # tokens include BOS=identity at position 0; generate seq_len operators
        tokens = sample_sequences(
            k_sample,
            params,
            bos_id=BOS_ID,
            length=cfg.seq_len,
            batch=cfg.group_size,
            temperature=cfg.temperature,
            repetition_penalty=cfg.repetition_penalty,
        )
        tokens_np = np.asarray(tokens)

        energies = []
        for m in range(cfg.group_size):
            # evaluate operator sequence after BOS
            seq = tokens_np[m, 1:].tolist()
            res = evaluate_sequence(
                system,
                pool,
                seq,
                shots=cfg.shots,
                d_max=cfg.d_max,
                seed=cfg.seed + it * 1000 + m,
                H_full=H_full,
                na=na,
                nb=nb,
            )
            energies.append(res.energy)
            if res.energy < best_energy:
                best_energy = res.energy
                best_tokens = seq

        energies_arr = jnp.asarray(energies, dtype=jnp.float32)
        rewards = -energies_arr
        advantages = standardize_rewards(rewards)

        old_lp = jax.lax.stop_gradient(old_log_probs(params, tokens))

        last_loss = 0.0
        for _ in range(cfg.policy_updates):
            params, opt_state, loss, _metrics = update_step(
                params, opt_state, tokens, old_lp, advantages
            )
            last_loss = float(loss)

        row = {
            "iter": it,
            "mean_E": float(np.mean(energies)),
            "min_E": float(np.min(energies)),
            "best_so_far": best_energy,
            "casci": casci,
            "error_mHa": 1e3 * (best_energy - casci),
            "loss": last_loss,
        }
        history.append(row)

    result = TrainResult(
        best_energy=best_energy,
        casci_energy=casci,
        hf_energy=hf,
        history=history,
        best_tokens=best_tokens,
        n_params=n_params,
        vocab_size=vocab_size,
        backbone=cfg.backbone,
    )
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out / "solution.json", "w") as f:
        json.dump(
            {
                "bond_length": cfg.bond_length,
                "best_energy": best_energy,
                "casci": casci,
                "hf": hf,
                "error_Ha": best_energy - casci,
                "error_mHa": 1e3 * (best_energy - casci),
                "best_tokens": best_tokens,
                "n_params": n_params,
                "vocab_size": vocab_size,
                "backbone": cfg.backbone,
            },
            f,
            indent=2,
        )
    return result
