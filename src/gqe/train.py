"""GRPO training loop: the generator writes circuits, the problem scores them."""

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

from gqe.circuit import sample_counts
from gqe.grpo import grpo_loss, standardize_rewards, token_log_probs
from gqe.model import count_params, init_transformer, sample_sequences, transformer_logits
from gqe.problem import Problem

BOS_ID = 0  # pool[0] is the identity; it doubles as the BOS token


@dataclass
class TrainConfig:
    n_iters: int = 40
    group_size: int = 8
    seq_len: int = 20
    shots: int = 256
    d_model: int = 32
    d_latent: int = 12
    d_ff: int = 128
    n_layers: int = 2
    n_heads: int = 4
    backbone: Literal["gqkae", "gqe"] = "gqkae"
    lr: float = 2e-3
    weight_decay: float = 0.01
    policy_updates: int = 10
    clip_eps: float = 0.2
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    seed: int = 0
    out_dir: str | None = None


@dataclass
class TrainResult:
    best_energy: float  # lowest energy of any bitstring measured during training
    best_index: int  # its basis index (wire 0 = MSB)
    best_score: float  # best circuit score (e.g. CVaR) seen
    best_tokens: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    n_params: int = 0
    vocab_size: int = 0
    backbone: str = "gqkae"
    shots_used: int = 0


def train(problem: Problem, cfg: TrainConfig | None = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    vocab_size = len(problem.pool)

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

    def loss_fn(p, tokens, old_lp, advantages):
        new_lp = token_log_probs(transformer_logits(tokens, p), tokens, temperature=cfg.temperature)
        return grpo_loss(new_lp, old_lp, advantages, clip_eps=cfg.clip_eps)

    @jax.jit
    def old_log_probs(p, tokens):
        return token_log_probs(transformer_logits(tokens, p), tokens, temperature=cfg.temperature)

    @jax.jit
    def update_step(p, opt_st, tokens, old_lp, advantages):
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, tokens, old_lp, advantages)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        return optax.apply_updates(p, updates), opt_st, loss

    best_energy, best_index = float("inf"), -1
    best_score, best_tokens = float("inf"), []
    history: list[dict] = []

    for it in trange(cfg.n_iters, desc=f"{cfg.backbone} n={problem.n_qubits}"):
        key, k_sample = jax.random.split(key)
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

        scores = []
        for m in range(cfg.group_size):
            seq = tokens_np[m, 1:].tolist()
            counts = sample_counts(problem, seq, cfg.shots, seed=cfg.seed + it * 1000 + m)
            s = problem.score(counts)
            e, i = problem.best_sample(counts)
            scores.append(s)
            if s < best_score:
                best_score, best_tokens = s, seq
            if e < best_energy:
                best_energy, best_index = e, i

        advantages = standardize_rewards(-jnp.asarray(scores, dtype=jnp.float32))
        old_lp = jax.lax.stop_gradient(old_log_probs(params, tokens))
        loss = 0.0
        for _ in range(cfg.policy_updates):
            params, opt_state, loss = update_step(params, opt_state, tokens, old_lp, advantages)

        history.append(
            {
                "iter": it,
                "mean_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "best_score": best_score,
                "best_energy": best_energy,
                "loss": float(loss),
            }
        )

    result = TrainResult(
        best_energy=best_energy,
        best_index=best_index,
        best_score=best_score,
        best_tokens=best_tokens,
        history=history,
        n_params=n_params,
        vocab_size=vocab_size,
        backbone=cfg.backbone,
        shots_used=cfg.n_iters * cfg.group_size * cfg.shots,
    )
    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (out / "result.json").write_text(json.dumps(asdict(result), indent=2))
    return result
