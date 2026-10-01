"""Clipped token-level GRPO (paper Eqs. 13–14)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def standardize_rewards(rewards: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    mean = jnp.mean(rewards)
    std = jnp.std(rewards)
    return (rewards - mean) / (std + eps)


def token_log_probs(logits: jnp.ndarray, tokens: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """logits (B,T,V) predict tokens[:,1:]; returns (B, T-1)."""
    logp = jax.nn.log_softmax(logits[:, :-1, :] / temperature, axis=-1)
    targets = tokens[:, 1:]
    return jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]


def grpo_loss(
    new_log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_eps: float = 0.2,
) -> tuple[jnp.ndarray, dict]:
    """L = -mean_t min(ρ Â, clip(ρ) Â) with Â from group-standardized rewards."""
    old_log_probs = jax.lax.stop_gradient(old_log_probs)
    advantages = jax.lax.stop_gradient(advantages)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    adv = advantages[:, None]
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = jnp.minimum(unclipped, clipped)
    pg = per_token.mean()
    return -pg, {"pg_objective": pg, "mean_ratio": ratio.mean()}
