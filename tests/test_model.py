"""Transformer and GRPO shapes (problem-independent)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gqe.grpo import grpo_loss, standardize_rewards
from gqe.model import count_params, init_transformer, transformer_logits


def test_gqe_and_gqkae_logits_shapes():
    key = jax.random.PRNGKey(0)
    tokens = jnp.zeros((3, 5), dtype=jnp.int32)
    for backbone in ("gqe", "gqkae"):
        params = init_transformer(
            key, vocab_size=27, d_model=32, n_layers=2, n_heads=4, max_len=20, backbone=backbone
        )
        logits = transformer_logits(tokens, params)
        assert logits.shape == (3, 5, 27)
        assert count_params(params) > 1000
        assert jnp.isfinite(logits).all()


def test_grpo_shapes():
    new = jnp.zeros((4, 5))
    old = jnp.zeros((4, 5))
    adv = standardize_rewards(jnp.array([-1.0, -1.1, -0.9, -1.05]))
    loss, metrics = grpo_loss(new, old, adv)
    assert jnp.isfinite(loss)
    assert "pg_objective" in metrics
