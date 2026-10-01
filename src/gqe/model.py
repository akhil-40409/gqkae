"""HQKANsformer: decoder-only transformer with DARUAN/QKAN feed-forward blocks.

Consolidates DARUAN (arXiv:2509.14026 / Jim137/qkan ``pz_encoding``), HQKAN
(linear → latent QKAN → linear), and causal attention into one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Union

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# DARUAN / QKAN
# ---------------------------------------------------------------------------


def _apply_h(state: jnp.ndarray) -> jnp.ndarray:
    a0, a1 = state[..., 0], state[..., 1]
    s = jnp.sqrt(0.5)
    return jnp.stack([s * (a0 + a1), s * (a0 - a1)], axis=-1)


def _apply_ry(state: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    half = theta / 2.0
    c, s = jnp.cos(half), jnp.sin(half)
    a0, a1 = state[..., 0], state[..., 1]
    while c.ndim < a0.ndim:
        c, s = c[None, ...], s[None, ...]
    return jnp.stack([c * a0 - s * a1, s * a0 + c * a1], axis=-1)


def _apply_rz(state: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    half = theta / 2.0
    e0, e1 = jnp.exp(-1j * half), jnp.exp(1j * half)
    while e0.ndim < state[..., 0].ndim:
        e0, e1 = e0[None, ...], e1[None, ...]
    return jnp.stack([e0 * state[..., 0], e1 * state[..., 1]], axis=-1)


def daruan_forward(
    x: jnp.ndarray,
    theta: jnp.ndarray,
    preacts_weight: jnp.ndarray,
    preacts_bias: jnp.ndarray,
) -> jnp.ndarray:
    """Single-qubit data-reuploading activation → ⟨Z⟩ (paper Eqs. 7–8)."""
    reps = preacts_weight.shape[-1]
    zeros = jnp.zeros_like(x, dtype=jnp.complex64)
    ones = jnp.ones_like(x, dtype=jnp.complex64)
    state = jnp.stack([ones, zeros], axis=-1)
    state = _apply_h(state)
    for ell in range(reps):
        state = _apply_rz(state, theta[..., ell, 0])
        state = _apply_ry(state, theta[..., ell, 1])
        enc = preacts_weight[..., ell] * x + preacts_bias[..., ell]
        state = _apply_rz(state, enc)
    state = _apply_rz(state, theta[..., reps, 0])
    state = _apply_ry(state, theta[..., reps, 1])
    p0 = jnp.real(state[..., 0] * jnp.conj(state[..., 0]))
    p1 = jnp.real(state[..., 1] * jnp.conj(state[..., 1]))
    return p0 - p1


@dataclass
class DARUANParams:
    theta: jnp.ndarray
    preacts_weight: jnp.ndarray
    preacts_bias: jnp.ndarray
    postact_weights: jnp.ndarray
    postact_bias: jnp.ndarray


def init_daruan(key: jax.Array, out_dim: int, in_dim: int, reps: int = 2) -> DARUANParams:
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.1
    return DARUANParams(
        theta=scale * jax.random.normal(k1, (out_dim, in_dim, reps + 1, 2)),
        preacts_weight=jnp.ones((out_dim, in_dim, reps))
        + 0.01 * jax.random.normal(k2, (out_dim, in_dim, reps)),
        preacts_bias=0.01 * jax.random.normal(k3, (out_dim, in_dim, reps)),
        postact_weights=jnp.ones((out_dim, in_dim)),
        postact_bias=jnp.zeros((out_dim, in_dim)),
    )


def qkan_layer(x: jnp.ndarray, params: DARUANParams) -> jnp.ndarray:
    """y_j = Σ_i ϕ_{j,i}(x_i)."""
    out_dim, in_dim = params.theta.shape[:2]
    xb = jnp.broadcast_to(x[:, None, :], (x.shape[0], out_dim, in_dim))
    phi = daruan_forward(xb, params.theta, params.preacts_weight, params.preacts_bias)
    return jnp.sum(phi * params.postact_weights + params.postact_bias, axis=-1)


# ---------------------------------------------------------------------------
# HQKAN FFN
# ---------------------------------------------------------------------------


@dataclass
class LinearParams:
    weight: jnp.ndarray
    bias: jnp.ndarray


@dataclass
class HQKANParams:
    enc: LinearParams
    qkan: DARUANParams
    dec: LinearParams


def init_hqkan(key: jax.Array, d_model: int, d_latent: int, reps: int = 2) -> HQKANParams:
    k1, k2, k3 = jax.random.split(key, 3)
    s = 0.02

    def lin(k, out_d, in_d):
        return LinearParams(s * jax.random.normal(k, (out_d, in_d)), jnp.zeros((out_d,)))

    return HQKANParams(enc=lin(k1, d_latent, d_model), qkan=init_daruan(k2, d_latent, d_latent, reps), dec=lin(k3, d_model, d_latent))


def hqkan_forward(h: jnp.ndarray, params: HQKANParams) -> jnp.ndarray:
    squeeze = h.ndim == 2
    if squeeze:
        h = h[:, None, :]
    b, t, d = h.shape
    flat = h.reshape(b * t, d)
    z = flat @ params.enc.weight.T + params.enc.bias
    z = qkan_layer(z, params.qkan)
    g = (z @ params.dec.weight.T + params.dec.bias).reshape(b, t, d)
    return g[:, 0, :] if squeeze else g


# ---------------------------------------------------------------------------
# Standard GPT-style MLP FFN (GQE baseline)
# ---------------------------------------------------------------------------


@dataclass
class MLPParams:
    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray


def init_mlp(key: jax.Array, d_model: int, d_ff: int) -> MLPParams:
    k1, k2 = jax.random.split(key)
    s = 0.02
    return MLPParams(
        w1=s * jax.random.normal(k1, (d_ff, d_model)),
        b1=jnp.zeros((d_ff,)),
        w2=s * jax.random.normal(k2, (d_model, d_ff)),
        b2=jnp.zeros((d_model,)),
    )


def mlp_forward(h: jnp.ndarray, params: MLPParams) -> jnp.ndarray:
    """Position-wise GELU MLP: W2 GELU(W1 h + b1) + b2 (paper §IV-A GPT-2 FFN)."""
    return (jax.nn.gelu(h @ params.w1.T + params.b1) @ params.w2.T) + params.b2


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


@dataclass
class AttentionParams:
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray
    wo: jnp.ndarray


@dataclass
class BlockParams:
    attn: AttentionParams
    ffn: Union[HQKANParams, MLPParams]
    kind: str = "hqkan"


@dataclass
class TransformerParams:
    tok_emb: jnp.ndarray
    pos_emb: jnp.ndarray
    blocks: list[BlockParams]
    out_w: jnp.ndarray
    out_b: jnp.ndarray
    d_model: int
    n_heads: int
    max_len: int
    vocab_size: int


def _layernorm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def causal_attention(x: jnp.ndarray, params: AttentionParams, n_heads: int) -> jnp.ndarray:
    b, t, d = x.shape
    dh = d // n_heads
    q = (x @ params.wq).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    k = (x @ params.wk).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    v = (x @ params.wv).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(dh)
    causal = jnp.tril(jnp.ones((t, t), dtype=x.dtype))
    scores = jnp.where(causal[None, None] > 0, scores, -1e9)
    ctx = jnp.matmul(jax.nn.softmax(scores, axis=-1), v)
    return ctx.transpose(0, 2, 1, 3).reshape(b, t, d) @ params.wo


def init_transformer(
    key: jax.Array,
    vocab_size: int,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    max_len: int = 20,
    d_latent: int = 12,
    reps: int = 2,
    backbone: Literal["gqkae", "gqe"] = "gqkae",
    d_ff: int | None = None,
) -> TransformerParams:
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
    d_ff = d_model * 4 if d_ff is None else d_ff
    keys = jax.random.split(key, 3 + n_layers)
    s = 0.02
    tok = s * jax.random.normal(keys[0], (vocab_size, d_model))
    pos = s * jax.random.normal(keys[1], (max_len + 1, d_model))
    blocks = []
    for i in range(n_layers):
        bk = jax.random.split(keys[2 + i], 2)
        ak = jax.random.split(bk[0], 4)
        attn = AttentionParams(
            wq=s * jax.random.normal(ak[0], (d_model, d_model)),
            wk=s * jax.random.normal(ak[1], (d_model, d_model)),
            wv=s * jax.random.normal(ak[2], (d_model, d_model)),
            wo=s * jax.random.normal(ak[3], (d_model, d_model)),
        )
        if backbone == "gqe":
            ffn: Union[HQKANParams, MLPParams] = init_mlp(bk[1], d_model, d_ff)
            kind = "mlp"
        else:
            ffn = init_hqkan(bk[1], d_model, d_latent, reps)
            kind = "hqkan"
        blocks.append(BlockParams(attn=attn, ffn=ffn, kind=kind))
    return TransformerParams(
        tok_emb=tok,
        pos_emb=pos,
        blocks=blocks,
        out_w=s * jax.random.normal(keys[-1], (vocab_size, d_model)),
        out_b=jnp.zeros((vocab_size,)),
        d_model=d_model,
        n_heads=n_heads,
        max_len=max_len,
        vocab_size=vocab_size,
    )


def transformer_logits(tokens: jnp.ndarray, params: TransformerParams) -> jnp.ndarray:
    """tokens (B, T) → logits (B, T, V)."""
    t = tokens.shape[1]
    h = params.tok_emb[tokens] + params.pos_emb[None, :t, :]
    for block in params.blocks:
        h = _layernorm(h + causal_attention(h, block.attn, params.n_heads))
        if block.kind == "mlp":
            residual = mlp_forward(h, block.ffn)
        else:
            residual = hqkan_forward(h, block.ffn)
        h = _layernorm(h + residual)
    return h @ params.out_w.T + params.out_b


def count_params(params: Any) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params) if hasattr(x, "size")))


def sample_sequences(
    key: jax.Array,
    params: TransformerParams,
    *,
    bos_id: int,
    length: int,
    batch: int,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
) -> jnp.ndarray:
    """Autoregressive sample of shape (batch, 1+length) including BOS."""
    return _sample_jit(key, params, bos_id, length, batch, temperature, repetition_penalty)


@partial(jax.jit, static_argnums=(2, 3, 4))
def _sample_jit(key, params, bos_id, length, batch, temperature, repetition_penalty):
    # Fixed-shape buffer: unfilled slots sit to the right of step t, and the
    # causal mask keeps them from influencing the logits at position t.
    tokens = jnp.full((batch, length + 1), bos_id, dtype=jnp.int32)
    positions = jnp.arange(length + 1)

    def step(t, carry):
        key, tokens = carry
        key, k_sample = jax.random.split(key)
        logits = transformer_logits(tokens, params)[:, t, :] / temperature
        # Each earlier occurrence of a token divides its logit by the penalty once.
        seen = jax.nn.one_hot(tokens, params.vocab_size) * (positions <= t)[None, :, None]
        logits = logits / repetition_penalty ** seen.sum(axis=1)
        nxt = jax.random.categorical(k_sample, logits).astype(jnp.int32)
        return key, tokens.at[:, t + 1].set(nxt)

    _, tokens = jax.lax.fori_loop(0, length, step, (key, tokens))
    return tokens


# Register pytrees once
def _register() -> None:
    if getattr(_register, "_done", False):
        return

    def reg(cls, fields):
        jax.tree_util.register_pytree_node(
            cls,
            lambda o: (tuple(getattr(o, f) for f in fields), None),
            lambda _, xs: cls(**dict(zip(fields, xs))),
        )

    reg(DARUANParams, ["theta", "preacts_weight", "preacts_bias", "postact_weights", "postact_bias"])
    reg(LinearParams, ["weight", "bias"])
    reg(HQKANParams, ["enc", "qkan", "dec"])
    reg(MLPParams, ["w1", "b1", "w2", "b2"])
    reg(AttentionParams, ["wq", "wk", "wv", "wo"])
    jax.tree_util.register_pytree_node(
        BlockParams,
        lambda o: ((o.attn, o.ffn), o.kind),
        lambda kind, xs: BlockParams(attn=xs[0], ffn=xs[1], kind=kind),
    )

    def flat(o: TransformerParams):
        return (
            (o.tok_emb, o.pos_emb, tuple(o.blocks), o.out_w, o.out_b),
            (o.d_model, o.n_heads, o.max_len, o.vocab_size),
        )

    def unflat(aux, xs):
        tok, pos, blocks, ow, ob = xs
        d_model, n_heads, max_len, vocab = aux
        return TransformerParams(tok, pos, list(blocks), ow, ob, d_model, n_heads, max_len, vocab)

    jax.tree_util.register_pytree_node(TransformerParams, flat, unflat)
    _register._done = True  # type: ignore[attr-defined]


_register()
