"""Circuits on ``lightning.qubit``, compiled once with Catalyst ``@qjit``.

The token sequence is an *input* to one compiled program instead of being baked
into the circuit. Each step looks up the token's gate kind, wires and angle in
small tables, then applies one RX, RY, RZ and ZZ on those (dynamic) wires with
every angle zeroed except the chosen kind's. A rotation by 0 is the identity,
so this is exactly the chosen circuit, it compiles once per sequence length,
and the cost per step does not grow with the number of local gates. Layer
gates (few, but each touching every wire) run under a ``cond`` on the token.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from catalyst import cond, for_loop, qjit

from gqe.gates import KINDS, LAYER_KINDS, Gate, apply_layer
from gqe.problem import Problem

PAD = -1  # indexes the trailing identity row of the lookup tables


def _tables(pool: tuple[Gate, ...], n_qubits: int):
    kind, w0, w1, angle = [], [], [], []
    for g in (*pool, Gate("id", (), 0.0, "PAD")):
        kind.append(KINDS.index(g.kind) if g.kind in KINDS else -1)
        a = g.wires[0] if g.wires else 0
        # single-qubit gates still feed the ZZ slot, which needs two distinct wires
        b = g.wires[1] if len(g.wires) > 1 else (a + 1) % max(n_qubits, 2)
        w0.append(a)
        w1.append(b)
        angle.append(g.angle)
    return (
        np.asarray(kind, dtype=np.int32),
        np.asarray(w0, dtype=np.int32),
        np.asarray(w1, dtype=np.int32),
        np.asarray(angle, dtype=np.float64),
    )


@lru_cache(maxsize=16)
def _compiled_probs(n_qubits: int, init: str, pool: tuple[Gate, ...], length: int):
    kind_t, w0_t, w1_t, angle_t = (jnp.asarray(x) for x in _tables(pool, n_qubits))
    present = [c for c, k in enumerate(KINDS) if any(g.kind == k for g in pool)]
    layers = [(j, g) for j, g in enumerate(pool) if g.kind in LAYER_KINDS]
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qjit
    @qml.qnode(dev)
    def probs(tokens):
        if init == "plus":
            for w in range(n_qubits):
                qml.Hadamard(wires=w)

        @for_loop(0, length, 1)
        def step(t):
            tok = tokens[t]
            k, i, j, a = kind_t[tok], w0_t[tok], w1_t[tok], angle_t[tok]
            for code in present:
                theta = jnp.where(k == code, a, 0.0)
                if KINDS[code] == "rx":
                    qml.RX(theta, wires=i)
                elif KINDS[code] == "ry":
                    qml.RY(theta, wires=i)
                elif KINDS[code] == "rz":
                    qml.RZ(theta, wires=i)
                else:
                    qml.IsingZZ(theta, wires=[i, j])
            for idx, gate in layers:
                cond(tok == idx)(lambda gate=gate: apply_layer(gate))()

        step()
        return qml.probs(wires=range(n_qubits))

    return probs


def statevector_probs(problem: Problem, token_ids: list[int] | np.ndarray) -> np.ndarray:
    """Exact |⟨x|ψ⟩|² over the 2^n computational basis states (wire 0 = leftmost bit)."""
    toks = [int(t) for t in token_ids] or [PAD]
    fn = _compiled_probs(problem.n_qubits, problem.init, tuple(problem.pool), len(toks))
    p = np.clip(np.asarray(fn(jnp.asarray(toks, dtype=jnp.int32)), dtype=np.float64), 0.0, None)
    return p / p.sum()


def sample_counts(
    problem: Problem,
    token_ids: list[int] | np.ndarray,
    shots: int,
    seed: int | None = None,
) -> np.ndarray:
    """Shots drawn as a multinomial over the exact probabilities (same law as measuring)."""
    p = statevector_probs(problem, token_ids)
    return np.random.default_rng(seed).multinomial(shots, p)
