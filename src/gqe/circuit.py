"""Circuits on ``lightning.qubit``, compiled once with Catalyst ``@qjit``.

The trick: the token sequence is an *input* to one compiled program instead of
being baked into the circuit. At every step we apply every pool gate, with its
angle zeroed unless it is the chosen token. A rotation by 0 is the identity, so
this is exactly the chosen circuit, and it compiles once per sequence length
(not once per sampled sequence).
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from catalyst import for_loop, qjit

from gqe.molecule import H4System
from gqe.operators import ExcitationOp, hf_occupation

PAD = -1  # matches no pool index → the step does nothing


@lru_cache(maxsize=16)
def _compiled_probs(n_qubits: int, hf: tuple[int, ...], pool: tuple[ExcitationOp, ...], length: int):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    gates = [(j, op) for j, op in enumerate(pool) if op.kind != "id"]

    @qjit
    @qml.qnode(dev)
    def probs(tokens):
        qml.BasisState(jnp.array(hf), wires=range(n_qubits))

        @for_loop(0, length, 1)
        def step(t):
            tok = tokens[t]
            for j, op in gates:
                theta = jnp.where(tok == j, op.angle, 0.0)
                if op.kind == "single":
                    qml.SingleExcitation(theta, wires=list(op.wires))
                else:
                    qml.DoubleExcitation(theta, wires=list(op.wires))

        step()
        return qml.probs(wires=range(n_qubits))

    return probs


def statevector_probs(
    system: H4System,
    pool: list[ExcitationOp],
    token_ids: list[int] | np.ndarray,
) -> np.ndarray:
    """Exact |⟨x|ψ⟩|² over the 2^n computational basis states (wire 0 = leftmost bit)."""
    toks = [int(t) for t in token_ids] or [PAD]
    fn = _compiled_probs(
        system.n_qubits,
        tuple(int(b) for b in hf_occupation(system)),
        tuple(pool),
        len(toks),
    )
    p = np.clip(np.asarray(fn(jnp.asarray(toks, dtype=jnp.int32)), dtype=np.float64), 0.0, None)
    return p / p.sum()


def sample_circuit(
    system: H4System,
    pool: list[ExcitationOp],
    token_ids: list[int] | np.ndarray,
    shots: int = 2048,
    seed: int | None = None,
) -> dict[str, int]:
    """Prepare HF → apply pool[token] gates → return {bitstring: count}.

    Shots are drawn as a multinomial over the exact probabilities, which has the
    same distribution as measuring the state ``shots`` times.
    """
    p = statevector_probs(system, pool, token_ids)
    counts = np.random.default_rng(seed).multinomial(shots, p)
    n = system.n_qubits
    return {format(i, f"0{n}b"): int(c) for i, c in enumerate(counts) if c}
