"""UCCSD-derived discrete operator pool for GQE / GQKAE.

Each token indexes a fixed-angle excitation unitary (or identity). Sequences of
tokens define candidate circuits acting on the Hartree–Fock reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

from gqe.molecule import H4System


@dataclass(frozen=True)
class ExcitationOp:
    """One pool operator: identity, single, or double excitation."""

    kind: str  # "id" | "single" | "double"
    wires: tuple[int, ...]
    angle: float
    label: str


def build_uccsd_pool(
    system: H4System,
    angle: float = 0.5 * np.pi,
    include_identity: bool = True,
) -> list[ExcitationOp]:
    """Build a discrete UCCSD-style pool for ``system``.

    Uses PennyLane ``qml.qchem.excitations`` over ``n_qubits`` spin-orbitals.
    Angles are fixed (default π/2) so the generative model only chooses
    *which* operators to compose — matching the GQE discrete-circuit view.
    """
    electrons = system.n_electrons
    qubits = system.n_qubits
    singles, doubles = qml.qchem.excitations(electrons, qubits)

    pool: list[ExcitationOp] = []
    if include_identity:
        pool.append(ExcitationOp("id", (), 0.0, "I"))

    for i, (p, q) in enumerate(singles):
        pool.append(ExcitationOp("single", (int(p), int(q)), float(angle), f"S{i}"))
    for i, (p, q, r, s) in enumerate(doubles):
        pool.append(
            ExcitationOp("double", (int(p), int(q), int(r), int(s)), float(angle), f"D{i}")
        )
    return pool


def hf_occupation(system: H4System) -> np.ndarray:
    """Jordan–Wigner HF bitstring used by PennyLane qchem (α/β interleaved)."""
    return np.asarray(qml.qchem.hf_state(system.n_electrons, system.n_qubits), dtype=int)


def apply_operator(op: ExcitationOp) -> None:
    """Apply one pool operator in-place inside a PennyLane QNode."""
    if op.kind == "id":
        return
    if op.kind == "single":
        qml.SingleExcitation(op.angle, wires=list(op.wires))
        return
    if op.kind == "double":
        qml.DoubleExcitation(op.angle, wires=list(op.wires))
        return
    raise ValueError(f"Unknown operator kind: {op.kind}")
