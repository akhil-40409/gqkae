"""The gate vocabulary: every token the generator can emit is one fixed-angle gate.

Local gates act on one or two wires. Layer gates act on every wire at once:
``zphase`` is exp(−iγ Σ_k c_k Z_{S_k}) for diagonal terms (|S_k| ≤ 2), and
``xmix`` is exp(−iβ Σ_i X_i).
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml

KINDS = ("rx", "ry", "rz", "zz")
LAYER_KINDS = ("zphase", "xmix")
_OPS = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ, "zz": qml.IsingZZ}


@dataclass(frozen=True)
class Gate:
    kind: str  # "id" | KINDS | LAYER_KINDS
    wires: tuple[int, ...]
    angle: float
    label: str
    terms: tuple[tuple[tuple[int, ...], float], ...] = ()  # zphase only


IDENTITY = Gate("id", (), 0.0, "I")


def apply_layer(gate: Gate, angle=None) -> None:
    """Apply a layer gate; ``angle`` may be traced (defaults to ``gate.angle``)."""
    a = gate.angle if angle is None else angle
    if gate.kind == "zphase":
        for wires, c in gate.terms:
            if len(wires) == 1:
                qml.RZ(2 * c * a, wires=wires[0])
            else:
                qml.IsingZZ(2 * c * a, wires=list(wires))
    elif gate.kind == "xmix":
        for w in gate.wires:
            qml.RX(2 * a, wires=w)
    else:
        raise ValueError(f"{gate.kind!r} is not a layer gate")


def apply_gate(gate: Gate) -> None:
    """Apply one pool gate inside a PennyLane QNode."""
    if gate.kind == "id":
        return
    if gate.kind in LAYER_KINDS:
        apply_layer(gate)
        return
    _OPS[gate.kind](gate.angle, wires=list(gate.wires))
