"""QUBO / Ising models and the gate pool the generator uses to search them.

Conventions (fixed everywhere in this package):

- A measured bit ``b_i`` *is* the QUBO variable ``x_i``.
- Its spin is the Z eigenvalue, ``s_i = 1 - 2 b_i`` (|0⟩ → +1, |1⟩ → −1).
- ``E(s) = offset + hᵀs + sᵀ J s``. ``J`` may be upper triangular or symmetric;
  it is stored strictly upper triangular (diagonal folded into ``offset``), so
  ``H = offset + Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j`` is diagonal and ``E`` is its
  eigenvalue on the basis state ``|b⟩``.
- ``QUBO(x) = xᵀ Q x + const`` for any square ``Q``, same rule as ``J``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pennylane as qml

from gqe.gates import IDENTITY, Gate

DEFAULT_GAMMAS = (0.1, 0.2, 0.4, 0.8)
DEFAULT_BETAS = (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4)
DEFAULT_LOCAL = (-np.pi / 4, np.pi / 4)


@dataclass(frozen=True, eq=False)
class Ising:
    h: np.ndarray  # (n,)
    J: np.ndarray  # (n, n), strictly upper triangular
    offset: float = 0.0

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=np.float64)
        J = np.asarray(self.J, dtype=np.float64)
        if J.shape != (h.size, h.size):
            raise ValueError(f"J must be {h.size}x{h.size}, got {J.shape}")
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "J", np.triu(J + J.T, k=1))
        object.__setattr__(self, "offset", float(self.offset + np.trace(J)))

    @property
    def n(self) -> int:
        return int(self.h.size)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(int(i), int(j)) for i, j in zip(*np.nonzero(self.J))]

    @classmethod
    def from_qubo(cls, Q: np.ndarray, const: float = 0.0) -> Ising:
        Q = np.asarray(Q, dtype=np.float64)
        d = np.diag(Q)
        W = np.triu(Q + Q.T, k=1)  # coefficient of x_i x_j, i < j
        W_sym = W + W.T
        h = -d / 2 - W_sym.sum(axis=1) / 4
        return cls(h=h, J=W / 4, offset=float(const + d.sum() / 2 + W.sum() / 4))

    def to_qubo(self) -> tuple[np.ndarray, float]:
        """Upper-triangular Q and const with ``xᵀQx + const == E(1 - 2x)``."""
        J_sym = self.J + self.J.T
        Q = 4 * self.J + np.diag(-2 * self.h - 2 * J_sym.sum(axis=1))
        return Q, float(self.offset + self.h.sum() + self.J.sum())

    def energy(self, bits: np.ndarray) -> np.ndarray:
        """Energy of bitstrings ``bits`` with shape (..., n)."""
        s = 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)
        return self.offset + s @ self.h + ((s @ self.J) * s).sum(axis=-1)

    def all_energies(self) -> np.ndarray:
        """E for every basis state, indexed like ``qml.probs`` (wire 0 = MSB)."""
        return self.energy(index_to_bits(np.arange(2**self.n), self.n))

    def ground_state(self) -> tuple[float, np.ndarray]:
        """Brute force. Fine up to ~22 variables."""
        E = self.all_energies()
        i = int(np.argmin(E))
        return float(E[i]), index_to_bits(np.array(i), self.n)

    def hamiltonian(self) -> qml.Hamiltonian:
        coeffs = [self.offset]
        ops = [qml.Identity(0)]
        for i, hi in enumerate(self.h):
            if hi:
                coeffs.append(hi)
                ops.append(qml.Z(i))
        for i, j in self.edges:
            coeffs.append(self.J[i, j])
            ops.append(qml.Z(i) @ qml.Z(j))
        return qml.Hamiltonian(coeffs, ops)


def index_to_bits(idx: np.ndarray, n: int) -> np.ndarray:
    return (np.asarray(idx)[..., None] >> np.arange(n - 1, -1, -1)) & 1


def bits_to_str(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in bits)


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def random_sk(n: int, seed: int = 0, h_scale: float = 0.0) -> Ising:
    """Sherrington–Kirkpatrick spin glass: J_ij ~ N(0, 1/n), optional fields h_i ~ N(0, h_scale²)."""
    rng = np.random.default_rng(seed)
    J = np.triu(rng.normal(size=(n, n)), k=1) / np.sqrt(n)
    return Ising(h=h_scale * rng.normal(size=n), J=J)


def maxcut(n: int, edges: list[tuple[int, int]], weights: list[float] | None = None) -> Ising:
    """Minimizing E ⇔ maximizing the cut; E = −(cut weight)."""
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=float)
    J = np.zeros((n, n))
    for (i, j), wij in zip(edges, w):
        J[min(i, j), max(i, j)] += wij / 2
    return Ising(h=np.zeros(n), J=J, offset=-float(w.sum()) / 2)


def load_qubo(path: str | Path) -> Ising:
    """``.npy`` holding Q, or ``.json`` with ``{"Q", "const"}`` or ``{"h", "J", "offset"}``."""
    path = Path(path)
    if path.suffix == ".npy":
        return Ising.from_qubo(np.load(path))
    data = json.loads(path.read_text())
    if "Q" in data:
        return Ising.from_qubo(np.asarray(data["Q"]), float(data.get("const", 0.0)))
    return Ising(h=np.asarray(data["h"]), J=np.asarray(data["J"]), offset=float(data.get("offset", 0.0)))


# ---------------------------------------------------------------------------
# Problem adapter
# ---------------------------------------------------------------------------


def build_pool(
    model: Ising,
    gammas: tuple[float, ...] = DEFAULT_GAMMAS,
    betas: tuple[float, ...] = DEFAULT_BETAS,
    local_angles: tuple[float, ...] = DEFAULT_LOCAL,
) -> list[Gate]:
    """Cost layers exp(−iγ H/‖H‖), mixers exp(−iβ ΣX), and single-qubit RY biases.

    H is scaled so its largest coefficient is 1, so the same γ grid works for
    any instance. Free sequences of these tokens include every discrete-angle
    QAOA circuit of depth ≤ seq_len/2.
    """
    scale = max(np.abs(model.h).max(initial=0.0), np.abs(model.J).max(initial=0.0)) or 1.0
    terms = tuple(((int(i),), float(model.h[i] / scale)) for i in np.flatnonzero(model.h)) + tuple(
        ((i, j), float(model.J[i, j] / scale)) for i, j in model.edges
    )
    wires = tuple(range(model.n))
    pool = [IDENTITY]
    pool += [Gate("zphase", wires, float(g), f"C({g:.2f})", terms) for g in gammas]
    pool += [Gate("xmix", wires, float(b), f"M({b:+.2f})") for b in betas]
    pool += [Gate("ry", (i,), float(a), f"Y{i}({a:+.2f})") for a in local_angles for i in wires]
    return pool


def cvar(energies: np.ndarray, counts: np.ndarray, alpha: float) -> float:
    """Mean energy of the best ``alpha`` fraction of shots (α=1: mean, α→0: min)."""
    idx = np.flatnonzero(counts)
    order = np.argsort(energies[idx])
    e = energies[idx][order]
    c = counts[idx][order].astype(np.float64)
    keep = max(alpha * c.sum(), 1.0)
    take = np.clip(keep - (np.cumsum(c) - c), 0.0, c)
    return float((take * e).sum() / take.sum())


@dataclass
class IsingProblem:
    """Ising/QUBO instance, start in |+⟩^n, reward = −CVaR_α of the sampled energies."""

    model: Ising
    gammas: tuple[float, ...] = DEFAULT_GAMMAS
    betas: tuple[float, ...] = DEFAULT_BETAS
    local_angles: tuple[float, ...] = DEFAULT_LOCAL
    cvar_alpha: float = 0.1
    init: Literal["zero", "plus"] = "plus"
    n_qubits: int = field(init=False)
    pool: list[Gate] = field(init=False)
    energies: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_qubits = self.model.n
        self.pool = build_pool(self.model, self.gammas, self.betas, self.local_angles)
        self.energies = self.model.all_energies()

    def score(self, counts: np.ndarray) -> float:
        return cvar(self.energies, counts, self.cvar_alpha)

    def best_sample(self, counts: np.ndarray) -> tuple[float, int]:
        idx = np.flatnonzero(counts)
        i = int(idx[np.argmin(self.energies[idx])])
        return float(self.energies[i]), i
