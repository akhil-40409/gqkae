"""Baselines for Ising/QUBO: brute force, random sampling, simulated annealing, QAOA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from catalyst import qjit
from scipy.optimize import minimize

from gqe.ising import Ising, IsingProblem


def random_search(problem: IsingProblem, n_samples: int, seed: int = 0) -> float:
    """Best of ``n_samples`` uniform bitstrings: measuring |+⟩^n with the same shot budget."""
    idx = np.random.default_rng(seed).integers(0, 2**problem.n_qubits, size=n_samples)
    return float(problem.energies[idx].min())


def simulated_annealing(
    model: Ising,
    *,
    sweeps: int = 1000,
    restarts: int = 32,
    beta_range: tuple[float, float] = (0.1, 10.0),
    seed: int = 0,
) -> float:
    """Single-spin-flip Metropolis on a geometric β schedule, ``restarts`` chains in parallel."""
    rng = np.random.default_rng(seed)
    n = model.n
    J = model.J + model.J.T
    scale = max(np.abs(model.h).max(initial=0.0), np.abs(J).max(initial=0.0), 1e-12)
    betas = np.geomspace(*beta_range, sweeps) / scale
    s = rng.choice([-1.0, 1.0], size=(restarts, n))
    for beta in betas:
        for i in range(n):
            dE = -2.0 * s[:, i] * (model.h[i] + s @ J[i])
            flip = (dE <= 0) | (rng.random(restarts) < np.exp(-beta * np.clip(dE, 0.0, None)))
            s[flip, i] *= -1
    return float(model.energy((1 - s) / 2).min())


@dataclass
class QAOAResult:
    expected_energy: float
    best_energy: float  # best of ``shots`` samples from the optimized state
    p: int
    n_evals: int


def qaoa(
    problem: IsingProblem,
    *,
    p: int = 2,
    maxiter: int = 300,
    shots: int = 1024,
    seed: int = 0,
) -> QAOAResult:
    """Depth-p QAOA, exact ⟨H⟩ from compiled probabilities, COBYLA from a linear ramp."""
    model, n = problem.model, problem.n_qubits
    energies = problem.energies
    dev = qml.device("lightning.qubit", wires=n)

    @qjit
    @qml.qnode(dev)
    def probs(x):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            gamma, beta = x[layer], x[p + layer]
            for i in np.flatnonzero(model.h):
                qml.RZ(2 * gamma * model.h[i], wires=int(i))
            for i, j in model.edges:
                qml.IsingZZ(2 * gamma * model.J[i, j], wires=[i, j])
            for w in range(n):
                qml.RX(2 * beta, wires=w)
        return qml.probs(wires=range(n))

    n_evals = 0

    def objective(x):
        nonlocal n_evals
        n_evals += 1
        return float(np.asarray(probs(np.asarray(x))) @ energies)

    ramp = (np.arange(p) + 0.5) / p
    x0 = np.concatenate([0.5 * ramp, 0.5 * (1 - ramp)])
    opt = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    pr = np.clip(np.asarray(probs(np.asarray(opt.x)), dtype=np.float64), 0.0, None)
    counts = np.random.default_rng(seed).multinomial(shots, pr / pr.sum())
    best, _ = problem.best_sample(counts)
    return QAOAResult(expected_energy=float(opt.fun), best_energy=best, p=p, n_evals=n_evals)
