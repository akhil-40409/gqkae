"""Ising/QUBO conventions, the compiled circuit, baselines, and a tiny end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pennylane as qml
import pytest

from gqkae.baselines import qaoa, random_search, simulated_annealing
from gqkae.circuit import sample_counts, statevector_probs
from gqkae.gates import apply_gate
from gqkae.ising import Ising, IsingProblem, cvar, index_to_bits, load_qubo, maxcut, random_sk
from gqkae.train import TrainConfig, train


def test_qubo_ising_roundtrip():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 5))
    model = Ising.from_qubo(Q, const=1.5)
    x = index_to_bits(np.arange(32), 5)
    qubo = np.einsum("ki,ij,kj->k", x, Q, x) + 1.5
    assert np.allclose(model.energy(x), qubo)
    Q2, c2 = model.to_qubo()
    assert np.allclose(np.einsum("ki,ij,kj->k", x, Q2, x) + c2, qubo)


def test_symmetric_and_upper_J_agree():
    rng = np.random.default_rng(1)
    U = np.triu(rng.normal(size=(4, 4)), k=1)
    h = rng.normal(size=4)
    a = Ising(h=h, J=U)
    b = Ising(h=h, J=(U + U.T) / 2)
    assert np.allclose(a.all_energies(), b.all_energies())


def test_energy_order_matches_pennylane():
    """E[i] is ⟨i|H|i⟩ with the same basis indexing as qml.probs."""
    model = random_sk(5, seed=2, h_scale=0.5)
    problem = IsingProblem(model)
    toks = np.random.default_rng(0).integers(0, len(problem.pool), size=12)
    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def expval():
        for w in range(5):
            qml.Hadamard(wires=w)
        for t in toks:
            apply_gate(problem.pool[int(t)])
        return qml.expval(model.hamiltonian())

    p = statevector_probs(problem, toks)
    assert abs(p @ problem.energies - float(expval())) < 1e-10


def test_compiled_circuit_matches_default_qubit():
    problem = IsingProblem(random_sk(6, seed=3, h_scale=0.3), init="zero")
    toks = np.random.default_rng(4).integers(0, len(problem.pool), size=20)
    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def reference():
        for t in toks:
            apply_gate(problem.pool[int(t)])
        return qml.probs(wires=range(6))

    assert np.allclose(statevector_probs(problem, toks), reference(), atol=1e-10)


def test_empty_circuit_is_uniform():
    problem = IsingProblem(random_sk(4))
    assert np.allclose(statevector_probs(problem, []), 1 / 16)


def test_cvar_limits():
    energies = np.array([3.0, 1.0, 2.0, 0.0])
    counts = np.array([5, 3, 2, 0])
    assert cvar(energies, counts, 1.0) == pytest.approx((15 + 3 + 4) / 10)
    assert cvar(energies, counts, 1e-6) == pytest.approx(1.0)
    assert cvar(energies, counts, 0.5) == pytest.approx((3 + 4) / 5)


def test_maxcut_optimum():
    square = maxcut(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    e0, bits = square.ground_state()
    assert e0 == pytest.approx(-4.0)
    assert list(bits) in ([0, 1, 0, 1], [1, 0, 1, 0])


def test_load_qubo(tmp_path):
    Q = [[-1.0, 2.0], [0.0, -1.0]]
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"Q": Q, "const": 0.5}))
    model = load_qubo(path)
    assert model.ground_state()[0] == pytest.approx(-0.5)  # x = 10 or 01
    np.save(tmp_path / "q.npy", np.array(Q))
    assert np.allclose(load_qubo(tmp_path / "q.npy").all_energies() + 0.5, model.all_energies())


def test_best_sample_is_measured_minimum():
    problem = IsingProblem(random_sk(5, seed=5))
    counts = sample_counts(problem, [], shots=64, seed=0)
    e, i = problem.best_sample(counts)
    assert counts[i] > 0
    assert e == problem.energies[counts > 0].min()


def test_baselines_bracket_ground_state():
    model = random_sk(8, seed=6)
    problem = IsingProblem(model)
    e0, _ = model.ground_state()
    # final states of 4 chains only, so a lucky early visit can't pass this
    assert simulated_annealing(random_sk(14, seed=6), sweeps=300, restarts=4) == pytest.approx(
        random_sk(14, seed=6).ground_state()[0]
    )
    assert random_search(problem, 4096, seed=0) >= e0 - 1e-12
    res = qaoa(problem, p=1, maxiter=40, shots=256)
    assert e0 - 1e-9 <= res.expected_energy < problem.energies.mean()


def test_train_smoke_reaches_ground_state():
    model = random_sk(6, seed=7)
    problem = IsingProblem(model)
    res = train(problem, TrainConfig(n_iters=3, group_size=4, shots=64, seq_len=8, policy_updates=2))
    assert len(res.history) == 3
    assert res.best_energy == pytest.approx(model.ground_state()[0])
    assert res.shots_used == 3 * 4 * 64
