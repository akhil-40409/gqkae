"""Tests for the H₄ GQKAE / GQE / VQE stack."""

from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from gqe.grpo import grpo_loss, standardize_rewards
from gqe.model import count_params, init_transformer, transformer_logits
from gqe.molecule import H4System, qubit_hamiltonian, reference_energies
from gqe.circuit import statevector_probs
from gqe.operators import apply_operator, build_uccsd_pool, hf_occupation
from gqe.qsci import (
    bitstring_to_ab,
    build_full_cas_hamiltonian,
    evaluate_sequence,
    qsci_energy_from_counts,
)
from gqe.vqe import run_vqe


def test_h4_casci_below_hf():
    sys = H4System(bond_length=1.0)
    assert sys.n_qubits == 8
    assert sys.casci_energy() < sys.hf_energy


def test_hf_occupation_interleaved():
    sys = H4System(bond_length=1.0)
    hf = hf_occupation(sys)
    assert hf.shape == (8,)
    assert int(hf.sum()) == 4
    assert list(hf[:4]) == [1, 1, 1, 1]


def test_pool_nonempty():
    sys = H4System(bond_length=1.0)
    pool = build_uccsd_pool(sys)
    assert len(pool) > 1
    assert pool[0].kind == "id"


def test_qsci_full_matrix_recovers_casci():
    sys = H4System(bond_length=1.0)
    H, na, nb = build_full_cas_hamiltonian(sys.hamiltonian)
    evals = np.linalg.eigvalsh(H)
    assert abs(evals[0] - sys.casci_energy()) < 1e-6


def test_bitstring_roundtrip_hf():
    bits = "11110000"
    ab = bitstring_to_ab(bits, 4)
    assert ab is not None
    a, b = ab
    assert bin(a).count("1") == 2 and bin(b).count("1") == 2


def test_empty_sequence_qsci_is_hf():
    sys = H4System(bond_length=1.0)
    pool = build_uccsd_pool(sys)
    H_full, na, nb = build_full_cas_hamiltonian(sys.hamiltonian)
    res = evaluate_sequence(
        sys, pool, token_ids=[], shots=256, seed=0, H_full=H_full, na=na, nb=nb
    )
    assert abs(res.energy - sys.hf_energy) < 1e-6
    assert res.subspace_dim == 1


def test_full_subspace_qsci_is_casci():
    sys = H4System(bond_length=1.0)
    ham = sys.hamiltonian
    counts: dict[str, int] = {}
    for a in combinations(range(4), 2):
        for b in combinations(range(4), 2):
            bits = ["0"] * 8
            for i in a:
                bits[2 * i] = "1"
            for j in b:
                bits[2 * j + 1] = "1"
            counts["".join(bits)] = 1
    H_full, na, nb = build_full_cas_hamiltonian(ham)
    energy, n_valid, dim = qsci_energy_from_counts(counts, ham, H_full, na, nb)
    assert n_valid == 36
    assert dim == 36
    assert abs(energy - sys.casci_energy()) < 1e-6


def test_compiled_circuit_matches_default_qubit():
    sys = H4System(bond_length=1.0)
    pool = build_uccsd_pool(sys)
    toks = np.random.default_rng(3).integers(0, len(pool), size=20)
    dev = qml.device("default.qubit", wires=sys.n_qubits)

    @qml.qnode(dev)
    def reference():
        qml.BasisState(hf_occupation(sys), wires=range(sys.n_qubits))
        for t in toks:
            apply_operator(pool[int(t)])
        return qml.probs(wires=range(sys.n_qubits))

    assert np.allclose(statevector_probs(sys, pool, toks), reference(), atol=1e-10)


def test_qubit_hamiltonian_hf_expectation():
    sys = H4System(bond_length=1.0)
    ham = qubit_hamiltonian(sys)
    hf = hf_occupation(sys)
    dev = qml.device("default.qubit", wires=sys.n_qubits)

    @qml.qnode(dev)
    def hf_expval():
        qml.BasisState(hf, wires=range(sys.n_qubits))
        return qml.expval(ham)

    assert abs(float(hf_expval()) - sys.hf_energy) < 1e-5


def test_vqe_zero_init_is_hf_and_improves():
    sys = H4System(bond_length=1.0)
    res = run_vqe(sys, maxiter=80, rhobeg=0.4)
    assert abs(res.hf_energy - sys.hf_energy) < 1e-8
    assert res.energy <= res.hf_energy + 1e-6
    assert res.energy >= res.casci_energy - 1e-4
    assert res.n_params > 0


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


def test_reference_scan_monotonic_region():
    refs = reference_energies([0.9, 1.1, 1.5])
    assert refs["CASCI"].shape == (3,)
    assert np.all(refs["CASCI"] < refs["HF"])
