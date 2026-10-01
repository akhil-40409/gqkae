"""Quantum-selected configuration interaction (QSCI) for the active space.

Pipeline (paper §III-B, Eq. 5–6):
  1. Measure the trial state in the computational basis → bitstrings.
  2. Keep particle-number–preserving determinants (α, β occupations).
  3. Truncate to the most frequent ``d_max`` determinants.
  4. Classically diagonalize H in that subspace → E_QSCI.
  5. Reward r = −E_QSCI.

For H₄ (4e, 4o) the full CASCI space has only 36 singlet determinants, so a
modest shot budget already approaches the exact active-space energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.fci import cistring, direct_spin1

from gqe.circuit import sample_circuit
from gqe.molecule import CASHamiltonian, H4System
from gqe.operators import ExcitationOp


@dataclass
class QSCIResult:
    energy: float
    n_sampled: int
    n_valid: int
    subspace_dim: int
    token_ids: tuple[int, ...]


def bitstring_to_ab(bits: str, norb: int) -> tuple[int, int] | None:
    """Map interleaved JW bitstring → (alpha_mask, beta_mask).

    Wire layout: qubit 2i = α_i, qubit 2i+1 = β_i  (matches PennyLane HF
    occupation [1]*N_elec + [0]*rest for closed-shell (2,2) in 4 orbitals).
    """
    if len(bits) != 2 * norb:
        return None
    alpha = 0
    beta = 0
    for i in range(norb):
        if bits[2 * i] not in "01" or bits[2 * i + 1] not in "01":
            return None
        if bits[2 * i] == "1":
            alpha |= 1 << i
        if bits[2 * i + 1] == "1":
            beta |= 1 << i
    return alpha, beta


def build_full_cas_hamiltonian(ham: CASHamiltonian) -> tuple[np.ndarray, int, int]:
    """Dense CASCI Hamiltonian matrix in the (α⊗β) FCI basis (+ e_core on diag)."""
    norb = ham.norb
    nelec = ham.nelec
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dim = na * nb
    h2e = direct_spin1.absorb_h1e(ham.h1, ham.h2, norb, nelec, 0.5)
    H = np.zeros((dim, dim), dtype=np.float64)
    eye = np.eye(dim, dtype=np.float64)
    for i in range(dim):
        c = eye[i].reshape(na, nb)
        Hc = direct_spin1.contract_2e(h2e, c, norb, nelec).ravel()
        H[:, i] = Hc
    np.fill_diagonal(H, H.diagonal() + ham.e_core)
    return H, na, nb


def det_address(alpha: int, beta: int, norb: int, nelec: tuple[int, int], nb: int) -> int:
    a = cistring.str2addr(norb, nelec[0], alpha)
    b = cistring.str2addr(norb, nelec[1], beta)
    return a * nb + b


def qsci_energy_from_counts(
    counts: dict[str, int],
    ham: CASHamiltonian,
    H_full: np.ndarray | None = None,
    na: int | None = None,
    nb: int | None = None,
    d_max: int = 2000,
) -> tuple[float, int, int]:
    """Diagonalize H in the truncated determinant subspace. Returns (E, n_valid, dim)."""
    norb = ham.norb
    nelec = ham.nelec
    if H_full is None:
        H_full, na, nb = build_full_cas_hamiltonian(ham)
    assert na is not None and nb is not None

    # Sort by frequency; keep number-preserving dets
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    addrs: list[int] = []
    seen: set[int] = set()
    n_valid = 0
    for bitstr, _ in ranked:
        ab = bitstring_to_ab(bitstr, norb)
        if ab is None:
            continue
        alpha, beta = ab
        if bin(alpha).count("1") != nelec[0] or bin(beta).count("1") != nelec[1]:
            continue
        n_valid += 1
        addr = det_address(alpha, beta, norb, nelec, nb)
        if addr in seen:
            continue
        seen.add(addr)
        addrs.append(addr)
        if len(addrs) >= d_max:
            break

    if not addrs:
        # Fallback: HF determinant only
        hf_a = sum(1 << i for i in range(nelec[0]))
        hf_b = sum(1 << i for i in range(nelec[1]))
        addrs = [det_address(hf_a, hf_b, norb, nelec, nb)]

    idx = np.asarray(addrs, dtype=int)
    H_sub = H_full[np.ix_(idx, idx)]
    evals = np.linalg.eigvalsh(H_sub)
    return float(evals[0]), n_valid, len(addrs)


def evaluate_sequence(
    system: H4System,
    pool: list[ExcitationOp],
    token_ids: list[int] | np.ndarray,
    *,
    shots: int = 2048,
    d_max: int = 2000,
    seed: int | None = None,
    H_full: np.ndarray | None = None,
    na: int | None = None,
    nb: int | None = None,
) -> QSCIResult:
    """Sample a circuit and return its QSCI energy / reward ingredients."""
    counts = sample_circuit(system, pool, token_ids, shots=shots, seed=seed)
    energy, n_valid, dim = qsci_energy_from_counts(
        counts,
        system.hamiltonian,
        H_full=H_full,
        na=na,
        nb=nb,
        d_max=d_max,
    )
    toks = tuple(int(t) for t in token_ids)
    return QSCIResult(
        energy=energy,
        n_sampled=sum(counts.values()),
        n_valid=n_valid,
        subspace_dim=dim,
        token_ids=toks,
    )
