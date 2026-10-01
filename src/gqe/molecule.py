"""Linear H₄ molecule: PySCF active-space integrals and classical references.

Paper setup (arXiv:2605.04604 §V-A): H₄ (4e, 4o) → 8 qubits, basis 6-31G,
active space centered on the HOMO–LUMO gap. Geometry is a linear chain with
uniform nearest-neighbour spacing R (Å).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pyscf import ao2mo, cc, gto, mcscf, scf


@dataclass(frozen=True)
class CASHamiltonian:
    """Active-space electronic Hamiltonian in the MO basis."""

    h1: np.ndarray  # (norb, norb)
    h2: np.ndarray  # (norb, norb, norb, norb) chemist notation
    e_core: float
    norb: int
    nelec: tuple[int, int]


@dataclass
class H4System:
    """H₄ (4e, 4o) at bond length ``bond_length`` Å."""

    bond_length: float
    basis: str = "6-31g"
    norb: int = 4
    nelec: tuple[int, int] = (2, 2)

    def __post_init__(self) -> None:
        atoms = make_linear_h4(self.bond_length)
        self.mol = gto.M(atom=atoms, basis=self.basis, charge=0, spin=0, unit="Angstrom", verbose=0)
        self.mf = scf.RHF(self.mol)
        self.mf.verbose = 0
        self.mf.kernel()
        self.mc = mcscf.CASCI(self.mf, self.norb, self.nelec)
        self.active = _active_indices(self.mol, self.mf, self.norb, self.nelec)
        self.hamiltonian = _cas_hamiltonian(
            self.mc, self.mf.mo_coeff, self.norb, self.nelec
        )
        self.n_qubits = 2 * self.norb
        self.n_electrons = int(sum(self.nelec))

    @property
    def hf_energy(self) -> float:
        return float(self.mf.e_tot)

    def casci_energy(self) -> float:
        e, _ = self.mc.fcisolver.kernel(
            self.hamiltonian.h1,
            self.hamiltonian.h2,
            self.norb,
            self.nelec,
            ecore=self.hamiltonian.e_core,
        )
        return float(e)

    def ccsd_energy(self) -> float:
        nmo = self.mf.mo_coeff.shape[1]
        frozen = [i for i in range(nmo) if i not in set(self.active)]
        mycc = cc.RCCSD(self.mf, frozen=frozen)
        mycc.verbose = 0
        mycc.kernel()
        return float(self.mf.e_tot + mycc.e_corr)


def make_linear_h4(bond_length: float) -> list[list]:
    """H–H–H–H along z with nearest-neighbour distance ``bond_length`` Å."""
    return [["H", [0.0, 0.0, i * bond_length]] for i in range(4)]


def _active_indices(mol, mf, norb: int, nelec: tuple[int, int]) -> list[int]:
    ncore = (mol.nelectron - sum(nelec)) // 2
    idx = list(range(ncore, ncore + norb))
    occ = int(np.sum(mf.mo_occ[idx]))
    if occ != sum(nelec):
        raise RuntimeError(f"Active-space electron count mismatch: {occ} vs {sum(nelec)}")
    return idx


def _cas_hamiltonian(mc, mo_coeff, norb: int, nelec: tuple[int, int]) -> CASHamiltonian:
    h1, e_core = mc.get_h1cas(mo_coeff)
    h2 = ao2mo.restore(1, mc.get_h2cas(mo_coeff), norb)
    return CASHamiltonian(
        h1=np.asarray(h1, dtype=np.float64),
        h2=np.asarray(h2, dtype=np.float64),
        e_core=float(e_core),
        norb=norb,
        nelec=nelec,
    )


def qubit_hamiltonian(system: H4System):
    """Jordan–Wigner qubit Hamiltonian of the CAS electronic Hamiltonian.

    Two-electron integrals are stored in chemist notation ``(pq|rs)``. PennyLane's
    ``fermionic_observable`` expects the physicist-style tensor obtained by
    ``swapaxes(1, 3)``, matching ``pennylane.qchem``'s PySCF conversion path.
    """
    ham = system.hamiltonian
    two = np.swapaxes(np.asarray(ham.h2, dtype=np.float64), 1, 3)
    ferm = qml.qchem.fermionic_observable(
        np.array([ham.e_core], dtype=np.float64),
        np.asarray(ham.h1, dtype=np.float64),
        two,
    )
    return qml.qchem.qubit_observable(ferm)


def reference_energies(bond_lengths: np.ndarray | list[float], basis: str = "6-31g") -> dict:
    """HF / CCSD / CASCI along a bond-length scan (Fig. 4a classical baselines).

    CASCI is the exact ground state in the (4e, 4o) active space — i.e. active-space FCI.
    """
    bond_lengths = np.asarray(bond_lengths, dtype=float)
    out = {"R": bond_lengths, "HF": [], "CCSD": [], "CASCI": []}
    for r in bond_lengths:
        sys = H4System(bond_length=float(r), basis=basis)
        out["HF"].append(sys.hf_energy)
        out["CCSD"].append(sys.ccsd_energy())
        out["CASCI"].append(sys.casci_energy())
    for k in ("HF", "CCSD", "CASCI"):
        out[k] = np.asarray(out[k], dtype=float)
    return out
