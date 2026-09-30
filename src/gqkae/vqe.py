"""UCCSD-VQE baseline for the H₄ active-space Hamiltonian (paper §V-A).

Paper: CUDA-Q UCCSD + COBYLA, maxiter=5000, zero-initialized amplitudes on HF.
Here: ``lightning.qubit`` compiled with Catalyst ``@qjit`` (exact ⟨H⟩, no shots).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from catalyst import qjit
from scipy.optimize import minimize

from gqkae.molecule import H4System, qubit_hamiltonian
from gqkae.operators import hf_occupation


@dataclass
class VQEResult:
    energy: float
    hf_energy: float
    casci_energy: float
    n_params: int
    n_evals: int
    success: bool
    message: str


def run_vqe(
    system: H4System | None = None,
    *,
    bond_length: float = 1.0,
    basis: str = "6-31g",
    maxiter: int = 1000,
    rhobeg: float = 0.5,
) -> VQEResult:
    """Minimize UCCSD ⟨H⟩ with COBYLA, starting from the HF reference (θ=0)."""
    system = system or H4System(bond_length=bond_length, basis=basis)
    ham = qubit_hamiltonian(system)
    n_qubits = system.n_qubits
    electrons = system.n_electrons
    singles, doubles = qml.qchem.excitations(electrons, n_qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    hf = np.asarray(hf_occupation(system), dtype=int)
    n_params = len(singles) + len(doubles)

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qjit
    @qml.qnode(dev)
    def energy_fn(params):
        qml.UCCSD(
            params,
            wires=range(n_qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf,
        )
        return qml.expval(ham)

    x0 = np.zeros(n_params, dtype=np.float64)
    n_evals = 0
    maxiter = max(int(maxiter), n_params + 3)

    def objective(x):
        nonlocal n_evals
        n_evals += 1
        return float(energy_fn(np.asarray(x, dtype=np.float64)))

    opt = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": rhobeg, "disp": False},
    )
    return VQEResult(
        energy=float(opt.fun),
        hf_energy=float(system.hf_energy),
        casci_energy=float(system.casci_energy()),
        n_params=n_params,
        n_evals=n_evals,
        success=bool(opt.success),
        message=str(opt.message),
    )
