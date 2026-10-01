"""What the generator needs from a problem, and nothing else.

A problem hands over a gate pool, a starting state, and a way to turn
measurement counts into a score (lower is better). The transformer, GRPO and
compiled circuits never look inside.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from gqe.gates import Gate


class Problem(Protocol):
    n_qubits: int
    pool: list[Gate]  # pool[0] must be the identity; it doubles as BOS
    init: Literal["zero", "plus"]

    def score(self, counts: np.ndarray) -> float:
        """counts[i] = shots on basis state i (wire 0 = most significant bit)."""
        ...

    def best_sample(self, counts: np.ndarray) -> tuple[float, int]:
        """(energy, basis index) of the best bitstring that was actually measured."""
        ...
