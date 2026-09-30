"""GQKAE: generative quantum-inspired Kolmogorov–Arnold eigensolver for H₄."""

from gqkae.molecule import H4System, reference_energies
from gqkae.train import TrainConfig, train_geometry

__all__ = [
    "H4System",
    "reference_energies",
    "TrainConfig",
    "train_geometry",
]
__version__ = "0.3.0"
