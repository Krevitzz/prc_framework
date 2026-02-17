"""
operators/gamma_hyp_001.py

GAM-001 : Saturation pure pointwise (tanh)
Forme   : T_{n+1}[i,j] = tanh(β · T_n[i,j])
Famille : markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Convergence rapide (saturation borne [-1,1])
- Attracteurs possiblement triviaux selon β
- Possible perte de diversité
"""

import numpy as np
from typing import Callable


class PureSaturationGamma:
    """Γ de saturation pure pointwise."""

    def __init__(self, beta: float = 2.0):
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * state)

    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"


def create(beta: float = 2.0, seed_run: int = None) -> Callable:
    """Factory GAM-001. seed_run ignoré (gamma déterministe)."""
    return PureSaturationGamma(beta=beta)


METADATA = {
    'id': 'GAM-001',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
