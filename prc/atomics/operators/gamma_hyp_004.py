"""
operators/gamma_hyp_004.py

GAM-004 : Décroissance exponentielle pointwise
Forme   : T_{n+1}[i,j] = T_n[i,j] · exp(-γ)
Famille : markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Convergence rapide vers zéro
- Perte totale d'information
- Attracteur trivial (zéro)
"""

import numpy as np
from typing import Callable


class ExponentialDecayGamma:
    """Γ de décroissance exponentielle vers zéro."""

    def __init__(self, gamma: float = 0.05):
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(-gamma)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return state * self._factor

    def __repr__(self):
        return f"ExponentialDecayGamma(gamma={self.gamma})"


def create(gamma: float = 0.05) -> Callable:
    """Factory GAM-004."""
    return ExponentialDecayGamma(gamma=gamma)


METADATA = {
    'id': 'GAM-004',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
