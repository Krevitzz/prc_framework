"""
operators/gamma_hyp_003.py

GAM-003 : Croissance exponentielle pointwise
Forme   : T_{n+1}[i,j] = T_n[i,j] · exp(γ)
Famille : markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Explosion rapide (toutes valeurs → ±∞)
- Conçu pour tester la robustesse du framework aux explosions
"""

import numpy as np
from typing import Callable


class ExponentialGrowthGamma:
    """Γ de croissance exponentielle — conçu pour exploser."""

    def __init__(self, gamma: float = 0.05):
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(gamma)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return state * self._factor

    def __repr__(self):
        return f"ExponentialGrowthGamma(gamma={self.gamma})"


def create(gamma: float = 0.05) -> Callable:
    """Factory GAM-003."""
    return ExponentialGrowthGamma(gamma=gamma)


METADATA = {
    'id': 'GAM-003',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
