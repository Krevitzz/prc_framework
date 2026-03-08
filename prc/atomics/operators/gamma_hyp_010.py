"""
operators/gamma_hyp_010.py

GAM-010 : Bruit multiplicatif avec saturation
Forme   : T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)
Famille : stochastic
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Amplification différentielle (grandes valeurs amplifiées davantage)
- Risque avalanche si σ trop fort
- Structures émergentes possibles
"""

import numpy as np
from typing import Callable


class MultiplicativeNoiseGamma:
    """Γ avec bruit multiplicatif + saturation."""

    def __init__(self, sigma: float = 0.05, seed_run: int = None):
        assert sigma >= 0, "sigma doit être >= 0"
        self.sigma = sigma
        self.rng = np.random.RandomState(seed_run) if seed_run is not None else np.random

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.sigma > 0:
            noise = 1.0 + self.sigma * self.rng.randn(*state.shape)
            return np.tanh(state * noise)
        return np.tanh(state)

    def __repr__(self):
        return f"MultiplicativeNoiseGamma(sigma={self.sigma})"


def create(sigma: float = 0.05, seed_run: int = None) -> Callable:
    """Factory GAM-010. seed_run paramètre ordinaire (depuis YAML)."""
    return MultiplicativeNoiseGamma(sigma=sigma, seed_run=seed_run)


METADATA = {
    'id': 'GAM-010',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
