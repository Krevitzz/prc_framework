"""
operators/gamma_hyp_009.py

GAM-009 : Saturation + bruit additif gaussien
Forme   : T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)
Famille : stochastic
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Équilibre stochastique possible (β fort, σ faible)
- Marche aléatoire bornée si σ fort
- Distribution stationnaire comme attracteur
"""

import numpy as np
from typing import Callable


class StochasticSaturationGamma:
    """Γ combinant saturation et bruit additif gaussien."""

    def __init__(self, beta: float = 1.0, sigma: float = 0.01,
                 seed_run: int = None):
        assert beta > 0, "beta doit être > 0"
        assert sigma >= 0, "sigma doit être >= 0"
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed_run) if seed_run is not None else np.random

    def __call__(self, state: np.ndarray) -> np.ndarray:
        saturated = np.tanh(self.beta * state)
        if self.sigma > 0:
            return saturated + self.rng.randn(*state.shape) * self.sigma
        return saturated

    def __repr__(self):
        return f"StochasticSaturationGamma(beta={self.beta}, sigma={self.sigma})"


def create(beta: float = 1.0, sigma: float = 0.01,
           seed_run: int = None) -> Callable:
    """Factory GAM-009. seed_run paramètre ordinaire (depuis YAML)."""
    return StochasticSaturationGamma(beta=beta, sigma=sigma, seed_run=seed_run)


METADATA = {
    'id': 'GAM-009',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
