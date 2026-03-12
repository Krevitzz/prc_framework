"""
operators/gamma_hyp_012.py

GAM-012 : Préservation symétrie forcée
Forme   : T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = tanh(β·)
Famille : structural
Applicabilité : SYM, ASY (rang 2 uniquement)

Comportement attendu :
- Symétrie garantie à chaque itération
- Robustesse au bruit asymétrique
- Convergence similaire GAM-001
"""

import numpy as np
from typing import Callable


class ForcedSymmetryGamma:
    """Γ qui force la symétrie par symétrisation explicite."""

    def __init__(self, beta: float = 2.0):
        assert beta > 0, "beta doit être > 0"
        self.beta = beta

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 2:
            raise ValueError(
                f"GAM-012 applicable rang 2 uniquement, reçu ndim={state.ndim}"
            )
        F = np.tanh(self.beta * state)
        return (F + F.T) / 2.0

    def __repr__(self):
        return f"ForcedSymmetryGamma(beta={self.beta})"


def create(beta: float = 2.0) -> Callable:
    """Factory GAM-012."""
    return ForcedSymmetryGamma(beta=beta)


METADATA = {
    'id': 'GAM-012',
    'd_applicability': ['SYM', 'ASY'],
}
