"""
operators/gamma_hyp_007.py

GAM-007 : Régulation par moyenne glissante 8-connexe
Forme   : T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)
Famille : markovian
Applicabilité : SYM, ASY (rang 2 uniquement)

Comportement attendu :
- Lissage progressif des structures
- Homogénéisation plus douce que diffusion pure
- Conditions périodiques
"""

import numpy as np
from typing import Callable


class SlidingAverageGamma:
    """Γ de régulation par moyenne des 8 voisins."""

    def __init__(self, epsilon: float = 0.1):
        assert 0 <= epsilon <= 1, "epsilon doit être dans [0, 1]"
        self.epsilon = epsilon

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 2:
            raise ValueError(
                f"GAM-007 applicable rang 2 uniquement, reçu ndim={state.ndim}"
            )
        # Moyenne 8-voisins via décalages (conditions périodiques)
        neighbors_sum = (
            np.roll(state, ( 1,  0), axis=(0, 1)) +
            np.roll(state, (-1,  0), axis=(0, 1)) +
            np.roll(state, ( 0,  1), axis=(0, 1)) +
            np.roll(state, ( 0, -1), axis=(0, 1)) +
            np.roll(state, ( 1,  1), axis=(0, 1)) +
            np.roll(state, ( 1, -1), axis=(0, 1)) +
            np.roll(state, (-1,  1), axis=(0, 1)) +
            np.roll(state, (-1, -1), axis=(0, 1))
        )
        mean_neighbors = neighbors_sum / 8.0
        return (1 - self.epsilon) * state + self.epsilon * mean_neighbors

    def __repr__(self):
        return f"SlidingAverageGamma(epsilon={self.epsilon})"


def create(epsilon: float = 0.1) -> Callable:
    """Factory GAM-007."""
    return SlidingAverageGamma(epsilon=epsilon)


METADATA = {
    'id': 'GAM-007',
    'd_applicability': ['SYM', 'ASY'],
}
