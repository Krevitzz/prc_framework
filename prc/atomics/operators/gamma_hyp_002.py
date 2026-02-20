"""
operators/gamma_hyp_002.py

GAM-002 : Diffusion pure (Laplacien discret)
Forme   : T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])
Famille : markovian
Applicabilité : SYM, ASY (rang 2 uniquement)

Comportement attendu :
- Homogénéisation progressive
- Perte totale de diversité
- Stabilité Von Neumann : α < 0.25
"""

import numpy as np
from typing import Callable


class PureDiffusionGamma:
    """Γ de diffusion pure via Laplacien discret 4-connexe."""

    def __init__(self, alpha: float = 0.05):
        assert 0 < alpha < 0.25, "alpha doit être dans (0, 0.25) pour stabilité Von Neumann"
        self.alpha = alpha

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 2:
            raise ValueError(
                f"GAM-002 applicable rang 2 uniquement, reçu ndim={state.ndim}"
            )
        laplacian = (
            np.roll(state, 1, axis=0) +
            np.roll(state, -1, axis=0) +
            np.roll(state, 1, axis=1) +
            np.roll(state, -1, axis=1) -
            4 * state
        )
        return state + self.alpha * laplacian

    def __repr__(self):
        return f"PureDiffusionGamma(alpha={self.alpha})"


def create(alpha: float = 0.05) -> Callable:
    """Factory GAM-002."""
    return PureDiffusionGamma(alpha=alpha)


METADATA = {
    'id': 'GAM-002',
    'd_applicability': ['SYM', 'ASY'],
}
