"""
operators/gamma_hyp_006.py

GAM-006 : Saturation avec mémoire ordre-1
Forme   : T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))
Famille : non_markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Inertie temporelle (convergence plus lente que markovien)
- Attracteurs non-triviaux possibles
- Première itération : comportement markovien (fallback)
"""

import numpy as np
from typing import Callable, Optional


class MemorySaturationGamma:
    """Γ combinant saturation et inertie — non-markovien ordre 1."""

    def __init__(self, beta: float = 1.0, alpha: float = 0.3):
        assert beta > 0, "beta doit être > 0"
        assert 0 <= alpha <= 1, "alpha doit être dans [0, 1]"
        self.beta = beta
        self.alpha = alpha
        self._previous_state: Optional[np.ndarray] = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self._previous_state is None:
            result = np.tanh(self.beta * state)
        else:
            velocity = state - self._previous_state
            result = np.tanh(self.beta * state + self.alpha * velocity)
        self._previous_state = state.copy()
        return result

    def reset(self):
        """Réinitialise mémoire — utile hors runner."""
        self._previous_state = None

    def __repr__(self):
        return f"MemorySaturationGamma(beta={self.beta}, alpha={self.alpha})"


def create(beta: float = 1.0, alpha: float = 0.3) -> Callable:
    """Factory GAM-006."""
    return MemorySaturationGamma(beta=beta, alpha=alpha)


METADATA = {
    'id': 'GAM-006',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
