"""
operators/gamma_hyp_008.py

GAM-008 : Mémoire différentielle avec saturation
Forme   : T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)
Famille : non_markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Oscillations amorties possibles
- Friction informationnelle
- Première itération : saturation simple (fallback)
"""

import numpy as np
from typing import Callable, Optional


class DifferentialMemoryGamma:
    """Γ combinant inertie, saturation et friction — non-markovien ordre 1."""

    def __init__(self, gamma: float = 0.3, beta: float = 1.0):
        assert 0 <= gamma <= 1, "gamma doit être dans [0, 1]"
        assert beta > 0, "beta doit être > 0"
        self.gamma = gamma
        self.beta = beta
        self._previous_state: Optional[np.ndarray] = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self._previous_state is None:
            result = np.tanh(self.beta * state)
        else:
            velocity = state - self._previous_state
            combined = state + self.gamma * velocity + self.beta * state
            result = np.tanh(combined)
        self._previous_state = state.copy()
        return result

    def reset(self):
        """Réinitialise mémoire — utile hors runner."""
        self._previous_state = None

    def __repr__(self):
        return f"DifferentialMemoryGamma(gamma={self.gamma}, beta={self.beta})"


def create(gamma: float = 0.3, beta: float = 1.0) -> Callable:
    """Factory GAM-008."""
    return DifferentialMemoryGamma(gamma=gamma, beta=beta)


METADATA = {
    'id': 'GAM-008',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
