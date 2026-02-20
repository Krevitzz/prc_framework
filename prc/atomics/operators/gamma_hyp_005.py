"""
operators/gamma_hyp_005.py

GAM-005 : Oscillateur harmonique linéaire
Forme   : T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}
Famille : non_markovian
Applicabilité : SYM, ASY, R3

Comportement attendu :
- Oscillations périodiques (période 2π/ω)
- Conservation énergie théorique
- Pas d'émergence de structure
"""

import numpy as np
from typing import Callable, Optional


class HarmonicOscillatorGamma:
    """Γ oscillateur harmonique discret — non-markovien ordre 1."""

    def __init__(self, omega: float = np.pi / 4):
        self.omega = omega
        self._cos_omega = np.cos(omega)
        self._sin_omega = np.sin(omega)
        self._previous_state: Optional[np.ndarray] = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self._previous_state is None:
            result = state.copy()
        else:
            result = self._cos_omega * state - self._sin_omega * self._previous_state
        self._previous_state = state.copy()
        return result

    def reset(self):
        """Réinitialise mémoire — utile hors runner."""
        self._previous_state = None

    def __repr__(self):
        return f"HarmonicOscillatorGamma(omega={self.omega:.4f})"


def create(omega: float = np.pi / 4) -> Callable:
    """Factory GAM-005."""
    return HarmonicOscillatorGamma(omega=omega)


METADATA = {
    'id': 'GAM-005',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
