"""
operators/gamma_hyp_005.py

HYP-GAM-005: Oscillateur harmonique linéaire

FORME: T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}

TYPE: Non-markovien ordre 1 (mémoire explicite)

ATTENDU: Oscillations périodiques, pas de complexité émergente
"""

import numpy as np
from typing import Callable, Optional
class HarmonicOscillatorGamma:
    """
    Γ oscillateur harmonique discret.
    
    Mécanisme:
    - Oscillation sinusoïdale de chaque élément
    - Conservation énergie (norme constante théoriquement)
    - Non-markovien: nécessite T_{n-1}
    
    ATTENDU:
    - Oscillations périodiques (période 2π/ω)
    - Pas de convergence
    - Pas d'émergence de structure
    """
    
    def __init__(self, omega: float = np.pi / 4, seed: int = None):
        """
        Args:
            omega: Fréquence angulaire (rad/iteration)
        """
        self.omega = omega
        self._cos_omega = np.cos(omega)
        self._sin_omega = np.sin(omega)
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique rotation harmonique.
        
        Première itération: comportement identité.
        Suivantes: T_{n+1} = cos(ω)T_n - sin(ω)T_{n-1}
        """
        if self._previous_state is None:
            # Première itération: juste copier
            result = state.copy()
        else:
            # Oscillateur harmonique
            result = self._cos_omega * state - self._sin_omega * self._previous_state
        
        # Stocker pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire."""
        self._previous_state = None
    
    def __repr__(self):
        return f"HarmonicOscillatorGamma(omega={self.omega:.4f})"


def create_gamma_hyp_005(omega: float = np.pi / 4, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-005."""
    return HarmonicOscillatorGamma(omega=omega)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'omega': np.pi / 4}
}

PARAM_GRID_PHASE2 = {
    'omega_slow': {'omega': np.pi / 8},
    'omega_nominal': {'omega': np.pi / 4},
    'omega_fast': {'omega': np.pi / 2},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-005',
    'PHASE' : "R0",
    'name': 'Oscillateur harmonique linéaire',
    'family': 'markovian',  # Techniquement non-markovien mais famille markovienne pure
    'form': 'T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}',
    'parameters': {
        'omega': {
            'type': 'float',
            'range': '(0, π)',
            'nominal': np.pi / 4,
            'description': 'Fréquence angulaire (période = 2π/ω)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Jamais (oscillations périodiques)',
        'diversity': 'Conservation (théorique)',
        'attractors': 'Cycle périodique',
        'trivial': False
    },
    'notes': [
        'Non-markovien ordre 1 (stocke T_{n-1})',
        'Conservation énergie théorique (norme constante)',
        'Période: 2π/ω itérations',
        'Pas de complexité émergente attendue',
        'Intéressant pour test détection périodicité'
    ]
}