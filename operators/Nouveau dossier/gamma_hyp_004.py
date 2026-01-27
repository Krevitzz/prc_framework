"""
operators/gamma_hyp_004.py

HYP-GAM-004: Décroissance exponentielle pointwise

FORME: T_{n+1}[i,j] = T_n[i,j] · exp(-γ)

ATTENDU: Convergence vers 0, trivialité
"""

import numpy as np
from typing import Callable
PHASE = "R0"
class ExponentialDecayGamma:
    """
    Γ de décroissance exponentielle.
    
    Mécanisme: Atténuation exponentielle de tous les éléments → 0.
    
    ATTENDU:
    - Convergence rapide vers zéro
    - Perte totale d'information
    - Trivialité (attracteur zéro)
    """
    
    def __init__(self, gamma: float = 0.05):
        """
        Args:
            gamma: Taux de décroissance (> 0)
        """
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(-gamma)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique décroissance exponentielle.
        
        Convergence exponentielle: T → 0
        """
        return state * self._factor
    
    def __repr__(self):
        return f"ExponentialDecayGamma(gamma={self.gamma})"


def create_gamma_hyp_004(gamma: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-004."""
    return ExponentialDecayGamma(gamma=gamma)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'gamma': 0.05}
}

PARAM_GRID_PHASE2 = {
    'gamma_low': {'gamma': 0.01},
    'gamma_nominal': {'gamma': 0.05},
    'gamma_high': {'gamma': 0.1},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-004',
    
    'PHASE' : "R0",
    'name': 'Décroissance exponentielle pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] · exp(-γ)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 0.05,
            'description': 'Taux de décroissance exponentielle'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Rapide (<500 iterations)',
        'diversity': 'Perte totale',
        'attractors': 'Zéro (trivial)',
        'trivial': True
    },
    'notes': [
        'Convergence exponentielle vers 0',
        'Perte systématique information',
        'Temps caractéristique: 1/γ itérations',
        'Attendu: REJECTED[R0] pour trivialité'
    ]
}