"""
operators/gamma_hyp_003.py

HYP-GAM-003: Croissance exponentielle pointwise

FORME: T_{n+1}[i,j] = T_n[i,j] · exp(γ)

ATTENDU: Explosion, REJECTED[GLOBAL] probable
"""

import numpy as np
from typing import Callable
class ExponentialGrowthGamma:
    """
    Γ de croissance exponentielle.
    
    Mécanisme: Amplification exponentielle de tous les éléments.
    
    ATTENDU:
    - Explosion rapide (toutes valeurs → ±∞)
    - Violation bornes systématique
    - Test de robustesse du framework aux explosions
    
    NOTE: Cet opérateur est conçu pour ÉCHOUER.
    Son rôle est de valider que le framework détecte correctement
    les explosions numériques.
    """
    
    def __init__(self, gamma: float = 0.05, seed: int = None):
        """
        Args:
            gamma: Taux de croissance (> 0)
        """
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(gamma)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique croissance exponentielle.
        
        WARNING: Diverge exponentiellement rapidement.
        """
        return state * self._factor
    
    def __repr__(self):
        return f"ExponentialGrowthGamma(gamma={self.gamma})"


def create_gamma_hyp_003(gamma: float = 0.05, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-003."""
    return ExponentialGrowthGamma(gamma=gamma)


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
    'id': 'GAM-003',
    'PHASE' : "R0",
    'name': 'Croissance exponentielle pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] · exp(γ)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 0.05,
            'description': 'Taux de croissance exponentielle'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Jamais (divergence)',
        'diversity': 'Explosion',
        'attractors': 'Aucun (divergence)',
        'trivial': False,
        'expected_failure': True
    },
    'notes': [
        'CONÇU POUR ÉCHOUER',
        'Validation détection explosions',
        'Devrait obtenir REJECTED[GLOBAL]',
        'Explosion typique < 100 itérations pour γ=0.05'
    ]
}