"""
operators/gamma_hyp_012.py

HYP-GAM-012: Préservation symétrie forcée

FORME: T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = saturation

ATTENDU: Réparation artificielle, robustesse au bruit asymétrique
"""

import numpy as np
from typing import Callable
class ForcedSymmetryGamma:
    """
    Γ qui force la symétrie par symétrisation explicite.
    
    Mécanisme:
    1. Applique transformation F (saturation)
    2. Symétrise: (F + F^T) / 2
    
    ATTENDU:
    - Préservation/création symétrie garantie
    - Robustesse au bruit asymétrique
    - Test si forçage artificiel aide non-trivialité
    """
    
    def __init__(self, beta: float = 2.0, seed: int = None):
        """
        Args:
            beta: Force de saturation dans F
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique saturation puis symétrise.
        
        Applicable: rang 2 uniquement.
        """
        if state.ndim != 2:
            raise ValueError(f"ForcedSymmetryGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        # Saturation
        F = np.tanh(self.beta * state)
        
        # Symétrisation forcée
        result = (F + F.T) / 2.0
        
        return result
    
    def __repr__(self):
        return f"ForcedSymmetryGamma(beta={self.beta})"


def create_gamma_hyp_012(beta: float = 2.0, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-012."""
    return ForcedSymmetryGamma(beta=beta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 2.0}
}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 1.0},
    'beta_nominal': {'beta': 2.0},
    'beta_high': {'beta': 5.0},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-012',
    'PHASE' : "R0",
    'name': 'Préservation symétrie forcée',
    'family': 'structural',
    'form': 'T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = tanh(β·)',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 2.0,
            'description': 'Force de saturation'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Similaire GAM-001 mais symétrique',
        'diversity': 'Possible perte comme saturation pure',
        'attractors': 'Symétriques garantis',
        'trivial': 'Possible (comme GAM-001)'
    },
    'notes': [
        'Force symétrie de manière artificielle',
        'TEST-SYM-001 devrait toujours PASS',
        'TEST-SYM-002: peut créer symétrie depuis ASY',
        'Robuste au bruit asymétrique (M1, M2)',
        'Intéressant pour comparer avec GAM-001',
        'Question: forçage aide-t-il non-trivialité ?'
    ]
}