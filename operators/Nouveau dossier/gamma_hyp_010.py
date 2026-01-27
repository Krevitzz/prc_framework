"""
operators/gamma_hyp_010.py

HYP-GAM-010: Bruit multiplicatif avec saturation

FORME: T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)

ATTENDU: Amplification sélective, risque avalanche
"""

import numpy as np
from typing import Callable
PHASE = "R0"
class MultiplicativeNoiseGamma:
    """
    Γ avec bruit multiplicatif + saturation.
    
    Mécanisme:
    - Bruit multiplicatif: amplifie grandes valeurs
    - Saturation: borne le résultat final
    - Amplification sélective selon valeur initiale
    
    ATTENDU:
    - Amplification différentielle (riches plus riches)
    - Possibilité avalanche si σ trop fort
    - Structures émergentes possibles
    """
    
    def __init__(self, sigma: float = 0.05, seed: int = None):
        """
        Args:
            sigma: Amplitude du bruit multiplicatif (≥ 0)
            seed: Graine aléatoire
        """
        assert sigma >= 0, "sigma doit être ≥ 0"
        self.sigma = sigma
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique bruit multiplicatif puis saturation.
        
        ATTENTION: σ trop grand peut causer explosions.
        """
        if self.sigma > 0:
            # Bruit multiplicatif
            noise = 1.0 + self.sigma * self.rng.randn(*state.shape)
            multiplied = state * noise
        else:
            multiplied = state
        
        # Saturation pour borner
        result = np.tanh(multiplied)
        
        return result
    
    def reset(self):
        """API consistente avec autres Γ stochastiques."""
        pass
    
    def __repr__(self):
        return f"MultiplicativeNoiseGamma(sigma={self.sigma})"


def create_gamma_hyp_010(sigma: float = 0.05, 
                         seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-010."""
    return MultiplicativeNoiseGamma(sigma=sigma, seed=seed)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'sigma': 0.05, 'seed': 42}
}

PARAM_GRID_PHASE2 = {
    'sigma_low': {'sigma': 0.01, 'seed': 42},
    'sigma_nominal': {'sigma': 0.05, 'seed': 42},
    'sigma_high': {'sigma': 0.1, 'seed': 42},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-010',
    'PHASE' : "R0",
    'name': 'Bruit multiplicatif',
    'family': 'stochastic',
    'form': 'T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)',
    'parameters': {
        'sigma': {
            'type': 'float',
            'range': '[0, +∞)',
            'nominal': 0.05,
            'description': 'Amplitude du bruit multiplicatif'
        },
        'seed': {
            'type': 'int',
            'range': 'N',
            'nominal': 42,
            'description': 'Graine aléatoire'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Variable (dépend σ)',
        'diversity': 'Possible augmentation (amplification)',
        'attractors': 'Structures amplifiées ou chaos',
        'trivial': False
    },
    'notes': [
        'Bruit multiplicatif: amplifie proportionnellement',
        'Différent de GAM-009 (bruit additif)',
        'Saturation nécessaire pour stabilité',
        'Risque avalanche si σ > 0.2',
        'Fixer seed pour reproductibilité',
        'Intéressant: compare GAM-009 vs GAM-010',
        'Peut créer hétérogénéité (riches plus riches)'
    ]
}