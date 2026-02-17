"""
operators/gamma_hyp_009.py

HYP-GAM-009: Saturation + bruit additif

FORME: T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)

TYPE: Stochastique

ATTENDU: Équilibre stochastique ou diffusion aléatoire
"""

import numpy as np
from typing import Callable
class StochasticSaturationGamma:
    """
    Γ combinant saturation et bruit additif gaussien.
    
    Mécanisme:
    - Saturation borne les valeurs
    - Bruit additif injecte exploration stochastique
    - Balance déterminisme / stochasticité
    
    ATTENDU:
    - Possibilité équilibre stochastique (si β fort, σ faible)
    - Ou marche aléatoire bornée (si σ fort)
    - Robustesse au bruit à tester
    """
    
    def __init__(self, beta: float = 1.0, sigma: float = 0.01, seed: int = None):
        """
        Args:
            beta: Force de saturation (> 0)
            sigma: Amplitude du bruit (≥ 0)
            seed: Graine aléatoire pour reproductibilité
        """
        assert beta > 0, "beta doit être > 0"
        assert sigma >= 0, "sigma doit être ≥ 0"
        
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique saturation + bruit.
        
        Note: Le bruit rend le processus non-déterministe.
        """
        # Saturation
        saturated = np.tanh(self.beta * state)
        
        # Bruit gaussien
        if self.sigma > 0:
            noise = self.rng.randn(*state.shape) * self.sigma
            result = saturated + noise
        else:
            result = saturated
        
        return result
    
    def reset(self):
        """Réinitialise le générateur aléatoire (optionnel)."""
        # Note: ne réinitialise pas vraiment, juste pour API consistente
        pass
    
    def __repr__(self):
        return f"StochasticSaturationGamma(beta={self.beta}, sigma={self.sigma})"


def create_gamma_hyp_009(beta: float = 1.0, sigma: float = 0.01, 
                         seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-009."""
    return StochasticSaturationGamma(beta=beta, sigma=sigma, seed=seed)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 1.0, 'sigma': 0.01, 'seed': 42}
}

PARAM_GRID_PHASE2 = {
    # Saturation faible, bruit faible
    'low_sat_low_noise': {'beta': 1.0, 'sigma': 0.01, 'seed': 42},
    
    # Saturation forte, bruit faible (équilibre possible)
    'high_sat_low_noise': {'beta': 2.0, 'sigma': 0.01, 'seed': 42},
    
    # Saturation faible, bruit fort (marche aléatoire)
    'low_sat_high_noise': {'beta': 1.0, 'sigma': 0.05, 'seed': 42},
    
    # Saturation forte, bruit fort
    'high_sat_high_noise': {'beta': 2.0, 'sigma': 0.05, 'seed': 42},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-009',
    
    'PHASE' : "R0",
    'name': 'Saturation + bruit additif',
    'family': 'stochastic',
    'form': 'T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force de saturation'
        },
        'sigma': {
            'type': 'float',
            'range': '[0, +∞)',
            'nominal': 0.01,
            'description': 'Amplitude du bruit gaussien'
        },
        'seed': {
            'type': 'int',
            'range': 'N',
            'nominal': 42,
            'description': 'Graine aléatoire (reproductibilité)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Équilibre stochastique possible',
        'diversity': 'Maintien possible avec σ adéquat',
        'attractors': 'Distribution stationnaire',
        'trivial': False
    },
    'notes': [
        'Processus stochastique (non-déterministe)',
        'Fixer seed pour reproductibilité',
        'Balance déterminisme (β) / exploration (σ)',
        'TEST-UNIV-004 (sensibilité CI) particulièrement pertinent',
        'Moyenner sur plusieurs seeds pour analyses'
    ]
}