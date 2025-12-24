"""
operators/gamma_hyp_001.py

HYP-GAM-001: Saturation pure pointwise (tanh)

FORME: T_{n+1}[i,j] = tanh(β · T_n[i,j])

PARAMÈTRES: β (force de saturation)

PRÉSUPPOSITIONS EXPLICITES:
- Transformation pointwise (pas de couplage inter-éléments)
- Markovien pur (pas de mémoire)
- Borne naturelle [-1, 1] via tanh
- Applicable à tout tenseur (rang 2 ou 3)

ATTENDU:
- Convergence monotone vers sign(D_ij)
- Possible trivialité (convergence vers signes)
"""

import numpy as np
from typing import Callable


class PureSaturationGamma:
    """
    Γ de saturation pure pointwise.
    
    AVEUGLEMENT COMPLET:
    - Ne connaît ni la dimension de l'état
    - Ne connaît ni sa structure (symétrie, bornes, etc.)
    - Ne connaît ni son interprétation
    
    Applique simplement: T_{n+1}[i,j] = tanh(β · T_n[i,j])
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Initialise l'opérateur de saturation.
        
        Args:
            beta: Force de saturation (β > 0)
                  β petit → saturation douce
                  β grand → saturation forte (vers ±1 rapidement)
        
        Exemples:
            gamma = PureSaturationGamma(beta=1.0)  # Saturation modérée
            gamma = PureSaturationGamma(beta=5.0)  # Saturation forte
        """
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique la saturation pointwise.
        
        FONCTION AVEUGLE:
        - Ne valide rien sur l'entrée
        - Ne présuppose aucune structure
        - Ne modifie pas la forme du tenseur
        
        Args:
            state: Tenseur d'état (np.ndarray de forme quelconque)
        
        Returns:
            Tenseur saturé de même forme
        
        Exemple:
            state_next = gamma(state_current)
        """
        return np.tanh(self.beta * state)
    
    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"


# ============================================================================
# FACTORY FUNCTION (pour compatibilité avec batch_runner)
# ============================================================================

def create_gamma_hyp_001(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant une instance de PureSaturationGamma.
    
    Args:
        beta: Force de saturation
    
    Returns:
        Callable (np.ndarray → np.ndarray)
    
    Exemple:
        gamma = create_gamma_hyp_001(beta=2.0)
        state_next = gamma(state)
    """
    return PureSaturationGamma(beta=beta)


# ============================================================================
# GRILLE DE PARAMÈTRES (pour Phase 1)
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 2.0},  # Valeur par défaut Phase 1
}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 0.5},
    'beta_nominal': {'beta': 1.0},
    'beta_high': {'beta': 2.0},
    'beta_very_high': {'beta': 5.0},
}

# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-001',
    'name': 'Saturation pure pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = tanh(β · T_n[i,j])',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'recommended': [0.5, 1.0, 2.0, 5.0],
            'nominal': 2.0,
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],  # Tous types
    'expected_behavior': {
        'convergence': 'monotone',
        'attractor': 'sign(D_ij)',
        'risk': 'trivialité (convergence vers signes)'
    }
}