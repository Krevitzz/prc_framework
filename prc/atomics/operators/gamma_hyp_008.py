"""
operators/gamma_hyp_008.py

HYP-GAM-008: Mémoire différentielle avec saturation

FORME: T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)

ATTENDU: "Friction informationnelle", oscillations contrôlées
"""

import numpy as np
from typing import Callable, Optional
class DifferentialMemoryGamma:
    """
    Γ combinant inertie, saturation et friction.
    
    Mécanisme:
    - Inertie: γ·(T_n - T_{n-1}) (vélocité)
    - Saturation: β·T_n (force de rappel)
    - Friction: contrôle via saturation globale
    
    ATTENDU:
    - Oscillations amorties si γ et β bien balancés
    - Friction informationnelle (ralentissement)
    - Convergence douce possible
    """
    
    def __init__(self, gamma: float = 0.3, beta: float = 1.0, seed: int = None):
        """
        Args:
            gamma: Poids inertie [0, 1]
            beta: Force saturation (> 0)
        """
        assert 0 <= gamma <= 1, "gamma doit être dans [0, 1]"
        assert beta > 0, "beta doit être > 0"
        
        self.gamma = gamma
        self.beta = beta
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique mémoire différentielle.
        
        Première itération: comportement markovien.
        """
        if self._previous_state is None:
            # Première itération: saturation simple
            result = np.tanh(self.beta * state)
        else:
            # Vélocité
            velocity = state - self._previous_state
            
            # Combinaison
            combined = state + self.gamma * velocity + self.beta * state
            
            # Saturation globale
            result = np.tanh(combined)
        
        # Stocker pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire."""
        self._previous_state = None
    
    def __repr__(self):
        return f"DifferentialMemoryGamma(gamma={self.gamma}, beta={self.beta})"


def create_gamma_hyp_008(gamma: float = 0.3, 
                         beta: float = 1.0, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-008."""
    return DifferentialMemoryGamma(gamma=gamma, beta=beta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'gamma': 0.3, 'beta': 1.0}
}

PARAM_GRID_PHASE2 = {
    # Inertie faible
    'low_inertia_low_sat': {'gamma': 0.1, 'beta': 1.0},
    'low_inertia_high_sat': {'gamma': 0.1, 'beta': 2.0},
    
    # Inertie forte
    'high_inertia_low_sat': {'gamma': 0.5, 'beta': 1.0},
    'high_inertia_high_sat': {'gamma': 0.5, 'beta': 2.0},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-008',
    'PHASE' : "R0",
    'name': 'Mémoire différentielle',
    'family': 'non_markovian',
    'form': 'T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.3,
            'description': 'Poids inertie (vélocité)'
        },
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force saturation'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Oscillations amorties possibles',
        'diversity': 'Maintien possible avec γ adéquat',
        'attractors': 'Non-triviaux possibles',
        'trivial': False
    },
    'notes': [
        'Non-markovien ordre 1',
        'Combine inertie + saturation + friction',
        'Balance γ (inertie) vs β (saturation)',
        'Similaire GAM-006 mais avec terme β additionnel',
        'Appeler reset() entre runs différents',
        'Oscillations amorties si bien paramétré'
    ]
}