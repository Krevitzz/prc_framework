"""
operators/gamma_hyp_006.py

HYP-GAM-006: Saturation avec mémoire ordre-1

FORME: T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))

TYPE: Non-markovien (stocke état précédent)

ATTENDU: Inertie temporelle, possibilité non-trivialité
"""

import numpy as np
from typing import Callable, Optional
class MemorySaturationGamma:
    """
    Γ avec mémoire ordre-1: combine saturation et inertie.
    
    Mécanisme:
    - Saturation: borne les valeurs via tanh
    - Mémoire: incorpore vélocité (T_n - T_{n-1})
    - Non-markovien: stocke explicitement T_{n-1}
    
    PATTERN NON-MARKOVIEN:
    Le kernel reste aveugle. C'est Γ lui-même qui gère sa mémoire
    via l'attribut interne _previous_state.
    """
    
    def __init__(self, beta: float = 1.0, alpha: float = 0.3, seed: int = None):
        """
        Args:
            beta: Force de saturation (> 0)
            alpha: Poids de la mémoire [0, 1]
        """
        assert beta > 0, "beta doit être > 0"
        assert 0 <= alpha <= 1, "alpha doit être dans [0, 1]"
        
        self.beta = beta
        self.alpha = alpha
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique Γ(T_n, T_{n-1}).
        
        À la première itération: comportement markovien (pas de mémoire).
        Itérations suivantes: utilise mémoire stockée.
        """
        if self._previous_state is None:
            # Première itération: comportement markovien
            result = np.tanh(self.beta * state)
        else:
            # Itérations suivantes: mémoire
            # Calcule vélocité
            velocity = state - self._previous_state
            
            # Combine saturation + inertie
            combined = self.beta * state + self.alpha * velocity
            result = np.tanh(combined)
        
        # Stocke état actuel pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire (utile entre runs)."""
        self._previous_state = None
    
    def __repr__(self):
        return f"MemorySaturationGamma(beta={self.beta}, alpha={self.alpha})"


def create_gamma_hyp_006(beta: float = 1.0, alpha: float = 0.3, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-006."""
    return MemorySaturationGamma(beta=beta, alpha=alpha)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 1.0, 'alpha': 0.3}
}

PARAM_GRID_PHASE2 = {
    # Mémoire faible
    'mem_weak_low_sat': {'beta': 1.0, 'alpha': 0.1},
    'mem_weak_high_sat': {'beta': 2.0, 'alpha': 0.1},
    
    # Mémoire moyenne
    'mem_mid_low_sat': {'beta': 1.0, 'alpha': 0.3},
    'mem_mid_high_sat': {'beta': 2.0, 'alpha': 0.3},
    
    # Mémoire forte
    'mem_strong_low_sat': {'beta': 1.0, 'alpha': 0.5},
    'mem_strong_high_sat': {'beta': 2.0, 'alpha': 0.5},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-006',
    
    'PHASE' : "R0",
    'name': 'Saturation + mémoire ordre-1',
    'family': 'non_markovian',
    'form': 'T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force de saturation'
        },
        'alpha': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.3,
            'description': 'Poids de la mémoire (inertie)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Plus lent que markovien (inertie)',
        'diversity': 'Possible préservation avec α adéquat',
        'attractors': 'Non-triviaux possibles',
        'trivial': False
    },
    'notes': [
        'Non-markovien: stocke T_{n-1} en interne',
        'Première itération: comportement markovien',
        'Inertie peut éviter attracteurs triviaux',
        'Appeler reset() entre runs différents'
    ]
}