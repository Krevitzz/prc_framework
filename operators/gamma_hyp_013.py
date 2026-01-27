"""
operators/gamma_hyp_013.py

HYP-GAM-013: Renforcement hebbien local

FORME: T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]

ATTENDU: Émergence clusters, croissance non-linéaire
"""

import numpy as np
from typing import Callable
PHASE = "R0"
class HebbianReinforcementGamma:
    """
    Γ de renforcement hebbien (apprentissage non-supervisé).
    
    Mécanisme:
    - Renforce connexions selon corrélations locales
    - Produit matriciel T @ T (auto-corrélation)
    - Apprentissage hebbien: "neurons that fire together, wire together"
    
    ATTENDU:
    - Émergence de clusters (renforcement mutuel)
    - Croissance non-linéaire (risque explosion)
    - Structures hiérarchiques possibles
    
    WARNING: Instable sans régulation (saturation recommandée).
    """
    
    def __init__(self, eta: float = 0.01):
        """
        Args:
            eta: Taux d'apprentissage [0, 0.1]
                Valeurs typiques: 0.001-0.05
        """
        assert 0 <= eta <= 0.1, "eta doit être dans [0, 0.1] pour stabilité"
        self.eta = eta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique renforcement hebbien.
        
        Applicable: rang 2 uniquement (produit matriciel).
        
        FORME:
        T_{n+1} = T_n + η·(T_n @ T_n)
        
        où @ est le produit matriciel.
        """
        if state.ndim != 2:
            raise ValueError(f"HebbianReinforcementGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"HebbianReinforcementGamma nécessite matrice carrée, reçu {state.shape}")
        
        # Produit matriciel (auto-corrélation)
        hebbian_term = state @ state
        
        # Mise à jour
        result = state + self.eta * hebbian_term
        
        return result
    
    def __repr__(self):
        return f"HebbianReinforcementGamma(eta={self.eta})"


def create_gamma_hyp_013(eta: float = 0.01) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-013."""
    return HebbianReinforcementGamma(eta=eta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'eta': 0.01}
}

PARAM_GRID_PHASE2 = {
    'eta_very_low': {'eta': 0.001},
    'eta_low': {'eta': 0.01},
    'eta_high': {'eta': 0.05},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-013',
    'PHASE' : "R0",
    'name': 'Renforcement hebbien local',
    'family': 'structural',
    'form': 'T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]',
    'parameters': {
        'eta': {
            'type': 'float',
            'range': '[0, 0.1]',
            'nominal': 0.01,
            'description': 'Taux d\'apprentissage hebbien'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 carré uniquement
    'expected_behavior': {
        'convergence': 'Instable (risque explosion)',
        'diversity': 'Augmentation (clusters)',
        'attractors': 'Structures émergentes ou explosion',
        'trivial': False
    },
    'notes': [
        'INSTABLE sans régulation additionnelle',
        'Produit matriciel T @ T (coûteux: O(N³))',
        'Nécessite matrices CARRÉES',
        'Risque explosion si η trop grand ou D mal conditionné',
        'Intéressant si combiné avec saturation (voir GAM-103[R1])',
        'Peut créer structures hiérarchiques',
        'TEST-UNIV-001 (norme) critique pour détecter explosions'
    ]
}