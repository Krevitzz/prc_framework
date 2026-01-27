"""
operators/gamma_hyp_007.py

HYP-GAM-007: Régulation par moyenne glissante

FORME: T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8(T_n))

ATTENDU: Homogénéisation douce locale
"""

import numpy as np
from typing import Callable
PHASE = "R0"
class SlidingAverageGamma:
    """
    Γ de régulation par moyenne des voisins 8-connexes.
    
    Mécanisme:
    - Chaque élément est mélangé avec moyenne de ses 8 voisins
    - Lissage progressif (ε contrôle force)
    - Similaire diffusion mais avec moyenne explicite
    
    ATTENDU:
    - Lissage progressif des structures
    - Homogénéisation locale puis globale
    - Perte diversité mais plus lente que diffusion pure
    """
    
    def __init__(self, epsilon: float = 0.1, seed: int = None):
        """
        Args:
            epsilon: Force de régulation [0, 1]
                    0 = identité, 1 = moyenne pure
        """
        assert 0 <= epsilon <= 1, "epsilon doit être dans [0, 1]"
        self.epsilon = epsilon
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique régulation par moyenne locale.
        
        Applicable: rang 2 uniquement (voisinage 2D).
        """
        if state.ndim != 2:
            raise ValueError(f"SlidingAverageGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        n, m = state.shape
        result = np.zeros_like(state)
        
        # Pour chaque élément, calculer moyenne voisins 8-connexes
        for i in range(n):
            for j in range(m):
                # Voisins avec conditions périodiques
                neighbors = [
                    state[(i-1) % n, (j-1) % m],  # Haut-gauche
                    state[(i-1) % n, j],          # Haut
                    state[(i-1) % n, (j+1) % m],  # Haut-droite
                    state[i, (j-1) % m],          # Gauche
                    state[i, (j+1) % m],          # Droite
                    state[(i+1) % n, (j-1) % m],  # Bas-gauche
                    state[(i+1) % n, j],          # Bas
                    state[(i+1) % n, (j+1) % m],  # Bas-droite
                ]
                
                mean_neighbors = np.mean(neighbors)
                
                # Mélange avec moyenne
                result[i, j] = (1 - self.epsilon) * state[i, j] + self.epsilon * mean_neighbors
        
        return result
    
    def __repr__(self):
        return f"SlidingAverageGamma(epsilon={self.epsilon})"


def create_gamma_hyp_007(epsilon: float = 0.1, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-007."""
    return SlidingAverageGamma(epsilon=epsilon)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'epsilon': 0.1}
}

PARAM_GRID_PHASE2 = {
    'epsilon_low': {'epsilon': 0.05},
    'epsilon_nominal': {'epsilon': 0.1},
    'epsilon_high': {'epsilon': 0.2},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-007',
    'PHASE' : "R0",
    'name': 'Régulation moyenne glissante',
    'family': 'non_markovian',  # Bien que markovien, classé ici pour régulation
    'form': 'T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)',
    'parameters': {
        'epsilon': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.1,
            'description': 'Force de régulation (0=identité, 1=moyenne pure)'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Moyenne (500-1000 iterations)',
        'diversity': 'Perte progressive',
        'attractors': 'Uniformes (plus lent que diffusion)',
        'trivial': True
    },
    'notes': [
        'Voisinage 8-connexe (diagonales incluses)',
        'Plus doux que diffusion Laplacienne',
        'Implémentation O(N²) (peut être lent)',
        'Conditions périodiques'
    ]
}