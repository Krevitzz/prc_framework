"""
operators/gamma_hyp_002.py

HYP-GAM-002: Diffusion pure (Laplacien discret)

FORME: T_{n+1}[i,j] = T_n[i,j] + α·(somme_voisins - 4·T_n[i,j])

APPLICABLE: SYM, ASY (rang 2 uniquement)

ATTENDU: Homogénéisation, perte diversité
"""

import numpy as np
from typing import Callable
PHASE = "R0"
class PureDiffusionGamma:
    """
    Γ de diffusion pure via Laplacien discret.
    
    Opérateur de diffusion 2D avec voisinage 4-connexe:
    - Voisins: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    - Laplacien: ∇²T[i,j] = somme_voisins - 4·T[i,j]
    - Mise à jour: T_{n+1} = T_n + α·∇²T_n
    
    AVEUGLEMENT:
    - Ne connaît ni la dimension de l'état
    - Ne connaît ni son interprétation
    - Applique simplement la diffusion locale
    """
    
    def __init__(self, alpha: float = 0.05, seed: int = None):
        """
        Args:
            alpha: Coefficient de diffusion (doit être < 0.25 pour stabilité)
        """
        assert 0 < alpha < 0.25, "alpha doit être dans (0, 0.25) pour stabilité Von Neumann"
        self.alpha = alpha
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique diffusion à l'état.
        
        Args:
            state: Tenseur d'état (doit être rang 2)
        
        Returns:
            État diffusé
        
        Raises:
            ValueError: Si state n'est pas rang 2
        """
        if state.ndim != 2:
            raise ValueError(f"PureDiffusionGamma applicable uniquement rang 2, reçu rang {state.ndim}")
        
        n, m = state.shape
        
        # Calcule Laplacien avec conditions limites périodiques
        laplacian = np.zeros_like(state)
        
        # Voisin haut
        laplacian += np.roll(state, 1, axis=0)
        # Voisin bas
        laplacian += np.roll(state, -1, axis=0)
        # Voisin gauche
        laplacian += np.roll(state, 1, axis=1)
        # Voisin droite
        laplacian += np.roll(state, -1, axis=1)
        # Centre (4 voisins)
        laplacian -= 4 * state
        
        # Mise à jour diffusive
        return state + self.alpha * laplacian
    
    def __repr__(self):
        return f"PureDiffusionGamma(alpha={self.alpha})"


def create_gamma_hyp_002(alpha: float = 0.05, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory pour créer GAM-002.
    
    Args:
        alpha: Coefficient de diffusion
    
    Returns:
        Instance callable de PureDiffusionGamma
    """
    return PureDiffusionGamma(alpha=alpha)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'alpha': 0.05}
}

PARAM_GRID_PHASE2 = {
    'alpha_low': {'alpha': 0.01},
    'alpha_nominal': {'alpha': 0.05},
    'alpha_high': {'alpha': 0.1},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-002',
    'PHASE' : "R0",
    'name': 'Diffusion pure',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])',
    'parameters': {
        'alpha': {
            'type': 'float',
            'range': '(0, 0.25)',
            'nominal': 0.05,
            'description': 'Coefficient de diffusion (stabilité: α < 0.25)'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Rapide (<500 iterations)',
        'diversity': 'Perte totale (homogénéisation)',
        'attractors': 'Uniformes (toutes valeurs égales)',
        'trivial': True
    },
    'notes': [
        'Voisinage 4-connexe avec conditions périodiques',
        'Stabilité Von Neumann: α < 0.25',
        'Lisse toute structure initiale'
    ]
}