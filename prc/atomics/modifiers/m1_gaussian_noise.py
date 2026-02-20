"""
modifiers/m1_gaussian_noise.py

M1 : Bruit gaussien additif
Forme : D' = D + N(0, σ)
Propriétés : stochastique, centré, distribution normale
Usage : Perturbation gaussienne des conditions initiales
"""

import numpy as np


METADATA = {
    'id': 'M1',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}


def create(state: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Args:
        state : Tenseur d'état
        sigma : Écart-type du bruit (0.05 par défaut)
    """
    return state + np.random.randn(*state.shape) * sigma
