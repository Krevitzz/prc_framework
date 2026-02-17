"""
modifiers/m1_gaussian_noise.py

M1 : Bruit gaussien additif
Forme : D' = D + N(0, σ)
Propriétés : stochastique, centré, distribution normale
Usage : Perturbation gaussienne, test robustesse
"""

import numpy as np


METADATA = {
    'id': 'M1',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}


def create(state: np.ndarray, seed_run: int = None,
           sigma: float = 0.05) -> np.ndarray:
    """
    Applique bruit gaussien additif.

    Args:
        state    : Tenseur d'état
        seed_run : Graine aléatoire (perturbations)
        sigma    : Écart-type du bruit (0.05 par défaut)

    Returns:
        Tenseur avec bruit gaussien ajouté
    """
    rng = np.random.RandomState(seed_run) if seed_run is not None else np.random
    return state + rng.randn(*state.shape) * sigma
