"""
modifiers/m1_gaussian_noise.py

Bruit gaussien additif.

PRÉSUPPOSITIONS EXPLICITES:
- Transformation: D' = D + N(0, σ)
- Bruit centré, amplitude σ
- Distribution normale
- Préserve structure moyenne
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'M1',
    'type': 'noise',
    'description': 'Bruit gaussien additif',
    'properties': ['stochastic', 'centered', 'normal_distribution'],
    'usage': 'Perturbation gaussienne, test robustesse'
}

# ============ FONCTION PRINCIPALE ============
def apply(state: np.ndarray, seed: int = None, sigma: float = 0.05) -> np.ndarray:
    """
    Applique bruit gaussien additif.
    
    FORME: D' = D + N(0, σ=0.05)
    USAGE: Perturbation gaussienne, test robustesse
    PROPRIÉTÉS: Bruit centré, distribution normale
    
    Args:
        state: Tenseur d'état
        seed: Graine aléatoire (reproductibilité)
        sigma: Écart-type du bruit (0.05 par défaut)
    
    Returns:
        Tenseur avec bruit gaussien ajouté
    
    Examples:
        >>> D = np.ones((3, 3))
        >>> D_noisy = apply(D, seed=42, sigma=0.1)
        >>> D_noisy.shape
        (3, 3)
        >>> not np.allclose(D, D_noisy)
        True
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    noise = rng.randn(*state.shape) * sigma
    return state + noise