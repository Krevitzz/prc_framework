"""
modifiers/m2_uniform_noise.py

Bruit uniforme additif.

PRÉSUPPOSITIONS EXPLICITES:
- Transformation: D' = D + U[-amplitude, +amplitude]
- Bruit uniforme, bornes [-amplitude, +amplitude]
- Distribution équiprobable
- Perturbation non gaussienne
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'M2',
    'type': 'noise',
    'description': 'Bruit uniforme additif',
    'properties': ['stochastic', 'bounded', 'uniform_distribution'],
    'usage': 'Perturbation uniforme, test robustesse non-gaussienne'
}

# ============ FONCTION PRINCIPALE ============
def apply(state: np.ndarray, seed: int = None, amplitude: float = 0.1) -> np.ndarray:
    """
    Applique bruit uniforme additif.
    
    FORME: D' = D + U[-amplitude, +amplitude]
    USAGE: Perturbation uniforme, test robustesse non-gaussienne
    PROPRIÉTÉS: Bruit uniforme, distribution équiprobable
    
    Args:
        state: Tenseur d'état
        seed: Graine aléatoire (reproductibilité)
        amplitude: Amplitude du bruit (0.1 par défaut)
    
    Returns:
        Tenseur avec bruit uniforme ajouté
    
    Examples:
        >>> D = np.ones((3, 3))
        >>> D_noisy = apply(D, seed=42, amplitude=0.2)
        >>> D_noisy.shape
        (3, 3)
        >>> not np.allclose(D, D_noisy)
        True
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    noise = rng.uniform(-amplitude, amplitude, size=state.shape)
    return state + noise