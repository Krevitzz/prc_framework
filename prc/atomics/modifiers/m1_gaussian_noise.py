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


METADATA = {
    'id': 'M1',
    'type': 'noise',
    'description': 'Bruit gaussien additif',
    'properties': ['stochastic', 'centered', 'normal_distribution'],
    'usage': 'Perturbation gaussienne, test robustesse'
}

# ============ FONCTION PRINCIPALE ============
# ============ FONCTION PRINCIPALE (À REMPLACER) ============
def apply(state: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Applique bruit gaussien additif.
    
    FORME: D' = D + N(0, σ=0.05)
    USAGE: Perturbation gaussienne, test robustesse
    PROPRIÉTÉS: Bruit centré, distribution normale
    
    SEED MANAGEMENT (PHASE 10):
    - Seed géré par prepare_state() en amont
    - Utilise np.random global (déjà seeded)
    - Reproductibilité garantie par seed centralisé
    
    Args:
        state: Tenseur d'état
        sigma: Écart-type du bruit (0.05 par défaut)
    
    Returns:
        Tenseur avec bruit gaussien ajouté
    
    Examples:
        >>> np.random.seed(42)  # Géré par prepare_state
        >>> D = np.ones((3, 3))
        >>> D_noisy = apply(D, sigma=0.1)
        >>> D_noisy.shape
        (3, 3)
        >>> not np.allclose(D, D_noisy)
        True
    """
    # Utilise np.random global (seeded par prepare_state)
    noise = np.random.randn(*state.shape) * sigma
    return state + noise