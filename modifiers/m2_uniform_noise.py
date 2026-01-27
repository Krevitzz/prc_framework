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
# ============ FONCTION PRINCIPALE (À REMPLACER) ============
def apply(state: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
    """
    Applique bruit uniforme additif.
    
    FORME: D' = D + U[-amplitude, +amplitude]
    USAGE: Perturbation uniforme, test robustesse non-gaussienne
    PROPRIÉTÉS: Bruit uniforme, distribution équiprobable
    
    SEED MANAGEMENT (PHASE 10):
    - Seed géré par prepare_state() en amont
    - Utilise np.random global (déjà seeded)
    - Reproductibilité garantie par seed centralisé
    
    Args:
        state: Tenseur d'état
        amplitude: Amplitude du bruit (0.1 par défaut)
    
    Returns:
        Tenseur avec bruit uniforme ajouté
    
    Examples:
        >>> np.random.seed(42)  # Géré par prepare_state
        >>> D = np.ones((3, 3))
        >>> D_noisy = apply(D, amplitude=0.2)
        >>> D_noisy.shape
        (3, 3)
        >>> not np.allclose(D, D_noisy)
        True
    """
    # Utilise np.random global (seeded par prepare_state)
    noise = np.random.uniform(-amplitude, amplitude, size=state.shape)
    return state + noise