"""
D_encodings/r3_001_random_uniform.py

Tenseur rang 3 aléatoire uniforme.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Aucune symétrie
- Bornes [-1, 1]
- Distribution uniforme

NOTE: Coût mémoire O(N³)
      N=20 → 8000 éléments
      N=30 → 27000 éléments
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'R3-001',
    'rank': 3,
    'type': 'random',
    'description': 'Tenseur rang 3 aléatoire uniforme',
    'properties': ['no_symmetry', 'bounded'],
    'usage': 'Test générique rang 3'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée tenseur rang 3 aléatoire uniforme.
    
    FORME: T_ijk ~ U[-1,1]
    USAGE: Test générique rang 3
    PROPRIÉTÉS: aucune symétrie
    
    Args:
        n_dof: Dimension du tenseur (N×N×N)
        seed: Graine aléatoire
    
    Returns:
        Tenseur rang 3 aléatoire (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(5, seed=42)
        >>> T.shape
        (5, 5, 5)
        >>> (T >= -1.0).all() and (T <= 1.0).all()
        True
    """
    
    return np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))