"""
D_encodings/asy_001_random_asymmetric.py

Matrice asymétrique aléatoire uniforme.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Asymétrie C[i,j] ≠ C[j,i] (général)
- Bornes [-1, 1]
- Pas de contrainte diagonale
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'ASY-001',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice asymétrique aléatoire uniforme',
    'properties': ['asymmetric', 'bounded'],
    'usage': 'Test asymétrie générique'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice asymétrique aléatoire uniforme.
    
    FORME: A_ij ~ U[-1,1] indépendants
    USAGE: Test asymétrie générique
    PROPRIÉTÉS: asymétrique, bornes [-1,1]
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice asymétrique aléatoire (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> (A >= -1.0).all() and (A <= 1.0).all()
        True
    """
    
    return np.random.uniform(-1.0, 1.0, (n_dof, n_dof))