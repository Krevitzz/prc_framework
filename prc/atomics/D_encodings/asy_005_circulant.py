"""
D_encodings/asy_005_circulant.py

Matrice circulante asymétrique.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Structure circulante: chaque ligne = précédente décalée
- Asymétrique par construction
- Bornes [-1, 1]
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'ASY-005',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice circulante asymétrique',
    'properties': ['asymmetric', 'circulant', 'periodic_structure'],
    'usage': 'Test structure périodique asymétrique'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice circulante asymétrique.
    
    FORME: Chaque ligne est la précédente décalée d'un cran
    USAGE: Test structure périodique asymétrique
    PROPRIÉTÉS: asymétrique, structure circulante
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice circulante asymétrique (n_dof, n_dof)
    
    Examples:
        >>> A = create(4, seed=42)
        >>> A.shape
        (4, 4)
        >>> np.allclose(A[1], np.roll(A[0], 1))
        True
    """
    
    # Génère première ligne
    first_row = np.random.uniform(-1.0, 1.0, n_dof)
    
    # Construit matrice circulante
    A = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        A[i] = np.roll(first_row, i)
    
    return A