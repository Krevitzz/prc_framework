"""
D_encodings/sym_007_uniform_correlation.py

Matrice avec corrélations uniformes.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Corrélation uniforme C[i,j] = correlation (i≠j)
- Diagonale C[i,i] = 1
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'SYM-007',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice avec corrélations uniformes',
    'properties': ['symmetric', 'uniform_correlation', 'unit_diagonal'],
    'usage': 'Test corrélations égales entre tous DOF'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, correlation: float = 0.5) -> np.ndarray:
    """
    Crée matrice avec corrélations uniformes.
    
    FORME: C[i,j] = correlation (i≠j), C[i,i] = 1
    INTERPRÉTATION: Tous DOF également corrélés
    USAGE: Test corrélations égales entre tous DOF
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire (ignoré, déterministe)
        correlation: Valeur de corrélation uniforme dans [-1, 1] (0.5 par défaut)
    
    Returns:
        Matrice corrélations uniformes (n_dof, n_dof)
    
    Raises:
        AssertionError: Si correlation hors [-1, 1]
    
    Examples:
        >>> C = create(3, correlation=0.7)
        >>> C.shape
        (3, 3)
        >>> np.allclose(C[0, 1], 0.7)
        True
        >>> np.allclose(np.diag(C), 1.0)
        True
    """
    assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"
    
    C = np.full((n_dof, n_dof), correlation)
    np.fill_diagonal(C, 1.0)
    return C