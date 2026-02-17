"""
D_encodings/r3_005_diagonal.py

Tenseur diagonal (éléments non-nuls ssi i=j=k).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Structure diagonale pure: T_ijk ≠ 0 ssi i=j=k
- Très sparse
- Bornes [-1, 1] pour éléments diagonaux
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'R3-005',
    'rank': 3,
    'type': 'diagonal',
    'description': 'Tenseur diagonal (T_ijk ≠ 0 ssi i=j=k)',
    'properties': ['diagonal', 'very_sparse'],
    'usage': 'Test structure diagonale rang 3'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée tenseur diagonal.
    
    FORME: T_ijk ≠ 0 ssi i=j=k
    USAGE: Test structure diagonale rang 3
    PROPRIÉTÉS: très sparse, diagonal
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur diagonal (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(5, seed=42)
        >>> T.shape
        (5, 5, 5)
        >>> T[0, 0, 1]  # Hors diagonale
        0.0
        >>> T[2, 2, 2] != 0.0  # Sur diagonale
        True
        >>> sparsity = np.sum(T == 0) / T.size
        >>> sparsity > 0.95  # Très sparse
        True
    """
    
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit seulement diagonale principale
    for i in range(n_dof):
        T[i, i, i] = np.random.uniform(-1.0, 1.0)
    
    return T