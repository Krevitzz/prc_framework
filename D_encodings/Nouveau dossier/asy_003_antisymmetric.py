"""
D_encodings/asy_003_antisymmetric.py

Matrice antisymétrique.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Antisymétrie A = -A^T
- Diagonale nulle A[i,i] = 0
- Bornes [-1, 1] pour partie triangulaire
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'ASY-003',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice antisymétrique',
    'properties': ['antisymmetric', 'zero_diagonal'],
    'usage': 'Test conservation antisymétrie'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice antisymétrique.
    
    FORME: A = -A^T, A_ij ~ U[-1,1] pour i>j
    USAGE: Test conservation antisymétrie
    PROPRIÉTÉS: antisymétrique (cas spécial asymétrique), diagonale nulle
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice antisymétrique (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> np.allclose(A, -A.T)
        True
        >>> np.allclose(np.diag(A), 0.0)
        True
    """
    
    # Génère partie triangulaire inférieure
    B = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    B = np.tril(B, k=-1)
    
    # Antisymétrise: A = B - B^T
    A = B - B.T
    
    return A