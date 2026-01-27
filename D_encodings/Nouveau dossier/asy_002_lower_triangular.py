"""
D_encodings/asy_002_lower_triangular.py

Matrice triangulaire inférieure.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Structure triangulaire inférieure stricte
- A_ij = 0 si i ≤ j
- Bornes [-1, 1] pour éléments non-nuls
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'ASY-002',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice triangulaire inférieure',
    'properties': ['asymmetric', 'sparse', 'triangular'],
    'usage': 'Test orientation directionnelle'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice triangulaire inférieure.
    
    FORME: A_ij = U[-1,1] si i > j, sinon 0
    USAGE: Test orientation directionnelle
    PROPRIÉTÉS: asymétrique, sparse, triangulaire
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice triangulaire inférieure (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> np.allclose(A[0, :], 0.0)
        True
        >>> np.allclose(np.diag(A), 0.0)
        True
    """
    
    # Génère matrice aléatoire
    A = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Garde seulement triangulaire inférieure (strict)
    A = np.tril(A, k=-1)
    
    return A