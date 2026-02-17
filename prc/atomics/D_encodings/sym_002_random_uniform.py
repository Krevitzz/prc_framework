"""
D_encodings/sym_002_random_uniform.py

Matrice symétrique aléatoire uniforme.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Bornes [-1, 1]
- Distribution uniforme
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'SYM-002',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice symétrique aléatoire uniforme',
    'properties': ['symmetric', 'bounded'],
    'usage': 'Test diversité maximale, générique'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice symétrique aléatoire uniforme.
    
    FORME: A = (B + B^T)/2, B_ij ~ U[-1,1]
    USAGE: Test diversité maximale, générique
    PROPRIÉTÉS: symétrique, bornes [-1,1]
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice symétrique aléatoire uniforme (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> np.allclose(A, A.T)
        True
        >>> (A >= -1.0).all() and (A <= 1.0).all()
        True
    """
    
    # Génère matrice uniforme
    B = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Symétrise
    A = (B + B.T) / 2.0
    
    return A