"""
D_encodings/sym_003_random_gaussian.py

Matrice symétrique aléatoire gaussienne.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Distribution gaussienne N(0, σ)
- Non bornée a priori
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'SYM-003',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice symétrique aléatoire gaussienne',
    'properties': ['symmetric', 'unbounded'],
    'usage': 'Test continuité, distribution normale'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, sigma: float = 0.3) -> np.ndarray:
    """
    Crée matrice symétrique gaussienne.
    
    FORME: A = (B + B^T)/2, B_ij ~ N(0, σ=0.3)
    USAGE: Test continuité, distribution normale
    PROPRIÉTÉS: symétrique, non bornée a priori
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
        sigma: Écart-type distribution gaussienne (0.3 par défaut)
    
    Returns:
        Matrice symétrique gaussienne (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> np.allclose(A, A.T)
        True
    """
    
    # Génère matrice gaussienne
    B = np.random.normal(0.0, sigma, (n_dof, n_dof))
    
    # Symétrise
    A = (B + B.T) / 2.0
    
    return A