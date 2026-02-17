"""
D_encodings/sym_004_correlation_matrix.py

Matrice de corrélation aléatoire (semi-définie positive).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Semi-définie positive (SPD)
- Diagonale C[i,i] = 1
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'SYM-004',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice de corrélation aléatoire (SPD)',
    'properties': ['symmetric', 'positive_semidefinite', 'unit_diagonal'],
    'usage': 'Test positivité définie'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice de corrélation (SPD).
    
    FORME: A = C·C^T normalisée, C_ij ~ N(0,1)
    USAGE: Test positivité définie
    PROPRIÉTÉS: symétrique, définie positive, diag=1
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice de corrélation (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42)
        >>> A.shape
        (3, 3)
        >>> np.allclose(A, A.T)
        True
        >>> np.allclose(np.diag(A), 1.0)
        True
        >>> np.all(np.linalg.eigvalsh(A) >= -1e-10)
        True
    """
    
    # Génère matrice gaussienne
    C = np.random.normal(0.0, 1.0, (n_dof, n_dof))
    
    # Produit C·C^T pour garantir positivité
    A = C @ C.T
    
    # Normalise pour avoir diagonale = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A)))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A