"""
D_encodings/sym_005_banded.py

Matrice bande symétrique.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Structure bande: C[i,j] ≠ 0 ssi |i-j| ≤ bandwidth
- Diagonale C[i,i] = 1
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'SYM-005',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice bande symétrique',
    'properties': ['symmetric', 'sparse', 'banded'],
    'usage': 'Test localité structurelle'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, bandwidth: int = 3, amplitude: float = 0.5) -> np.ndarray:
    """
    Crée matrice bande symétrique.
    
    FORME: A_ij ≠ 0 ssi |i-j| ≤ bandwidth, valeurs ~ U[-amplitude, amplitude]
    USAGE: Test localité structurelle
    PROPRIÉTÉS: symétrique, sparse, bande
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
        bandwidth: Largeur de bande (3 par défaut)
        amplitude: Amplitude des valeurs hors-diagonale (0.5 par défaut)
    
    Returns:
        Matrice bande symétrique (n_dof, n_dof)
    
    Examples:
        >>> A = create(5, seed=42, bandwidth=1)
        >>> A.shape
        (5, 5)
        >>> np.allclose(A, A.T)
        True
        >>> np.allclose(np.diag(A), 1.0)
        True
        >>> A[0, 3]  # Hors bande
        0.0
    """
    
    # Initialise matrice nulle
    A = np.zeros((n_dof, n_dof))
    
    # Remplit la bande
    for i in range(n_dof):
        for j in range(max(0, i - bandwidth), min(n_dof, i + bandwidth + 1)):
            if i != j:
                value = np.random.uniform(-amplitude, amplitude)
                A[i, j] = value
                A[j, i] = value  # Symétrie
    
    # Diagonale = 1
    np.fill_diagonal(A, 1.0)
    
    return A