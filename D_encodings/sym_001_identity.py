"""
D_encodings/sym_001_identity.py

Matrice identité - DOF indépendants (corrélations nulles).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Diagonale C[i,i] = 1
- Structure: matrice identité
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'SYM-001',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice identité - DOF indépendants (corrélations nulles)',
    'properties': ['symmetric', 'positive_definite', 'sparse'],
    'usage': 'Test stabilité minimale, point fixe trivial'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée matrice identité.
    
    INTERPRÉTATION: DOF indépendants (corrélations nulles).
    USAGE: Test stabilité minimale, point fixe trivial.
    PROPRIÉTÉS: symétrique, définie positive, sparse
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire (ignoré, déterministe)
    
    Returns:
        Matrice identité n_dof × n_dof
    
    Examples:
        >>> I = create(3)
        >>> I.shape
        (3, 3)
        >>> np.allclose(I, np.eye(3))
        True
    """
    return np.eye(n_dof)