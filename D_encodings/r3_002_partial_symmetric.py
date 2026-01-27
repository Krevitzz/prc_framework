"""
D_encodings/r3_002_partial_symmetric.py

Tenseur symétrique partiel (indices 2-3).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Symétrie partielle T[i,j,k] = T[i,k,j]
- Bornes [-1, 1]
- Distribution uniforme
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'R3-002',
    'rank': 3,
    'type': 'partial_symmetric',
    'description': 'Tenseur symétrique partiel (indices 2-3)',
    'properties': ['partial_symmetry', 'bounded'],
    'usage': 'Test symétries partielles'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée tenseur symétrique partiel.
    
    FORME: T_ijk = T_ikj, valeurs ~ U[-1,1]
    USAGE: Test symétries partielles
    PROPRIÉTÉS: symétrie sur 2 indices (j,k)
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur avec T[i,j,k] = T[i,k,j] (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(5, seed=42)
        >>> T.shape
        (5, 5, 5)
        >>> np.allclose(T, np.transpose(T, (0, 2, 1)))
        True
    """
    
    # Génère tenseur aléatoire
    T = np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    
    # Symétrise indices 1 et 2 (j et k)
    T_sym = (T + np.transpose(T, (0, 2, 1))) / 2.0
    
    return T_sym