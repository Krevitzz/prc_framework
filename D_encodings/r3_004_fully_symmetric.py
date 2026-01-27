"""
D_encodings/r3_004_fully_symmetric.py

Tenseur totalement symétrique (6 permutations).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Symétrie totale: invariant par toutes permutations d'indices
- Bornes [-1, 1]
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'R3-004',
    'rank': 3,
    'type': 'fully_symmetric',
    'description': 'Tenseur totalement symétrique (6 permutations)',
    'properties': ['full_symmetry', 'permutation_invariant'],
    'usage': 'Test symétrie complète'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée tenseur totalement symétrique.
    
    FORME: T_ijk = T_jik = T_ikj = T_jki = T_kij = T_kji
    USAGE: Test symétrie complète
    PROPRIÉTÉS: invariant par permutation indices
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur totalement symétrique (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(3, seed=42)
        >>> T.shape
        (3, 3, 3)
        >>> np.allclose(T, np.transpose(T, (0, 2, 1)))
        True
        >>> np.allclose(T, np.transpose(T, (1, 0, 2)))
        True
    """
    
    # Génère tenseur aléatoire
    T = np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    
    # Moyenne sur toutes les permutations
    T_sym = (T + 
             np.transpose(T, (0, 2, 1)) +
             np.transpose(T, (1, 0, 2)) +
             np.transpose(T, (1, 2, 0)) +
             np.transpose(T, (2, 0, 1)) +
             np.transpose(T, (2, 1, 0))) / 6.0
    
    return T_sym