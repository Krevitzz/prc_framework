"""
D_encodings/r3_003_local_coupling.py

Tenseur avec couplages locaux.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Localité géométrique: T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2*radius
- Sparse par construction
- Bornes [-1, 1] pour éléments non-nuls
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'R3-003',
    'rank': 3,
    'type': 'local_coupling',
    'description': 'Tenseur avec couplages locaux',
    'properties': ['sparse', 'geometric_locality'],
    'usage': 'Test localité 3-corps'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, radius: int = 2) -> np.ndarray:
    """
    Crée tenseur avec couplages locaux.
    
    FORME: T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2*radius
    USAGE: Test localité 3-corps
    PROPRIÉTÉS: sparse, localité géométrique
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
        radius: Rayon de localité (2 par défaut → 2*radius=4)
    
    Returns:
        Tenseur sparse avec couplages locaux (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(5, seed=42, radius=1)
        >>> T.shape
        (5, 5, 5)
        >>> T[0, 0, 4]  # Hors localité
        0.0
    """
    
    # Initialise tenseur nul
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit selon contrainte de localité
    for i in range(n_dof):
        for j in range(n_dof):
            for k in range(n_dof):
                distance = abs(i - j) + abs(j - k) + abs(k - i)
                if distance <= 2 * radius:
                    T[i, j, k] = np.random.uniform(-1.0, 1.0)
    
    return T