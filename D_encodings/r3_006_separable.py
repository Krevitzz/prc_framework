"""
D_encodings/r3_006_separable.py

Tenseur séparable (produit externe, rang tensoriel 1).

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Structure séparable: T_ijk = u_i · v_j · w_k
- Rang tensoriel = 1
- Bornes [-1, 1] pour vecteurs générateurs
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'R3-006',
    'rank': 3,
    'type': 'separable',
    'description': 'Tenseur séparable (produit externe, rang tensoriel 1)',
    'properties': ['separable', 'factorized', 'low_rank'],
    'usage': 'Test structure factorisée'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int) -> np.ndarray:
    """
    Crée tenseur séparable.
    
    FORME: T_ijk = u_i · v_j · w_k (produit externe)
    USAGE: Test structure factorisée
    PROPRIÉTÉS: rang tensoriel = 1
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur séparable (n_dof, n_dof, n_dof)
    
    Examples:
        >>> T = create(3, seed=42)
        >>> T.shape
        (3, 3, 3)
        >>> # Vérifier structure séparable (rang 1)
        >>> u = T[:, 0, 0]
        >>> v = T[0, :, 0]
        >>> w = T[0, 0, :]
        >>> reconstructed = np.outer(u, np.outer(v, w)).reshape(3, 3, 3)
        >>> np.allclose(T / T[0,0,0], reconstructed / reconstructed[0,0,0])
        True
    """
    
    # Génère 3 vecteurs aléatoires
    u = np.random.uniform(-1.0, 1.0, n_dof)
    v = np.random.uniform(-1.0, 1.0, n_dof)
    w = np.random.uniform(-1.0, 1.0, n_dof)
    
    # Produit externe
    T = np.outer(u, np.outer(v, w)).reshape(n_dof, n_dof, n_dof)
    
    return T