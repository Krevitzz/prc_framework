"""
D_encodings/sym_008_random_clipped.py

Matrice symétrique aléatoire avec paramètres mean/std.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Distribution normale paramétrable (mean, std)
- Clipping dans [-1, 1]
- Diagonale C[i,i] = 1
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'SYM-008',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice symétrique aléatoire avec paramètres mean/std',
    'properties': ['symmetric', 'parametric_normal', 'clipped', 'unit_diagonal'],
    'usage': 'Test distribution normale paramétrable avec clipping'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, mean: float = 0.0, std: float = 0.3) -> np.ndarray:
    """
    Crée matrice symétrique aléatoire avec paramètres mean/std.
    
    FORME: C = (N(mean, std) + N(mean, std)^T)/2, clippé [-1,1], diag=1
    USAGE: Test distribution normale paramétrable avec clipping
    PROPRIÉTÉS: symétrique, distribution paramétrable, clippé, diagonale=1
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
        mean: Moyenne des corrélations (0.0 par défaut)
        std: Écart-type des corrélations (0.3 par défaut)
    
    Returns:
        Matrice symétrique aléatoire clippée (n_dof, n_dof)
    
    Examples:
        >>> C = create(3, seed=42, mean=0.0, std=0.5)
        >>> C.shape
        (3, 3)
        >>> np.allclose(C, C.T)
        True
        >>> (C >= -1.0).all() and (C <= 1.0).all()
        True
        >>> np.allclose(np.diag(C), 1.0)
        True
    """
    
    # Génère matrice aléatoire
    C = np.random.normal(mean, std, (n_dof, n_dof))
    
    # Symétrise
    C = (C + C.T) / 2
    
    # Diagonale = 1
    np.fill_diagonal(C, 1.0)
    
    # Clip dans [-1, 1]
    C = np.clip(C, -1.0, 1.0)
    
    return C