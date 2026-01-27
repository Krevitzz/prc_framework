"""
D_encodings/r3_007_block_structure.py

Tenseur par blocs 3D.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Structure par blocs dans les 3 dimensions
- Blocs diagonaux denses, couplage inter-blocs faible
- Contrainte: n_dof divisible par n_blocks
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'R3-007',
    'rank': 3,
    'type': 'block_structure',
    'description': 'Tenseur par blocs 3D',
    'properties': ['block_structure', 'hierarchical', 'modular'],
    'usage': 'Test modularité rang 3'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, n_blocks: int = 4) -> np.ndarray:
    """
    Crée tenseur par blocs.
    
    FORME: Structure par blocs dans les 3 dimensions
    USAGE: Test modularité rang 3
    PROPRIÉTÉS: structure hiérarchique
    
    Args:
        n_dof: Dimension du tenseur (doit être divisible par n_blocks)
        seed: Graine aléatoire
        n_blocks: Nombre de blocs par dimension (4 par défaut)
    
    Returns:
        Tenseur par blocs (n_dof, n_dof, n_dof)
    
    Raises:
        AssertionError: Si n_dof non divisible par n_blocks
    
    Examples:
        >>> T = create(8, seed=42, n_blocks=2)
        >>> T.shape
        (8, 8, 8)
        >>> # Bloc diagonal 0 devrait être plus dense
        >>> bloc0 = T[0:4, 0:4, 0:4]
        >>> inter_bloc = T[0:4, 4:8, 0:4]
        >>> np.abs(bloc0).mean() > np.abs(inter_bloc).mean()
        True
    """
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
   
    
    block_size = n_dof // n_blocks
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit blocs diagonaux (i_block = j_block = k_block)
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        T[start:end, start:end, start:end] = np.random.uniform(
            -1.0, 1.0, (block_size, block_size, block_size)
        )
    
    # Ajoute faible couplage inter-blocs
    T += np.random.uniform(-0.1, 0.1, (n_dof, n_dof, n_dof))
    
    return T