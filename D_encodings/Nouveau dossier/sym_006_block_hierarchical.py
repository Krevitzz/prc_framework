"""
D_encodings/sym_006_block_hierarchical.py

Matrice hiérarchique par blocs.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Structure blocs: forte corrélation intra-bloc, faible inter-blocs
- Diagonale C[i,i] = 1
- Contrainte: n_dof divisible par n_blocks
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'SYM-006',
    'rank': 2,
    'type': 'symmetric',
    'description': 'Matrice hiérarchique par blocs',
    'properties': ['symmetric', 'block_structure', 'hierarchical'],
    'usage': 'Test préservation structure modulaire'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, n_blocks: int = 10, 
           intra_corr: float = 0.7, inter_corr: float = 0.1) -> np.ndarray:
    """
    Crée matrice hiérarchique par blocs.
    
    FORME: Blocs denses intra (corrélation forte), sparse inter (corrélation faible)
    USAGE: Test préservation structure modulaire
    PROPRIÉTÉS: symétrique, structure blocs
    
    Args:
        n_dof: Nombre de degrés de liberté (doit être divisible par n_blocks)
        seed: Graine aléatoire
        n_blocks: Nombre de blocs (10 par défaut)
        intra_corr: Corrélation intra-bloc (0.7 par défaut)
        inter_corr: Corrélation inter-blocs (0.1 par défaut)
    
    Returns:
        Matrice hiérarchique par blocs (n_dof, n_dof)
    
    Raises:
        AssertionError: Si n_dof non divisible par n_blocks
    
    Examples:
        >>> A = create(20, seed=42, n_blocks=4)
        >>> A.shape
        (20, 20)
        >>> np.allclose(A, A.T)
        True
        >>> np.allclose(np.diag(A), 1.0)
        True
    """
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
    
    
    block_size = n_dof // n_blocks
    
    # Initialise avec corrélation inter-blocs
    A = np.full((n_dof, n_dof), inter_corr)
    
    # Remplit blocs avec corrélation intra-bloc
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        A[start:end, start:end] = intra_corr
    
    # Diagonale = 1
    np.fill_diagonal(A, 1.0)
    
    # Ajoute légère variation aléatoire
    noise = np.random.normal(0.0, 0.05, (n_dof, n_dof))
    noise = (noise + noise.T) / 2  # Symétrise
    A = A + noise
    
    # Clip dans [-1, 1]
    A = np.clip(A, -1.0, 1.0)
    np.fill_diagonal(A, 1.0)  # Restaure diagonale
    
    return A