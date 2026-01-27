"""
D_encodings/asy_006_sparse.py

Matrice asymétrique sparse avec densité contrôlée.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Sparsité: density% d'éléments non-nuls
- Asymétrique
- Bornes [-1, 1] pour éléments non-nuls
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'ASY-006',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice asymétrique sparse avec densité contrôlée',
    'properties': ['asymmetric', 'sparse', 'controlled_density'],
    'usage': 'Test structures creuses asymétriques'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, density: float = 0.2) -> np.ndarray:
    """
    Crée matrice asymétrique sparse.
    
    FORME: density% des éléments non-nuls, asymétrique
    USAGE: Test structures creuses asymétriques
    PROPRIÉTÉS: asymétrique, sparse
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
        density: Densité de valeurs non-nulles (0.2 = 20% par défaut)
    
    Returns:
        Matrice asymétrique sparse (n_dof, n_dof)
    
    Examples:
        >>> A = create(10, seed=42, density=0.1)
        >>> A.shape
        (10, 10)
        >>> sparsity = np.sum(A == 0) / A.size
        >>> sparsity > 0.8  # Au moins 80% de zéros
        True
    """
    
    # Génère matrice complète
    A = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Masque pour sparsité
    mask = np.random.random((n_dof, n_dof)) < density
    A = A * mask
    
    return A