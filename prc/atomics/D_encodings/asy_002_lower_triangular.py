"""
D_encodings/asy_002_lower_triangular.py

ASY-002 : Matrice triangulaire inférieure
Forme   : A_ij = U[-1,1] si i > j, sinon 0
Rang    : 2
Propriétés : asymétrique, sparse, triangulaire
Usage   : Test orientation directionnelle
"""

import numpy as np


METADATA = {
    'id': 'ASY-002',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    """
    Crée matrice triangulaire inférieure stricte.

    Args:
        n_dof    : Nombre de degrés de liberté
        seed_CI  : Graine aléatoire (conditions initiales)

    Returns:
        Matrice triangulaire inférieure (n_dof, n_dof)
    """
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    A = rng.uniform(-1.0, 1.0, (n_dof, n_dof))
    return np.tril(A, k=-1)
