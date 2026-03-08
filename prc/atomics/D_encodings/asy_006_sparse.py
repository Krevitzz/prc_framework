"""
D_encodings/asy_006_sparse.py

ASY-006 : Matrice asymétrique sparse
Forme   : density% des éléments non-nuls, asymétrique
Rang    : 2
Propriétés : asymétrique, sparse, densité contrôlée
Usage   : Test structures creuses asymétriques
"""

import numpy as np


METADATA = {
    'id': 'ASY-006',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None, density: float = 0.2) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    A = rng.uniform(-1.0, 1.0, (n_dof, n_dof))
    mask = rng.random((n_dof, n_dof)) < density
    return A * mask
