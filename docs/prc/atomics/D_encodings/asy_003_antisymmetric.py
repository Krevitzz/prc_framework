"""
D_encodings/asy_003_antisymmetric.py

ASY-003 : Matrice antisymétrique
Forme   : A = B - B^T, B_ij ~ U[-1,1] triangulaire inférieure
Rang    : 2
Propriétés : antisymétrique (A = -A^T), diagonale nulle
Usage   : Test conservation antisymétrie
"""

import numpy as np


METADATA = {
    'id': 'ASY-003',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    B = np.tril(rng.uniform(-1.0, 1.0, (n_dof, n_dof)), k=-1)
    return B - B.T
