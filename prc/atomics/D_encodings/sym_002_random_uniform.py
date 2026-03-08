"""
D_encodings/sym_002_random_uniform.py

SYM-002 : Matrice symétrique aléatoire uniforme
Forme   : A = (B + B^T)/2, B_ij ~ U[-1,1]
Rang    : 2
Propriétés : symétrique, bornes [-1,1]
Usage   : Test diversité maximale, générique
"""

import numpy as np


METADATA = {
    'id': 'SYM-002',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    B = rng.uniform(-1.0, 1.0, (n_dof, n_dof))
    return (B + B.T) / 2.0
