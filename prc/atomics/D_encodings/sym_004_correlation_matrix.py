"""
D_encodings/sym_004_correlation_matrix.py

SYM-004 : Matrice de corrélation aléatoire (SPD)
Forme   : A = C·C^T normalisée, C_ij ~ N(0,1)
Rang    : 2
Propriétés : symétrique, semi-définie positive, diagonale=1
Usage   : Test positivité définie
"""

import numpy as np


METADATA = {
    'id': 'SYM-004',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    C = rng.normal(0.0, 1.0, (n_dof, n_dof))
    A = C @ C.T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A)))
    return D_inv_sqrt @ A @ D_inv_sqrt
