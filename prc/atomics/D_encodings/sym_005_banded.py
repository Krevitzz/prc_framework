"""
D_encodings/sym_005_banded.py

SYM-005 : Matrice bande symétrique
Forme   : A_ij ≠ 0 ssi |i-j| ≤ bandwidth, valeurs ~ U[-amplitude, amplitude]
Rang    : 2
Propriétés : symétrique, sparse, bande
Usage   : Test localité structurelle

Optimisation : double boucle → np.triu_indices vectorisé
"""

import numpy as np


METADATA = {
    'id': 'SYM-005',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None,
           bandwidth: int = 3, amplitude: float = 0.5) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    A = np.zeros((n_dof, n_dof))

    # Indices dans la bande (hors diagonale)
    rows, cols = np.triu_indices(n_dof, k=1)
    mask = (cols - rows) <= bandwidth
    rows, cols = rows[mask], cols[mask]

    values = rng.uniform(-amplitude, amplitude, len(rows))
    A[rows, cols] = values
    A[cols, rows] = values  # Symétrie

    np.fill_diagonal(A, 1.0)
    return A
