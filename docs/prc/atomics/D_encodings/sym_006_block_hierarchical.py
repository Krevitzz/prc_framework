"""
D_encodings/sym_006_block_hierarchical.py

SYM-006 : Matrice hiérarchique par blocs
Forme   : Blocs denses intra (intra_corr), sparse inter (inter_corr)
Rang    : 2
Propriétés : symétrique, structure blocs, diagonale=1
Usage   : Test préservation structure modulaire
"""

import numpy as np


METADATA = {
    'id': 'SYM-006',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None,
           n_blocks: int = 10, intra_corr: float = 0.7,
           inter_corr: float = 0.1) -> np.ndarray:
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random

    block_size = n_dof // n_blocks
    A = np.full((n_dof, n_dof), inter_corr)

    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        A[s:e, s:e] = intra_corr

    np.fill_diagonal(A, 1.0)

    noise = rng.normal(0.0, 0.05, (n_dof, n_dof))
    noise = (noise + noise.T) / 2.0
    A = np.clip(A + noise, -1.0, 1.0)
    np.fill_diagonal(A, 1.0)
    return A
