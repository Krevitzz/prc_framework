"""
D_encodings/r3_004_fully_symmetric.py

R3-004 : Tenseur totalement symétrique (6 permutations)
Forme   : T_ijk = T_jik = T_ikj = T_jki = T_kij = T_kji
Rang    : 3
Propriétés : invariant par toutes permutations d'indices
Usage   : Test symétrie complète
"""

import numpy as np


METADATA = {
    'id': 'R3-004',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    T = rng.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    return (
        T +
        np.transpose(T, (0, 2, 1)) +
        np.transpose(T, (1, 0, 2)) +
        np.transpose(T, (1, 2, 0)) +
        np.transpose(T, (2, 0, 1)) +
        np.transpose(T, (2, 1, 0))
    ) / 6.0
