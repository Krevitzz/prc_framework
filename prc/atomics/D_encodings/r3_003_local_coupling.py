"""
D_encodings/r3_003_local_coupling.py

R3-003 : Tenseur avec couplages locaux
Forme   : T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2·radius
Rang    : 3
Propriétés : sparse, localité géométrique
Usage   : Test localité 3-corps

Optimisation : triple boucle → meshgrid + masque vectorisé
"""

import numpy as np


METADATA = {
    'id': 'R3-003',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None, radius: int = 2) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random

    # Vectorisé : grille 3D des indices
    idx = np.arange(n_dof)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    distance = np.abs(i - j) + np.abs(j - k) + np.abs(k - i)
    mask = distance <= 2 * radius

    T = np.zeros((n_dof, n_dof, n_dof))
    n_nonzero = mask.sum()
    T[mask] = rng.uniform(-1.0, 1.0, n_nonzero)
    return T
