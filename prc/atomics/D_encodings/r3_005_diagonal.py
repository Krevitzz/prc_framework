"""
D_encodings/r3_005_diagonal.py

R3-005 : Tenseur diagonal (T_ijk ≠ 0 ssi i=j=k)
Forme   : T_ijk ≠ 0 ssi i=j=k, valeurs ~ U[-1,1]
Rang    : 3
Propriétés : diagonal, très sparse, déterministe
Usage   : Test structure diagonale rang 3

Optimisation : boucle for → indexation directe vectorisée
"""

import numpy as np


METADATA = {
    'id': 'R3-005',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    T = np.zeros((n_dof, n_dof, n_dof))
    idx = np.arange(n_dof)
    T[idx, idx, idx] = rng.uniform(-1.0, 1.0, n_dof)
    return T
