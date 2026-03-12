"""
D_encodings/r3_002_partial_symmetric.py

R3-002 : Tenseur symétrique partiel (indices j,k)
Forme   : T_ijk = T_ikj, valeurs ~ U[-1,1]
Rang    : 3
Propriétés : symétrie partielle sur 2 indices
Usage   : Test symétries partielles
"""

import numpy as np


METADATA = {
    'id': 'R3-002',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    T = rng.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    return (T + np.transpose(T, (0, 2, 1))) / 2.0
