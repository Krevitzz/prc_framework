"""
D_encodings/r3_001_random_uniform.py

R3-001 : Tenseur rang 3 aléatoire uniforme
Forme   : T_ijk ~ U[-1,1]
Rang    : 3
Propriétés : aucune symétrie, bornes [-1,1]
Usage   : Test générique rang 3
"""

import numpy as np


METADATA = {
    'id': 'R3-001',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    return rng.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
