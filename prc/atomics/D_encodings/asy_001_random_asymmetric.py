"""
D_encodings/asy_001_random_asymmetric.py

ASY-001 : Matrice asymétrique aléatoire uniforme
Forme   : A_ij ~ U[-1,1] indépendants
Rang    : 2
Propriétés : asymétrique, bornes [-1,1]
Usage   : Test asymétrie générique
"""

import numpy as np


METADATA = {
    'id': 'ASY-001',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    return rng.uniform(-1.0, 1.0, (n_dof, n_dof))
