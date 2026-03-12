"""
D_encodings/sym_003_random_gaussian.py

SYM-003 : Matrice symétrique aléatoire gaussienne
Forme   : A = (B + B^T)/2, B_ij ~ N(0, σ)
Rang    : 2
Propriétés : symétrique, non bornée a priori
Usage   : Test continuité, distribution normale
"""

import numpy as np


METADATA = {
    'id': 'SYM-003',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None, sigma: float = 0.3) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    B = rng.normal(0.0, sigma, (n_dof, n_dof))
    return (B + B.T) / 2.0
