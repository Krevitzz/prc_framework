"""
D_encodings/sym_008_random_clipped.py

SYM-008 : Matrice symétrique aléatoire clippée
Forme   : C = (N(mean,std) + N(mean,std)^T)/2, clippé [-1,1], diag=1
Rang    : 2
Propriétés : symétrique, distribution paramétrable, clippée, diagonale=1
Usage   : Test distribution normale paramétrable avec clipping
"""

import numpy as np


METADATA = {
    'id': 'SYM-008',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, seed_CI: int = None,
           mean: float = 0.0, std: float = 0.3) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    C = rng.normal(mean, std, (n_dof, n_dof))
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C
