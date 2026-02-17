"""
D_encodings/r3_006_separable.py

R3-006 : Tenseur séparable (produit externe, rang tensoriel 1)
Forme  : T_ijk = u_i · v_j · w_k
Rang   : 3
Propriétés : séparable, factorisé, rang tensoriel=1
Usage  : Test structure factorisée
"""

import numpy as np


METADATA = {
    'id': 'R3-006',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    """
    Crée tenseur séparable.

    Args:
        n_dof   : Dimension du tenseur
        seed_CI : Graine aléatoire (conditions initiales)

    Returns:
        Tenseur séparable (n_dof, n_dof, n_dof)
    """
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random

    u = rng.uniform(-1.0, 1.0, n_dof)
    v = rng.uniform(-1.0, 1.0, n_dof)
    w = rng.uniform(-1.0, 1.0, n_dof)

    return np.outer(u, np.outer(v, w)).reshape(n_dof, n_dof, n_dof)
