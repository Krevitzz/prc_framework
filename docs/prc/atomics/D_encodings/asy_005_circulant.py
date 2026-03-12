"""
D_encodings/asy_005_circulant.py

ASY-005 : Matrice circulante asymétrique
Forme   : Chaque ligne = précédente décalée d'un cran
Rang    : 2
Propriétés : asymétrique, structure circulante périodique
Usage   : Test structure périodique asymétrique
"""

import numpy as np


METADATA = {
    'id': 'ASY-005',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    first_row = rng.uniform(-1.0, 1.0, n_dof)
    # Vectorisé : décalages successifs via indexation circulaire
    idx = (np.arange(n_dof)[:, None] + np.arange(n_dof)[None, :]) % n_dof
    return first_row[idx]
