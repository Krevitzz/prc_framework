"""
D_encodings/sym_007_uniform_correlation.py

SYM-007 : Matrice avec corrélations uniformes
Forme   : C[i,j] = correlation (i≠j), C[i,i] = 1
Rang    : 2
Propriétés : symétrique, corrélation uniforme, diagonale=1
Usage   : Test corrélations égales entre tous DOF
"""

import numpy as np


METADATA = {
    'id': 'SYM-007',
    'd_applicability': ['SYM'],
}


def create(n_dof: int, correlation: float = 0.5) -> np.ndarray:
    """
    Args:
        n_dof       : Nombre de degrés de liberté
        correlation : Valeur uniforme dans [-1, 1] (0.5 par défaut)
    """
    assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"
    C = np.full((n_dof, n_dof), correlation)
    np.fill_diagonal(C, 1.0)
    return C
