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


def create(n_dof: int, seed_CI: int = None,
           correlation: float = 0.5) -> np.ndarray:
    """
    Crée matrice avec corrélations uniformes.

    Args:
        n_dof       : Nombre de degrés de liberté
        seed_CI     : Ignoré (encoding déterministe)
        correlation : Valeur uniforme dans [-1, 1] (0.5 par défaut)

    Returns:
        Matrice corrélations uniformes (n_dof, n_dof)
    """
    assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"

    C = np.full((n_dof, n_dof), correlation)
    np.fill_diagonal(C, 1.0)
    return C
