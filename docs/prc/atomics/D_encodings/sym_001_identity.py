"""
D_encodings/sym_001_identity.py

SYM-001 : Matrice identité
Forme   : I[i,j] = δ_ij
Rang    : 2
Propriétés : symétrique, définie positive, sparse
Usage   : Test stabilité minimale, point fixe trivial
"""

import numpy as np


METADATA = {
    'id': 'SYM-001',
    'd_applicability': ['SYM'],
}


def create(n_dof: int) -> np.ndarray:
    """Déterministe — pas de seed_CI."""
    return np.eye(n_dof)
