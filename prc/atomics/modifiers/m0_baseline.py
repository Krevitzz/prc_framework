"""
modifiers/m0_baseline.py

M0 : Modifier baseline — identité (aucune transformation)
Forme : D' = D
Propriétés : déterministe, identité
Usage : Référence pure pour comparaison
"""

import numpy as np


METADATA = {
    'id': 'M0',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}


def create(state: np.ndarray, seed_run: int = None) -> np.ndarray:
    """
    Transformation identité. seed_run ignoré (déterministe).

    Args:
        state    : Tenseur d'état
        seed_run : Ignoré

    Returns:
        Copie du tenseur inchangé
    """
    return state.copy()
