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


def create(state: np.ndarray) -> np.ndarray:
    """Transformation identité — retourne copie inchangée."""
    return state.copy()
