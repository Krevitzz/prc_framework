"""
modifiers/m0_baseline.py

Modifier baseline - identité (aucune transformation).

PRÉSUPPOSITIONS EXPLICITES:
- Transformation identité D' = D
- Aucune modification
- Référence pure pour comparaison
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============
PHASE = "R0"

METADATA = {
    'id': 'M0',
    'type': 'baseline',
    'description': 'Modifier baseline - identité (aucune transformation)',
    'properties': ['deterministic', 'identity'],
    'usage': 'Référence pure, aucune modification'
}

# ============ FONCTION PRINCIPALE ============
def apply(state: np.ndarray) -> np.ndarray:
    """
    Applique transformation identité (aucune modification).
    
    FORME: D' = D
    USAGE: Référence pure pour comparaison
    PROPRIÉTÉS: Déterministe, identité
    
    Args:
        state: Tenseur d'état
        seed: Graine aléatoire (ignoré, déterministe)
    
    Returns:
        Tenseur inchangé
    
    Examples:
        >>> D = np.array([[1, 2], [3, 4]])
        >>> D_modified = apply(D)
        >>> np.allclose(D, D_modified)
        True
    """
    return state.copy()