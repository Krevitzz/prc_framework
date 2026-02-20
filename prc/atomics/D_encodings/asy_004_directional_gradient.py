"""
D_encodings/asy_004_directional_gradient.py

ASY-004 : Matrice avec gradient directionnel
Forme   : A_ij = gradient·(i-j) + U[-noise_amplitude, +noise_amplitude]
Rang    : 2
Propriétés : asymétrique, gradient linéaire
Usage   : Test brisure symétrie avec structure
"""

import numpy as np


METADATA = {
    'id': 'ASY-004',
    'd_applicability': ['ASY'],
}


def create(n_dof: int, seed_CI: int = None,
           gradient: float = 0.1, noise_amplitude: float = 0.2) -> np.ndarray:
    """
    Args:
        n_dof           : Nombre de degrés de liberté
        seed_CI         : Paramètre reproductibilité (depuis YAML defaults)
        gradient        : Pente du gradient (0.1 par défaut)
        noise_amplitude : Amplitude du bruit additif uniforme (0.2 par défaut)
    """
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random
    i_idx, j_idx = np.meshgrid(range(n_dof), range(n_dof), indexing='ij')
    A = gradient * (i_idx - j_idx).astype(float)
    A += rng.uniform(-noise_amplitude, noise_amplitude, (n_dof, n_dof))
    return A
