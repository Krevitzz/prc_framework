"""
modifiers/m2_uniform_noise.py

M2 : Bruit uniforme additif
Forme : D' = D + U[-amplitude, +amplitude]
Propriétés : stochastique, borné, distribution uniforme
Usage : Perturbation uniforme des conditions initiales
"""

import numpy as np


METADATA = {
    'id': 'M2',
    'd_applicability': ['SYM', 'ASY', 'R3'],
}


def create(state: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
    """
    Args:
        state     : Tenseur d'état
        amplitude : Amplitude du bruit (0.1 par défaut)
    """
    noise = np.random.uniform(-amplitude, amplitude, size=state.shape)
    return state + noise
