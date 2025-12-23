"""
modifiers/noise.py

Modificateurs ajoutant du bruit à un état D.

USAGE:
    D = prepare_state(base, [add_gaussian_noise(sigma=0.05)])
"""

import numpy as np
from typing import Callable


def add_gaussian_noise(sigma: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant une fonction qui ajoute du bruit Gaussien.
    
    Args:
        sigma: Amplitude du bruit
        seed: Graine aléatoire (pour reproductibilité)
    
    Returns:
        Fonction (np.ndarray → np.ndarray) qui ajoute bruit
    
    Exemple:
        modifier = add_gaussian_noise(sigma=0.05, seed=42)
        D_noisy = modifier(D_base)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def modifier(state: np.ndarray) -> np.ndarray:
        """Ajoute bruit Gaussien à state."""
        noise = rng.randn(*state.shape) * sigma
        return state + noise
    
    return modifier


def add_uniform_noise(amplitude: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant une fonction qui ajoute du bruit uniforme.
    
    Args:
        amplitude: Amplitude du bruit (dans [-amplitude, +amplitude])
        seed: Graine aléatoire
    
    Returns:
        Fonction qui ajoute bruit uniforme
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def modifier(state: np.ndarray) -> np.ndarray:
        """Ajoute bruit uniforme à state."""
        noise = rng.uniform(-amplitude, amplitude, size=state.shape)
        return state + noise
    
    return modifier