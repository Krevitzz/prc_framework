"""
prc.featuring.registries.universal_lite

Responsabilité : Fonctions extraction universelles (tout tenseur) — 3-4 features

Layer : universal (applicable rank 2, 3, N)
"""

import numpy as np


def euclidean_norm(state: np.ndarray) -> float:
    """
    Norme euclidienne (L2) — flatten automatique.
    
    Applicable : tout tenseur (universal)
    
    Returns:
        Norme euclidienne (float)
    """
    return float(np.linalg.norm(state))


def entropy(state: np.ndarray, bins: int = 50) -> float:
    """
    Entropie Shannon distribution valeurs.
    
    Applicable : tout tenseur (universal)
    
    Args:
        state : Tenseur état
        bins  : Nombre bins histogram (défaut 50)
    
    Returns:
        Entropie Shannon (float), toujours ≥ 0
    
    Notes:
        - Flatten automatique
        - Protection log(0) via filtrage hist > 0
        - Normalisation correcte → probabilités (somme = 1)
        - Valeurs typiques : [0, log(bins)]
    """
    flat = state.flatten()
    
    # Histogram brut (pas density=True qui peut donner valeurs > 1)
    hist, _ = np.histogram(flat, bins=bins)
    
    # Normaliser → probabilités
    hist = hist.astype(float)
    hist_sum = hist.sum()
    
    if hist_sum == 0:
        return 0.0
    
    hist = hist / hist_sum
    
    # Filtrer valeurs nulles (éviter log(0))
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    # Entropie Shannon : H = -Σ p_i log(p_i)
    entropy_val = -np.sum(hist * np.log(hist))
    
    return float(entropy_val)


def mean_value(state: np.ndarray) -> float:
    """
    Moyenne valeurs tenseur.
    
    Applicable : tout tenseur (universal)
    """
    return float(np.mean(state))


def std_value(state: np.ndarray) -> float:
    """
    Écart-type valeurs tenseur.
    
    Applicable : tout tenseur (universal)
    """
    return float(np.std(state))
