"""
prc.featuring.registries.tensor_3d_lite

Responsabilité : Features rank 3 (tenseurs)

Applicabilité : R3 (rank 3 uniquement)
"""

import numpy as np


def mode_variance_0(state: np.ndarray) -> float:
    """
    Variance le long du mode 0 (axe i).
    
    Args:
        state : Tenseur (n, m, p)
    
    Returns:
        Variance moyenne des fibres mode-0
    
    Notes:
        - Fiber mode-0 : T[i, :, :]
        - Mesure hétérogénéité selon dimension 0
    """
    if state.ndim != 3:
        raise ValueError(f"mode_variance_0 requiert rank 3, reçu {state.ndim}")
    
    # Variance le long de l'axe 0
    variances = np.var(state, axis=0)
    return float(np.mean(variances))


def mode_variance_1(state: np.ndarray) -> float:
    """
    Variance le long du mode 1 (axe j).
    
    Args:
        state : Tenseur (n, m, p)
    
    Returns:
        Variance moyenne des fibres mode-1
    
    Notes:
        - Fiber mode-1 : T[:, j, :]
        - Mesure hétérogénéité selon dimension 1
    """
    if state.ndim != 3:
        raise ValueError(f"mode_variance_1 requiert rank 3, reçu {state.ndim}")
    
    # Variance le long de l'axe 1
    variances = np.var(state, axis=1)
    return float(np.mean(variances))


def mode_variance_2(state: np.ndarray) -> float:
    """
    Variance le long du mode 2 (axe k).
    
    Args:
        state : Tenseur (n, m, p)
    
    Returns:
        Variance moyenne des fibres mode-2
    
    Notes:
        - Fiber mode-2 : T[:, :, k]
        - Mesure hétérogénéité selon dimension 2
    """
    if state.ndim != 3:
        raise ValueError(f"mode_variance_2 requiert rank 3, reçu {state.ndim}")
    
    # Variance le long de l'axe 2
    variances = np.var(state, axis=2)
    return float(np.mean(variances))
