"""
prc.featuring.layers_lite

Responsabilité : Inspection shape/rank history — routing layer universal

Minimal : universal layer uniquement
"""

import numpy as np
from typing import Dict


def inspect_history(history: np.ndarray) -> Dict[str, any]:
    """
    Inspecte history pour déterminer layer applicable.
    
    Args:
        history : np.ndarray (T, *state_shape)
    
    Returns:
        {
            'rank': int (rank du state, pas du history),
            'shape': tuple (shape du state),
            'is_square': bool (rank 2 carrée),
            'is_cubic': bool (rank 3 cubique),
        }
    
    Examples:
        >>> history = np.random.rand(201, 10, 10)
        >>> info = inspect_history(history)
        >>> info['rank']
        2
        >>> info['is_square']
        True
    """
    # State = history[0]
    state_shape = history.shape[1:]
    rank = len(state_shape)
    
    # Détection square/cubic
    is_square = False
    is_cubic = False
    
    if rank == 2:
        is_square = (state_shape[0] == state_shape[1])
    elif rank == 3:
        is_cubic = (state_shape[0] == state_shape[1] == state_shape[2])
    
    return {
        'rank': rank,
        'shape': state_shape,
        'is_square': is_square,
        'is_cubic': is_cubic,
    }
