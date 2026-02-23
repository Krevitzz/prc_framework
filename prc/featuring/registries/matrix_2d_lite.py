"""
prc.featuring.registries.matrix_2d_lite

Responsabilité : Features rank 2 (matrices)

Applicabilité : SYM, ASY (rank 2 uniquement)
"""

import numpy as np


def trace(state: np.ndarray) -> float:
    """
    Trace de la matrice.
    
    Args:
        state : Matrice (n, n)
    
    Returns:
        Trace (somme diagonale)
    
    Raises:
        ValueError : Si non carrée
    """
    if state.ndim != 2:
        raise ValueError(f"trace requiert rank 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"trace requiert matrice carrée, reçu {state.shape}")
    
    return float(np.trace(state))


def eigenvalue_max(state: np.ndarray) -> float:
    """
    Valeur propre maximale (module).
    
    Args:
        state : Matrice (n, n)
    
    Returns:
        |λ_max|
    
    Notes:
        - Protection matrices singulières (retourne 0.0)
    """
    if state.ndim != 2:
        raise ValueError(f"eigenvalue_max requiert rank 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"eigenvalue_max requiert matrice carrée, reçu {state.shape}")
    
    try:
        eigenvalues = np.linalg.eigvals(state)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        # Matrice singulière ou numériquement instable
        return 0.0


def asymmetry_norm(state: np.ndarray) -> float:
    """
    Norme asymétrie ||A - A^T||_F.
    
    Args:
        state : Matrice (n, n)
    
    Returns:
        Norme Frobenius de la partie asymétrique
    
    Notes:
        - 0.0 si symétrique parfait
        - >0 si asymétrique
    """
    if state.ndim != 2:
        raise ValueError(f"asymmetry_norm requiert rank 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"asymmetry_norm requiert matrice carrée, reçu {state.shape}")
    
    asymmetric_part = state - state.T
    return float(np.linalg.norm(asymmetric_part, 'fro'))


def condition_number(state: np.ndarray) -> float:
    """
    Condition number κ(A) = σ_max / σ_min.
    
    Args:
        state : Matrice (n, n)
    
    Returns:
        Condition number
    
    Notes:
        - Protection singularité : retourne np.inf si σ_min = 0
        - κ(A) grand → matrice mal conditionnée
    """
    if state.ndim != 2:
        raise ValueError(f"condition_number requiert rank 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"condition_number requiert matrice carrée, reçu {state.shape}")
    
    try:
        return float(np.linalg.cond(state))
    except np.linalg.LinAlgError:
        return np.inf


def determinant(state: np.ndarray) -> float:
    """
    Déterminant det(A).
    
    Args:
        state : Matrice (n, n)
    
    Returns:
        Déterminant
    
    Notes:
        - Protection overflow : clip [-1e15, 1e15]
    """
    if state.ndim != 2:
        raise ValueError(f"determinant requiert rank 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(f"determinant requiert matrice carrée, reçu {state.shape}")
    
    try:
        det = np.linalg.det(state)
        # Clip pour éviter overflow
        return float(np.clip(det, -1e15, 1e15))
    except np.linalg.LinAlgError:
        return 0.0
