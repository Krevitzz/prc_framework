"""
prc.featuring.registries.matrix_square_lite

Responsabilité : Features matrices carrées (rank 2, is_square)

Applicabilité : matrix_square layer
    - rank == 2
    - is_square == True

Fonctions : trace, eigenvalue_max, asymmetry_norm, condition_number, determinant
"""

import numpy as np


def trace(state: np.ndarray) -> float:
    """
    Trace de la matrice : Σ diag(A).

    Returns:
        float — somme diagonale
    """
    return float(np.trace(state))


def eigenvalue_max(state: np.ndarray) -> float:
    """
    Valeur propre maximale (module) : max|λ_i|.

    Returns:
        float — |λ_max|

    Notes:
        - Protection LinAlgError → retourne np.nan (signal physique)
    """
    try:
        eigenvalues = np.linalg.eigvals(state)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        return np.nan


def asymmetry_norm(state: np.ndarray) -> float:
    """
    Norme asymétrie : ||A - A^T||_F.

    Returns:
        float — norme Frobenius partie antisymétrique
        0.0 si symétrique parfait
    """
    asymmetric_part = state - state.T
    return float(np.linalg.norm(asymmetric_part, 'fro'))


def condition_number(state: np.ndarray) -> float:
    """
    Condition number : κ(A) = σ_max / σ_min.

    Returns:
        float — condition number
        np.inf si singulière (σ_min = 0) → signal physique

    Notes:
        - κ grand → matrice mal conditionnée
        - np.inf = information (pas erreur)
    """
    try:
        return float(np.linalg.cond(state))
    except np.linalg.LinAlgError:
        return np.inf


def determinant(state: np.ndarray) -> float:
    """
    Déterminant : det(A).

    Returns:
        float — déterminant, clippé [-1e15, 1e15] (protection overflow)
    """
    try:
        det = np.linalg.det(state)
        return float(np.clip(det, -1e15, 1e15))
    except np.linalg.LinAlgError:
        return np.nan

def symmetry_deviation(state: np.ndarray) -> float:
    """
    Déviation relative à la symétrie parfaite.

    Mesure à quel point la matrice s'écarte de la symétrie exacte,
    normalisée par la norme Frobenius.

    Returns:
        float — ||A - A^T||_F / (||A||_F + ε)
        0.0 si symétrique parfait
        >0 si déviation numérique présente

    Notes:
        - Complémentaire de asymmetry_norm (normalisé ici)
        - Utile pour détecter bris de symétrie progressif
    """
    norm_asym = np.linalg.norm(state - state.T, 'fro')
    norm_state = np.linalg.norm(state, 'fro')
    return float(norm_asym / (norm_state + 1e-10))