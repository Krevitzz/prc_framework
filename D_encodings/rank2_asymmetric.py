"""
D_encodings/rank2_asymmetric.py

Créateurs d'états D^(base) pour tenseurs rang 2 asymétriques.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Asymétrie C[i,j] ≠ C[j,i] (général)
- Pas de contrainte diagonale (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)

Ces présuppositions sont HORS du core.
"""

import numpy as np


def create_random_asymmetric(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-ASY-001: Matrice asymétrique aléatoire uniforme.
    
    FORME: A_ij ~ U[-1,1] indépendants
    USAGE: Test asymétrie générique
    PROPRIÉTÉS: asymétrique, bornes [-1,1]
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice asymétrique aléatoire
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(-1.0, 1.0, (n_dof, n_dof))


def create_lower_triangular(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-ASY-002: Matrice triangulaire inférieure.
    
    FORME: A_ij = U[-1,1] si i > j, sinon 0
    USAGE: Test orientation directionnelle
    PROPRIÉTÉS: asymétrique, sparse, triangulaire
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice triangulaire inférieure
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice aléatoire
    A = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Garde seulement triangulaire inférieure (strict)
    A = np.tril(A, k=-1)
    
    return A


def create_antisymmetric(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-ASY-003: Matrice antisymétrique.
    
    FORME: A = -A^T, A_ij ~ U[-1,1] pour i>j
    USAGE: Test conservation antisymétrie
    PROPRIÉTÉS: antisymétrique (cas spécial asymétrique), diagonale nulle
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice antisymétrique
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère partie triangulaire inférieure
    B = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    B = np.tril(B, k=-1)
    
    # Antisymétrise: A = B - B^T
    A = B - B.T
    
    return A


def create_directional_gradient(n_dof: int, gradient: float = 0.1,
                                noise_amplitude: float = 0.2,
                                seed: int = None) -> np.ndarray:
    """
    D-ASY-004: Matrice avec gradient directionnel.
    
    FORME: A_ij = gradient·(i-j) + U[-noise, +noise]
    USAGE: Test brisure symétrie avec structure
    PROPRIÉTÉS: asymétrique, gradient linéaire
    
    Args:
        n_dof: Nombre de degrés de liberté
        gradient: Pente du gradient (0.1 par défaut)
        noise_amplitude: Amplitude du bruit additif
        seed: Graine aléatoire
    
    Returns:
        Matrice avec gradient directionnel
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Crée grille d'indices
    i_indices, j_indices = np.meshgrid(range(n_dof), range(n_dof), indexing='ij')
    
    # Calcule gradient
    A = gradient * (i_indices - j_indices)
    
    # Ajoute bruit
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, (n_dof, n_dof))
    A = A + noise
    
    return A


def create_circulant_asymmetric(n_dof: int, seed: int = None) -> np.ndarray:
    """
    Matrice circulante asymétrique (bonus).
    
    FORME: Chaque ligne est la précédente décalée d'un cran
    USAGE: Test structure périodique asymétrique
    PROPRIÉTÉS: asymétrique, structure circulante
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice circulante asymétrique
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère première ligne
    first_row = np.random.uniform(-1.0, 1.0, n_dof)
    
    # Construit matrice circulante
    A = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        A[i] = np.roll(first_row, i)
    
    return A


def create_sparse_asymmetric(n_dof: int, density: float = 0.2,
                             seed: int = None) -> np.ndarray:
    """
    Matrice asymétrique sparse (bonus).
    
    FORME: density% des éléments non-nuls, asymétrique
    USAGE: Test structures creuses asymétriques
    PROPRIÉTÉS: asymétrique, sparse
    
    Args:
        n_dof: Nombre de degrés de liberté
        density: Densité de valeurs non-nulles (0.2 = 20%)
        seed: Graine aléatoire
    
    Returns:
        Matrice asymétrique sparse
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice complète
    A = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Masque pour sparsité
    mask = np.random.random((n_dof, n_dof)) < density
    A = A * mask
    
    return A