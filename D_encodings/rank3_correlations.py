"""
D_encodings/rank3_correlations.py

Créateurs d'états D^(base) pour tenseurs rang 3.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 3 (tenseur N×N×N)
- Aucune symétrie par défaut (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)

Ces présuppositions sont HORS du core.

NOTE: Les tenseurs rang 3 sont plus coûteux en mémoire.
      Pour N=20: 20³ = 8000 éléments
      Pour N=30: 30³ = 27000 éléments
"""

import numpy as np


def create_random_rank3(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-R3-001: Tenseur aléatoire uniforme.
    
    FORME: T_ijk ~ U[-1,1]
    USAGE: Test générique rang 3
    PROPRIÉTÉS: aucune symétrie
    
    Args:
        n_dof: Dimension du tenseur (N×N×N)
        seed: Graine aléatoire
    
    Returns:
        Tenseur rang 3 aléatoire (N, N, N)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))


def create_partial_symmetric_rank3(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-R3-002: Tenseur symétrique partiel (indices 2-3).
    
    FORME: T_ijk = T_ikj, valeurs ~ U[-1,1]
    USAGE: Test symétries partielles
    PROPRIÉTÉS: symétrie sur 2 indices
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur avec T[i,j,k] = T[i,k,j]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère tenseur aléatoire
    T = np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    
    # Symétrise indices 1 et 2 (j et k)
    T_sym = (T + np.transpose(T, (0, 2, 1))) / 2.0
    
    return T_sym


def create_fully_symmetric_rank3(n_dof: int, seed: int = None) -> np.ndarray:
    """
    Tenseur totalement symétrique (bonus).
    
    FORME: T_ijk = T_jik = T_ikj = T_jki = T_kij = T_kji
    USAGE: Test symétrie complète
    PROPRIÉTÉS: invariant par permutation indices
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur totalement symétrique
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère tenseur aléatoire
    T = np.random.uniform(-1.0, 1.0, (n_dof, n_dof, n_dof))
    
    # Moyenne sur toutes les permutations
    T_sym = (T + 
             np.transpose(T, (0, 2, 1)) +
             np.transpose(T, (1, 0, 2)) +
             np.transpose(T, (1, 2, 0)) +
             np.transpose(T, (2, 0, 1)) +
             np.transpose(T, (2, 1, 0))) / 6.0
    
    return T_sym


def create_local_coupling_rank3(n_dof: int, radius: int = 2,
                               seed: int = None) -> np.ndarray:
    """
    D-R3-003: Tenseur avec couplages locaux.
    
    FORME: T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2*radius
    USAGE: Test localité 3-corps
    PROPRIÉTÉS: sparse, localité géométrique
    
    Args:
        n_dof: Dimension du tenseur
        radius: Rayon de localité (5 par défaut → 2*radius=10)
        seed: Graine aléatoire
    
    Returns:
        Tenseur sparse avec couplages locaux
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialise tenseur nul
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit selon contrainte de localité
    for i in range(n_dof):
        for j in range(n_dof):
            for k in range(n_dof):
                distance = abs(i - j) + abs(j - k) + abs(k - i)
                if distance <= 2 * radius:
                    T[i, j, k] = np.random.uniform(-1.0, 1.0)
    
    return T


def create_diagonal_rank3(n_dof: int, seed: int = None) -> np.ndarray:
    """
    Tenseur diagonal (bonus).
    
    FORME: T_ijk ≠ 0 ssi i=j=k
    USAGE: Test structure diagonale rang 3
    PROPRIÉTÉS: très sparse, diagonal
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur diagonal
    """
    if seed is not None:
        np.random.seed(seed)
    
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit seulement diagonale principale
    for i in range(n_dof):
        T[i, i, i] = np.random.uniform(-1.0, 1.0)
    
    return T


def create_separable_rank3(n_dof: int, seed: int = None) -> np.ndarray:
    """
    Tenseur séparable (bonus).
    
    FORME: T_ijk = u_i · v_j · w_k (produit externe)
    USAGE: Test structure factorisée
    PROPRIÉTÉS: rang tensoriel = 1
    
    Args:
        n_dof: Dimension du tenseur
        seed: Graine aléatoire
    
    Returns:
        Tenseur séparable
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère 3 vecteurs aléatoires
    u = np.random.uniform(-1.0, 1.0, n_dof)
    v = np.random.uniform(-1.0, 1.0, n_dof)
    w = np.random.uniform(-1.0, 1.0, n_dof)
    
    # Produit externe
    T = np.outer(u, np.outer(v, w)).reshape(n_dof, n_dof, n_dof)
    
    return T


def create_block_rank3(n_dof: int, n_blocks: int = 4,
                      seed: int = None) -> np.ndarray:
    """
    Tenseur par blocs (bonus).
    
    FORME: Structure par blocs dans les 3 dimensions
    USAGE: Test modularité rang 3
    PROPRIÉTÉS: structure hiérarchique
    
    Args:
        n_dof: Dimension du tenseur (doit être divisible par n_blocks)
        n_blocks: Nombre de blocs par dimension
        seed: Graine aléatoire
    
    Returns:
        Tenseur par blocs
    """
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
    
    if seed is not None:
        np.random.seed(seed)
    
    block_size = n_dof // n_blocks
    T = np.zeros((n_dof, n_dof, n_dof))
    
    # Remplit blocs diagonaux (i_block = j_block = k_block)
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        T[start:end, start:end, start:end] = np.random.uniform(
            -1.0, 1.0, (block_size, block_size, block_size)
        )
    
    # Ajoute faible couplage inter-blocs
    T += np.random.uniform(-0.1, 0.1, (n_dof, n_dof, n_dof))
    
    return T