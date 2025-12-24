"""
D_encodings/rank2_symmetric.py

Créateurs d'états D^(base) pour tenseurs rang 2 symétriques.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Diagonale C[i,i] = 1 (optionnelle selon type)
- Bornes [-1, 1]

Ces présuppositions sont HORS du core.
"""

import numpy as np


def create_identity(n_dof: int) -> np.ndarray:
    """
    D-SYM-001: Matrice identité.
    
    INTERPRÉTATION: DOF indépendants (corrélations nulles).
    USAGE: Test stabilité minimale, point fixe trivial.
    PROPRIÉTÉS: symétrique, définie positive, sparse
    
    Args:
        n_dof: Nombre de degrés de liberté
    
    Returns:
        Matrice identité n_dof × n_dof
    """
    return np.eye(n_dof)


def create_uniform(n_dof: int, correlation: float = 0.5) -> np.ndarray:
    """
    Crée matrice avec corrélations uniformes.
    
    INTERPRÉTATION: Tous DOF également corrélés.
    
    Args:
        n_dof: Nombre de degrés de liberté
        correlation: Valeur de corrélation uniforme (dans [-1, 1])
    
    Returns:
        Matrice n_dof × n_dof avec C[i,j] = correlation (i≠j), C[i,i] = 1
    """
    assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"
    
    C = np.full((n_dof, n_dof), correlation)
    np.fill_diagonal(C, 1.0)
    return C


def create_random(n_dof: int, mean: float = 0.0, std: float = 0.3, 
                  seed: int = None) -> np.ndarray:
    """
    Crée matrice avec corrélations aléatoires.
    
    Args:
        n_dof: Nombre de degrés de liberté
        mean: Moyenne des corrélations
        std: Écart-type des corrélations
        seed: Graine aléatoire
    
    Returns:
        Matrice symétrique aléatoire
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice aléatoire
    C = np.random.normal(mean, std, (n_dof, n_dof))
    
    # Symétrise
    C = (C + C.T) / 2
    
    # Diagonale = 1
    np.fill_diagonal(C, 1.0)
    
    # Clip dans [-1, 1]
    C = np.clip(C, -1.0, 1.0)
    
    return C


def create_random_uniform(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-SYM-002: Matrice aléatoire symétrique uniforme.
    
    FORME: A = (B + B^T)/2, B_ij ~ U[-1,1]
    USAGE: Test diversité maximale, générique
    PROPRIÉTÉS: symétrique, bornes [-1,1]
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice symétrique aléatoire uniforme
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice uniforme
    B = np.random.uniform(-1.0, 1.0, (n_dof, n_dof))
    
    # Symétrise
    A = (B + B.T) / 2.0
    
    return A


def create_random_gaussian(n_dof: int, sigma: float = 0.3, 
                          seed: int = None) -> np.ndarray:
    """
    D-SYM-003: Matrice aléatoire symétrique gaussienne.
    
    FORME: A = (B + B^T)/2, B_ij ~ N(0, σ=0.3)
    USAGE: Test continuité, distribution normale
    PROPRIÉTÉS: symétrique, non bornée a priori
    
    Args:
        n_dof: Nombre de degrés de liberté
        sigma: Écart-type de la distribution gaussienne
        seed: Graine aléatoire
    
    Returns:
        Matrice symétrique gaussienne
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice gaussienne
    B = np.random.normal(0.0, sigma, (n_dof, n_dof))
    
    # Symétrise
    A = (B + B.T) / 2.0
    
    return A


def create_correlation_matrix(n_dof: int, seed: int = None) -> np.ndarray:
    """
    D-SYM-004: Matrice de corrélation aléatoire (SPD).
    
    FORME: A = C·C^T normalisée, C_ij ~ N(0,1)
    USAGE: Test positivité définie
    PROPRIÉTÉS: symétrique, définie positive, diag=1
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
    
    Returns:
        Matrice de corrélation (SPD, diagonale=1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Génère matrice gaussienne
    C = np.random.normal(0.0, 1.0, (n_dof, n_dof))
    
    # Produit C·C^T pour garantir positivité
    A = C @ C.T
    
    # Normalise pour avoir diagonale = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A)))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A


def create_banded(n_dof: int, bandwidth: int = 3, 
                  amplitude: float = 0.5, seed: int = None) -> np.ndarray:
    """
    D-SYM-005: Matrice bande symétrique.
    
    FORME: A_ij ≠ 0 ssi |i-j| ≤ bandwidth, valeurs ~ U[-amplitude, amplitude]
    USAGE: Test localité structurelle
    PROPRIÉTÉS: symétrique, sparse, bande
    
    Args:
        n_dof: Nombre de degrés de liberté
        bandwidth: Largeur de bande (3 par défaut)
        amplitude: Amplitude des valeurs hors-diagonale
        seed: Graine aléatoire
    
    Returns:
        Matrice bande symétrique
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialise matrice nulle
    A = np.zeros((n_dof, n_dof))
    
    # Remplit la bande
    for i in range(n_dof):
        for j in range(max(0, i - bandwidth), min(n_dof, i + bandwidth + 1)):
            if i != j:
                value = np.random.uniform(-amplitude, amplitude)
                A[i, j] = value
                A[j, i] = value  # Symétrie
    
    # Diagonale = 1
    np.fill_diagonal(A, 1.0)
    
    return A


def create_block_hierarchical(n_dof: int, n_blocks: int = 10,
                              intra_corr: float = 0.7,
                              inter_corr: float = 0.1,
                              seed: int = None) -> np.ndarray:
    """
    D-SYM-006: Matrice hiérarchique par blocs.
    
    FORME: Blocs denses intra (corrélation forte), sparse inter (corrélation faible)
    USAGE: Test préservation structure modulaire
    PROPRIÉTÉS: symétrique, structure blocs
    
    Args:
        n_dof: Nombre de degrés de liberté (doit être divisible par n_blocks)
        n_blocks: Nombre de blocs
        intra_corr: Corrélation intra-bloc
        inter_corr: Corrélation inter-blocs
        seed: Graine aléatoire
    
    Returns:
        Matrice hiérarchique par blocs
    """
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
    
    if seed is not None:
        np.random.seed(seed)
    
    block_size = n_dof // n_blocks
    
    # Initialise avec corrélation inter-blocs
    A = np.full((n_dof, n_dof), inter_corr)
    
    # Remplit blocs avec corrélation intra-bloc
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        A[start:end, start:end] = intra_corr
    
    # Diagonale = 1
    np.fill_diagonal(A, 1.0)
    
    # Ajoute légère variation aléatoire
    noise = np.random.normal(0.0, 0.05, (n_dof, n_dof))
    noise = (noise + noise.T) / 2  # Symétrise
    A = A + noise
    
    # Clip dans [-1, 1]
    A = np.clip(A, -1.0, 1.0)
    np.fill_diagonal(A, 1.0)  # Restaure diagonale
    
    return A