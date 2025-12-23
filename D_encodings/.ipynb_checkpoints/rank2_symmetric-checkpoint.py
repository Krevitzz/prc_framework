"""
encodings/rank2_symmetric.py

Créateurs d'états D^(base) pour tenseurs rang 2 symétriques.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Diagonale C[i,i] = 1
- Bornes [-1, 1]

Ces présuppositions sont HORS du core.
"""

import numpy as np


def create_identity(n_dof: int) -> np.ndarray:
    """
    Crée matrice identité.
    
    INTERPRÉTATION: DOF indépendants (corrélations nulles).
    
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