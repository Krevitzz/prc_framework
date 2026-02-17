"""
D_encodings/asy_004_directional_gradient.py

Matrice avec gradient directionnel.

PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Structure gradient linéaire A_ij ~ (i-j)
- Bruit additif pour variation
- Asymétrique par construction
"""

import numpy as np

# ============ METADATA (OBLIGATOIRE) ============


METADATA = {
    'id': 'ASY-004',
    'rank': 2,
    'type': 'asymmetric',
    'description': 'Matrice avec gradient directionnel',
    'properties': ['asymmetric', 'linear_gradient'],
    'usage': 'Test brisure symétrie avec structure'
}

# ============ FONCTION PRINCIPALE ============
def create(n_dof: int, gradient: float = 0.1, 
           noise_amplitude: float = 0.2) -> np.ndarray:
    """
    Crée matrice avec gradient directionnel.
    
    FORME: A_ij = gradient·(i-j) + U[-noise, +noise]
    USAGE: Test brisure symétrie avec structure
    PROPRIÉTÉS: asymétrique, gradient linéaire
    
    Args:
        n_dof: Nombre de degrés de liberté
        seed: Graine aléatoire
        gradient: Pente du gradient (0.1 par défaut)
        noise_amplitude: Amplitude du bruit additif (0.2 par défaut)
    
    Returns:
        Matrice avec gradient directionnel (n_dof, n_dof)
    
    Examples:
        >>> A = create(3, seed=42, gradient=0.5, noise_amplitude=0.0)
        >>> A.shape
        (3, 3)
        >>> A[2, 0] > A[0, 2]  # Gradient visible
        True
    """
    
    # Crée grille d'indices
    i_indices, j_indices = np.meshgrid(range(n_dof), range(n_dof), indexing='ij')
    
    # Calcule gradient
    A = gradient * (i_indices - j_indices)
    
    # Ajoute bruit
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, (n_dof, n_dof))
    A = A + noise
    
    return A