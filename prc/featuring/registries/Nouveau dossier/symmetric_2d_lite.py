"""
prc.featuring.registries.symmetric_2d_lite

Responsabilité : Features matrices symétriques (rank 2, is_square, is_symmetric)

Applicabilité : symmetric_2d layer
    - rank == 2
    - is_square == True
    - is_symmetric == True (vérifié sur état final, tolérance numérique)

Fonctions : symmetry_deviation, eigenvalue_spread

Notes :
    - Les matrices symétriques ont des valeurs propres réelles garanties
    - eigenvalue_spread exploite cette propriété (np.linalg.eigh plus stable)
"""

import numpy as np





def eigenvalue_spread(state: np.ndarray) -> float:
    """
    Écart entre valeur propre max et min (spectre réel garanti).

    Pour matrices symétriques : λ_max - λ_min (valeurs propres réelles).

    Returns:
        float — λ_max - λ_min
        np.nan si échec numérique

    Notes:
        - Utilise np.linalg.eigh (plus stable pour matrices symétriques)
        - Mesure l'étendue spectrale → liée à la condition number
    """
    try:
        eigenvalues = np.linalg.eigh(state)[0]  # Valeurs propres réelles triées
        return float(eigenvalues[-1] - eigenvalues[0])
    except np.linalg.LinAlgError:
        return np.nan
