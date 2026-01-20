"""
core/kernel.py

Moteur d'exécution PRC universel.

RESPONSABILITÉ UNIQUE:
- Appliquer itérativement state_{n+1} = gamma(state_n)
- Générer les états successifs
- AUCUNE connaissance de gamma, du state, ou de leur interprétation

USAGE:
    for iteration, state in run_kernel(D_initial, gamma, max_iterations=1000):
        if detect_explosion(state):
            break
"""

import numpy as np
from typing import Callable, Optional, Tuple, Generator, Union, List


def run_kernel(
    initial_state: np.ndarray,
    gamma: Callable[[np.ndarray], np.ndarray],
    max_iterations: int = 10000,
    convergence_check: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    record_history: bool = False
) -> Generator[Union[Tuple[int, np.ndarray], Tuple[int, np.ndarray, List[np.ndarray]]], None, None]:
    """
    Générateur d'états successifs state_{n+1} = gamma(state_n).
    
    FONCTION AVEUGLE:
    - Ne connaît ni gamma (juste l'applique)
    - Ne connaît ni le state (juste le propage)
    - Ne connaît ni leur dimension, structure, ou interprétation
    
    Le kernel ne décide JAMAIS d'arrêter par lui-même.
    L'arrêt vient soit de :
    - max_iterations atteint
    - convergence_check retourne True
    - Le TM break la boucle
    
    Args:
        initial_state: Tenseur initial (np.ndarray de shape quelconque)
        gamma: Fonction (np.ndarray → np.ndarray)
        max_iterations: Limite de sécurité
        convergence_check: Fonction (state_n, state_{n+1}) → bool
                           Si retourne True, arrête l'itération
                           Si None, ne vérifie jamais
        record_history: Si True, stocke et yield l'historique complet
                        Si False, ne yield que (iteration, state)
    
    Yields:
        Si record_history=False : (iteration, state)
        Si record_history=True  : (iteration, state, history)
        
        où:
        - iteration : int (0, 1, 2, ...)
        - state : np.ndarray (état à l'iteration n)
        - history : List[np.ndarray] (tous les états jusqu'à n)
    
    Exemples:
        # Sans historique (économie mémoire)
        for i, state in run_kernel(D, gamma, max_iterations=1000):
            if i % 100 == 0:
                print(f"Iteration {i}")
            if detect_explosion(state):
                break
        
        # Avec historique complet
        for i, state, history in run_kernel(D, gamma, max_iterations=1000, 
                                            record_history=True):
            pass
        
        # Avec convergence automatique
        def check_conv(s_n, s_next):
            return np.linalg.norm(s_next - s_n) < 1e-6
        
        for i, state in run_kernel(D, gamma, convergence_check=check_conv):
            pass
    
    Notes:
        - Le TM contrôle entièrement quand arrêter (via break)
        - convergence_check est optionnel (défini dans le TM)
        - Si record_history=True, mémoire O(n × size(state))
    """
    state = initial_state.copy()
    history = [] if record_history else None
    
    for iteration in range(max_iterations):
        # Enregistre état actuel (si demandé)
        if record_history:
            history.append(state.copy())
        
        # Yield état actuel
        if record_history:
            yield iteration, state, history
        else:
            yield iteration, state
        
        # Calcule état suivant
        state_next = gamma(state)
        
        # Vérifie convergence (si check fourni)
        if convergence_check is not None:
            if convergence_check(state, state_next):
                # Yield état final avant de sortir
                if record_history:
                    history.append(state_next.copy())
                    yield iteration + 1, state_next, history
                else:
                    yield iteration + 1, state_next
                return
        
        # Met à jour état
        state = state_next
    
    # Si max_iterations atteint, yield dernier état
    if record_history:
        yield max_iterations, state, history
    else:
        yield max_iterations, state
        
"""
core/state_preparation.py

Composition aveugle d'états D à partir de sources multiples.

RESPONSABILITÉ UNIQUE:
- Appliquer séquentiellement des modificateurs sur un tenseur de base
- AUCUNE connaissance du contenu, dimension, ou interprétation

USAGE:
    D_base = create_base_state(50)
    D_final = prepare_state(D_base, [
        add_noise(sigma=0.05),
        apply_constraint(params)
    ])
"""

import numpy as np
from typing import List, Callable, Optional


def prepare_state(base: np.ndarray,
                  modifiers: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None) -> np.ndarray:
    """
    Compose un état D par application séquentielle de modificateurs.
    
    FONCTION AVEUGLE:
    - Ne connaît ni la dimension du tenseur
    - Ne connaît ni sa structure (symétrie, bornes, etc.)
    - Ne connaît ni son interprétation (corrélations, champ, etc.)
    
    Applique simplement: state → modifier_1 → modifier_2 → ... → state_final
    
    Args:
        base: Tenseur de base (np.ndarray de shape quelconque)
        modifiers: Liste de fonctions (np.ndarray → np.ndarray)
                   Chaque fonction transforme le tenseur
                   Si None ou [], retourne base inchangé
    
    Returns:
        Tenseur composé final (np.ndarray)
    
    Exemples:
        # Sans modificateur
        D = prepare_state(base_state)
        
        # Avec modificateurs
        D = prepare_state(base_state, [
            add_gaussian_noise(sigma=0.05),
            apply_periodic_constraint()
        ])
    
    Notes:
        - Chaque modifier reçoit le résultat du précédent
        - Les modifiers sont définis HORS du core (dans modifiers/)
        - Le core ne valide RIEN sur le contenu
    """
    state = base.copy()
    
    if modifiers is not None and len(modifiers) > 0:
        for modifier in modifiers:
            state = modifier(state)
    
    return state
    
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

"""
modifiers/noise.py

Modificateurs ajoutant du bruit à un état D.

USAGE:
    D = prepare_state(base, [add_gaussian_noise(sigma=0.05)])
"""

import numpy as np
from typing import Callable


def add_gaussian_noise(sigma: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant une fonction qui ajoute du bruit Gaussien.
    
    Args:
        sigma: Amplitude du bruit
        seed: Graine aléatoire (pour reproductibilité)
    
    Returns:
        Fonction (np.ndarray → np.ndarray) qui ajoute bruit
    
    Exemple:
        modifier = add_gaussian_noise(sigma=0.05, seed=42)
        D_noisy = modifier(D_base)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def modifier(state: np.ndarray) -> np.ndarray:
        """Ajoute bruit Gaussien à state."""
        noise = rng.randn(*state.shape) * sigma
        return state + noise
    
    return modifier


def add_uniform_noise(amplitude: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant une fonction qui ajoute du bruit uniforme.
    
    Args:
        amplitude: Amplitude du bruit (dans [-amplitude, +amplitude])
        seed: Graine aléatoire
    
    Returns:
        Fonction qui ajoute bruit uniforme
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def modifier(state: np.ndarray) -> np.ndarray:
        """Ajoute bruit uniforme à state."""
        noise = rng.uniform(-amplitude, amplitude, size=state.shape)
        return state + noise
    
    return modifier

"""
operators/gamma_hyp_001.py

HYP-GAM-001: Saturation pure pointwise (tanh)
"""

import numpy as np
from typing import Callable

class PureSaturationGamma:
    """Γ de saturation pure pointwise."""
    
    def __init__(self, beta: float = 2.0):
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * state)
    
    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"

def create_gamma_hyp_001(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    return PureSaturationGamma(beta=beta)

PARAM_GRID_PHASE1 = {'nominal': {'beta': 2.0}}
PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 0.5},
    'beta_nominal': {'beta': 1.0},
    'beta_high': {'beta': 2.0},
    'beta_very_high': {'beta': 5.0},
}

METADATA = {
    'id': 'GAM-001',
    'name': 'Saturation pure pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = tanh(β · T_n[i,j])',
    'parameters': {'beta': {'type': 'float', 'range': '(0, +∞)', 'nominal': 2.0}},
    'd_applicability': ['SYM', 'ASY', 'R3'],
}

"""
operators/gamma_hyp_002.py

HYP-GAM-002: Diffusion pure (Laplacien discret)

FORME: T_{n+1}[i,j] = T_n[i,j] + α·(somme_voisins - 4·T_n[i,j])

APPLICABLE: SYM, ASY (rang 2 uniquement)

ATTENDU: Homogénéisation, perte diversité
"""

import numpy as np
from typing import Callable


class PureDiffusionGamma:
    """
    Γ de diffusion pure via Laplacien discret.
    
    Opérateur de diffusion 2D avec voisinage 4-connexe:
    - Voisins: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    - Laplacien: ∇²T[i,j] = somme_voisins - 4·T[i,j]
    - Mise à jour: T_{n+1} = T_n + α·∇²T_n
    
    AVEUGLEMENT:
    - Ne connaît ni la dimension de l'état
    - Ne connaît ni son interprétation
    - Applique simplement la diffusion locale
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Coefficient de diffusion (doit être < 0.25 pour stabilité)
        """
        assert 0 < alpha < 0.25, "alpha doit être dans (0, 0.25) pour stabilité Von Neumann"
        self.alpha = alpha
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique diffusion à l'état.
        
        Args:
            state: Tenseur d'état (doit être rang 2)
        
        Returns:
            État diffusé
        
        Raises:
            ValueError: Si state n'est pas rang 2
        """
        if state.ndim != 2:
            raise ValueError(f"PureDiffusionGamma applicable uniquement rang 2, reçu rang {state.ndim}")
        
        n, m = state.shape
        
        # Calcule Laplacien avec conditions limites périodiques
        laplacian = np.zeros_like(state)
        
        # Voisin haut
        laplacian += np.roll(state, 1, axis=0)
        # Voisin bas
        laplacian += np.roll(state, -1, axis=0)
        # Voisin gauche
        laplacian += np.roll(state, 1, axis=1)
        # Voisin droite
        laplacian += np.roll(state, -1, axis=1)
        # Centre (4 voisins)
        laplacian -= 4 * state
        
        # Mise à jour diffusive
        return state + self.alpha * laplacian
    
    def __repr__(self):
        return f"PureDiffusionGamma(alpha={self.alpha})"


def create_gamma_hyp_002(alpha: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory pour créer GAM-002.
    
    Args:
        alpha: Coefficient de diffusion
    
    Returns:
        Instance callable de PureDiffusionGamma
    """
    return PureDiffusionGamma(alpha=alpha)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'alpha': 0.05}
}

PARAM_GRID_PHASE2 = {
    'alpha_low': {'alpha': 0.01},
    'alpha_nominal': {'alpha': 0.05},
    'alpha_high': {'alpha': 0.1},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-002',
    'name': 'Diffusion pure',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])',
    'parameters': {
        'alpha': {
            'type': 'float',
            'range': '(0, 0.25)',
            'nominal': 0.05,
            'description': 'Coefficient de diffusion (stabilité: α < 0.25)'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Rapide (<500 iterations)',
        'diversity': 'Perte totale (homogénéisation)',
        'attractors': 'Uniformes (toutes valeurs égales)',
        'trivial': True
    },
    'notes': [
        'Voisinage 4-connexe avec conditions périodiques',
        'Stabilité Von Neumann: α < 0.25',
        'Lisse toute structure initiale'
    ]
}

"""
operators/gamma_hyp_003.py

HYP-GAM-003: Croissance exponentielle pointwise

FORME: T_{n+1}[i,j] = T_n[i,j] · exp(γ)

ATTENDU: Explosion, REJECTED[GLOBAL] probable
"""

import numpy as np
from typing import Callable


class ExponentialGrowthGamma:
    """
    Γ de croissance exponentielle.
    
    Mécanisme: Amplification exponentielle de tous les éléments.
    
    ATTENDU:
    - Explosion rapide (toutes valeurs → ±∞)
    - Violation bornes systématique
    - Test de robustesse du framework aux explosions
    
    NOTE: Cet opérateur est conçu pour ÉCHOUER.
    Son rôle est de valider que le framework détecte correctement
    les explosions numériques.
    """
    
    def __init__(self, gamma: float = 0.05):
        """
        Args:
            gamma: Taux de croissance (> 0)
        """
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(gamma)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique croissance exponentielle.
        
        WARNING: Diverge exponentiellement rapidement.
        """
        return state * self._factor
    
    def __repr__(self):
        return f"ExponentialGrowthGamma(gamma={self.gamma})"


def create_gamma_hyp_003(gamma: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-003."""
    return ExponentialGrowthGamma(gamma=gamma)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'gamma': 0.05}
}

PARAM_GRID_PHASE2 = {
    'gamma_low': {'gamma': 0.01},
    'gamma_nominal': {'gamma': 0.05},
    'gamma_high': {'gamma': 0.1},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-003',
    'name': 'Croissance exponentielle pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] · exp(γ)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 0.05,
            'description': 'Taux de croissance exponentielle'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Jamais (divergence)',
        'diversity': 'Explosion',
        'attractors': 'Aucun (divergence)',
        'trivial': False,
        'expected_failure': True
    },
    'notes': [
        'CONÇU POUR ÉCHOUER',
        'Validation détection explosions',
        'Devrait obtenir REJECTED[GLOBAL]',
        'Explosion typique < 100 itérations pour γ=0.05'
    ]
}

"""
operators/gamma_hyp_004.py

HYP-GAM-004: Décroissance exponentielle pointwise

FORME: T_{n+1}[i,j] = T_n[i,j] · exp(-γ)

ATTENDU: Convergence vers 0, trivialité
"""

import numpy as np
from typing import Callable


class ExponentialDecayGamma:
    """
    Γ de décroissance exponentielle.
    
    Mécanisme: Atténuation exponentielle de tous les éléments → 0.
    
    ATTENDU:
    - Convergence rapide vers zéro
    - Perte totale d'information
    - Trivialité (attracteur zéro)
    """
    
    def __init__(self, gamma: float = 0.05):
        """
        Args:
            gamma: Taux de décroissance (> 0)
        """
        assert gamma > 0, "gamma doit être > 0"
        self.gamma = gamma
        self._factor = np.exp(-gamma)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique décroissance exponentielle.
        
        Convergence exponentielle: T → 0
        """
        return state * self._factor
    
    def __repr__(self):
        return f"ExponentialDecayGamma(gamma={self.gamma})"


def create_gamma_hyp_004(gamma: float = 0.05) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-004."""
    return ExponentialDecayGamma(gamma=gamma)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'gamma': 0.05}
}

PARAM_GRID_PHASE2 = {
    'gamma_low': {'gamma': 0.01},
    'gamma_nominal': {'gamma': 0.05},
    'gamma_high': {'gamma': 0.1},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-004',
    'name': 'Décroissance exponentielle pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = T_n[i,j] · exp(-γ)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 0.05,
            'description': 'Taux de décroissance exponentielle'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Rapide (<500 iterations)',
        'diversity': 'Perte totale',
        'attractors': 'Zéro (trivial)',
        'trivial': True
    },
    'notes': [
        'Convergence exponentielle vers 0',
        'Perte systématique information',
        'Temps caractéristique: 1/γ itérations',
        'Attendu: REJECTED[R0] pour trivialité'
    ]
}

"""
operators/gamma_hyp_005.py

HYP-GAM-005: Oscillateur harmonique linéaire

FORME: T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}

TYPE: Non-markovien ordre 1 (mémoire explicite)

ATTENDU: Oscillations périodiques, pas de complexité émergente
"""

import numpy as np
from typing import Callable, Optional


class HarmonicOscillatorGamma:
    """
    Γ oscillateur harmonique discret.
    
    Mécanisme:
    - Oscillation sinusoïdale de chaque élément
    - Conservation énergie (norme constante théoriquement)
    - Non-markovien: nécessite T_{n-1}
    
    ATTENDU:
    - Oscillations périodiques (période 2π/ω)
    - Pas de convergence
    - Pas d'émergence de structure
    """
    
    def __init__(self, omega: float = np.pi / 4):
        """
        Args:
            omega: Fréquence angulaire (rad/iteration)
        """
        self.omega = omega
        self._cos_omega = np.cos(omega)
        self._sin_omega = np.sin(omega)
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique rotation harmonique.
        
        Première itération: comportement identité.
        Suivantes: T_{n+1} = cos(ω)T_n - sin(ω)T_{n-1}
        """
        if self._previous_state is None:
            # Première itération: juste copier
            result = state.copy()
        else:
            # Oscillateur harmonique
            result = self._cos_omega * state - self._sin_omega * self._previous_state
        
        # Stocker pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire."""
        self._previous_state = None
    
    def __repr__(self):
        return f"HarmonicOscillatorGamma(omega={self.omega:.4f})"


def create_gamma_hyp_005(omega: float = np.pi / 4) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-005."""
    return HarmonicOscillatorGamma(omega=omega)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'omega': np.pi / 4}
}

PARAM_GRID_PHASE2 = {
    'omega_slow': {'omega': np.pi / 8},
    'omega_nominal': {'omega': np.pi / 4},
    'omega_fast': {'omega': np.pi / 2},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-005',
    'name': 'Oscillateur harmonique linéaire',
    'family': 'markovian',  # Techniquement non-markovien mais famille markovienne pure
    'form': 'T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}',
    'parameters': {
        'omega': {
            'type': 'float',
            'range': '(0, π)',
            'nominal': np.pi / 4,
            'description': 'Fréquence angulaire (période = 2π/ω)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Jamais (oscillations périodiques)',
        'diversity': 'Conservation (théorique)',
        'attractors': 'Cycle périodique',
        'trivial': False
    },
    'notes': [
        'Non-markovien ordre 1 (stocke T_{n-1})',
        'Conservation énergie théorique (norme constante)',
        'Période: 2π/ω itérations',
        'Pas de complexité émergente attendue',
        'Intéressant pour test détection périodicité'
    ]
}

"""
operators/gamma_hyp_006.py

HYP-GAM-006: Saturation avec mémoire ordre-1

FORME: T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))

TYPE: Non-markovien (stocke état précédent)

ATTENDU: Inertie temporelle, possibilité non-trivialité
"""

import numpy as np
from typing import Callable, Optional


class MemorySaturationGamma:
    """
    Γ avec mémoire ordre-1: combine saturation et inertie.
    
    Mécanisme:
    - Saturation: borne les valeurs via tanh
    - Mémoire: incorpore vélocité (T_n - T_{n-1})
    - Non-markovien: stocke explicitement T_{n-1}
    
    PATTERN NON-MARKOVIEN:
    Le kernel reste aveugle. C'est Γ lui-même qui gère sa mémoire
    via l'attribut interne _previous_state.
    """
    
    def __init__(self, beta: float = 1.0, alpha: float = 0.3):
        """
        Args:
            beta: Force de saturation (> 0)
            alpha: Poids de la mémoire [0, 1]
        """
        assert beta > 0, "beta doit être > 0"
        assert 0 <= alpha <= 1, "alpha doit être dans [0, 1]"
        
        self.beta = beta
        self.alpha = alpha
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique Γ(T_n, T_{n-1}).
        
        À la première itération: comportement markovien (pas de mémoire).
        Itérations suivantes: utilise mémoire stockée.
        """
        if self._previous_state is None:
            # Première itération: comportement markovien
            result = np.tanh(self.beta * state)
        else:
            # Itérations suivantes: mémoire
            # Calcule vélocité
            velocity = state - self._previous_state
            
            # Combine saturation + inertie
            combined = self.beta * state + self.alpha * velocity
            result = np.tanh(combined)
        
        # Stocke état actuel pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire (utile entre runs)."""
        self._previous_state = None
    
    def __repr__(self):
        return f"MemorySaturationGamma(beta={self.beta}, alpha={self.alpha})"


def create_gamma_hyp_006(beta: float = 1.0, alpha: float = 0.3) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-006."""
    return MemorySaturationGamma(beta=beta, alpha=alpha)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 1.0, 'alpha': 0.3}
}

PARAM_GRID_PHASE2 = {
    # Mémoire faible
    'mem_weak_low_sat': {'beta': 1.0, 'alpha': 0.1},
    'mem_weak_high_sat': {'beta': 2.0, 'alpha': 0.1},
    
    # Mémoire moyenne
    'mem_mid_low_sat': {'beta': 1.0, 'alpha': 0.3},
    'mem_mid_high_sat': {'beta': 2.0, 'alpha': 0.3},
    
    # Mémoire forte
    'mem_strong_low_sat': {'beta': 1.0, 'alpha': 0.5},
    'mem_strong_high_sat': {'beta': 2.0, 'alpha': 0.5},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-006',
    'name': 'Saturation + mémoire ordre-1',
    'family': 'non_markovian',
    'form': 'T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force de saturation'
        },
        'alpha': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.3,
            'description': 'Poids de la mémoire (inertie)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Plus lent que markovien (inertie)',
        'diversity': 'Possible préservation avec α adéquat',
        'attractors': 'Non-triviaux possibles',
        'trivial': False
    },
    'notes': [
        'Non-markovien: stocke T_{n-1} en interne',
        'Première itération: comportement markovien',
        'Inertie peut éviter attracteurs triviaux',
        'Appeler reset() entre runs différents'
    ]
}

"""
operators/gamma_hyp_007.py

HYP-GAM-007: Régulation par moyenne glissante

FORME: T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8(T_n))

ATTENDU: Homogénéisation douce locale
"""

import numpy as np
from typing import Callable


class SlidingAverageGamma:
    """
    Γ de régulation par moyenne des voisins 8-connexes.
    
    Mécanisme:
    - Chaque élément est mélangé avec moyenne de ses 8 voisins
    - Lissage progressif (ε contrôle force)
    - Similaire diffusion mais avec moyenne explicite
    
    ATTENDU:
    - Lissage progressif des structures
    - Homogénéisation locale puis globale
    - Perte diversité mais plus lente que diffusion pure
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Force de régulation [0, 1]
                    0 = identité, 1 = moyenne pure
        """
        assert 0 <= epsilon <= 1, "epsilon doit être dans [0, 1]"
        self.epsilon = epsilon
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique régulation par moyenne locale.
        
        Applicable: rang 2 uniquement (voisinage 2D).
        """
        if state.ndim != 2:
            raise ValueError(f"SlidingAverageGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        n, m = state.shape
        result = np.zeros_like(state)
        
        # Pour chaque élément, calculer moyenne voisins 8-connexes
        for i in range(n):
            for j in range(m):
                # Voisins avec conditions périodiques
                neighbors = [
                    state[(i-1) % n, (j-1) % m],  # Haut-gauche
                    state[(i-1) % n, j],          # Haut
                    state[(i-1) % n, (j+1) % m],  # Haut-droite
                    state[i, (j-1) % m],          # Gauche
                    state[i, (j+1) % m],          # Droite
                    state[(i+1) % n, (j-1) % m],  # Bas-gauche
                    state[(i+1) % n, j],          # Bas
                    state[(i+1) % n, (j+1) % m],  # Bas-droite
                ]
                
                mean_neighbors = np.mean(neighbors)
                
                # Mélange avec moyenne
                result[i, j] = (1 - self.epsilon) * state[i, j] + self.epsilon * mean_neighbors
        
        return result
    
    def __repr__(self):
        return f"SlidingAverageGamma(epsilon={self.epsilon})"


def create_gamma_hyp_007(epsilon: float = 0.1) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-007."""
    return SlidingAverageGamma(epsilon=epsilon)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'epsilon': 0.1}
}

PARAM_GRID_PHASE2 = {
    'epsilon_low': {'epsilon': 0.05},
    'epsilon_nominal': {'epsilon': 0.1},
    'epsilon_high': {'epsilon': 0.2},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-007',
    'name': 'Régulation moyenne glissante',
    'family': 'non_markovian',  # Bien que markovien, classé ici pour régulation
    'form': 'T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)',
    'parameters': {
        'epsilon': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.1,
            'description': 'Force de régulation (0=identité, 1=moyenne pure)'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Moyenne (500-1000 iterations)',
        'diversity': 'Perte progressive',
        'attractors': 'Uniformes (plus lent que diffusion)',
        'trivial': True
    },
    'notes': [
        'Voisinage 8-connexe (diagonales incluses)',
        'Plus doux que diffusion Laplacienne',
        'Implémentation O(N²) (peut être lent)',
        'Conditions périodiques'
    ]
}

"""
operators/gamma_hyp_008.py

HYP-GAM-008: Mémoire différentielle avec saturation

FORME: T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)

ATTENDU: "Friction informationnelle", oscillations contrôlées
"""

import numpy as np
from typing import Callable, Optional


class DifferentialMemoryGamma:
    """
    Γ combinant inertie, saturation et friction.
    
    Mécanisme:
    - Inertie: γ·(T_n - T_{n-1}) (vélocité)
    - Saturation: β·T_n (force de rappel)
    - Friction: contrôle via saturation globale
    
    ATTENDU:
    - Oscillations amorties si γ et β bien balancés
    - Friction informationnelle (ralentissement)
    - Convergence douce possible
    """
    
    def __init__(self, gamma: float = 0.3, beta: float = 1.0):
        """
        Args:
            gamma: Poids inertie [0, 1]
            beta: Force saturation (> 0)
        """
        assert 0 <= gamma <= 1, "gamma doit être dans [0, 1]"
        assert beta > 0, "beta doit être > 0"
        
        self.gamma = gamma
        self.beta = beta
        self._previous_state: Optional[np.ndarray] = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique mémoire différentielle.
        
        Première itération: comportement markovien.
        """
        if self._previous_state is None:
            # Première itération: saturation simple
            result = np.tanh(self.beta * state)
        else:
            # Vélocité
            velocity = state - self._previous_state
            
            # Combinaison
            combined = state + self.gamma * velocity + self.beta * state
            
            # Saturation globale
            result = np.tanh(combined)
        
        # Stocker pour prochaine itération
        self._previous_state = state.copy()
        
        return result
    
    def reset(self):
        """Réinitialise la mémoire."""
        self._previous_state = None
    
    def __repr__(self):
        return f"DifferentialMemoryGamma(gamma={self.gamma}, beta={self.beta})"


def create_gamma_hyp_008(gamma: float = 0.3, 
                         beta: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-008."""
    return DifferentialMemoryGamma(gamma=gamma, beta=beta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'gamma': 0.3, 'beta': 1.0}
}

PARAM_GRID_PHASE2 = {
    # Inertie faible
    'low_inertia_low_sat': {'gamma': 0.1, 'beta': 1.0},
    'low_inertia_high_sat': {'gamma': 0.1, 'beta': 2.0},
    
    # Inertie forte
    'high_inertia_low_sat': {'gamma': 0.5, 'beta': 1.0},
    'high_inertia_high_sat': {'gamma': 0.5, 'beta': 2.0},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-008',
    'name': 'Mémoire différentielle',
    'family': 'non_markovian',
    'form': 'T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)',
    'parameters': {
        'gamma': {
            'type': 'float',
            'range': '[0, 1]',
            'nominal': 0.3,
            'description': 'Poids inertie (vélocité)'
        },
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force saturation'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Oscillations amorties possibles',
        'diversity': 'Maintien possible avec γ adéquat',
        'attractors': 'Non-triviaux possibles',
        'trivial': False
    },
    'notes': [
        'Non-markovien ordre 1',
        'Combine inertie + saturation + friction',
        'Balance γ (inertie) vs β (saturation)',
        'Similaire GAM-006 mais avec terme β additionnel',
        'Appeler reset() entre runs différents',
        'Oscillations amorties si bien paramétré'
    ]
}

"""
operators/gamma_hyp_009.py

HYP-GAM-009: Saturation + bruit additif

FORME: T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)

TYPE: Stochastique

ATTENDU: Équilibre stochastique ou diffusion aléatoire
"""

import numpy as np
from typing import Callable


class StochasticSaturationGamma:
    """
    Γ combinant saturation et bruit additif gaussien.
    
    Mécanisme:
    - Saturation borne les valeurs
    - Bruit additif injecte exploration stochastique
    - Balance déterminisme / stochasticité
    
    ATTENDU:
    - Possibilité équilibre stochastique (si β fort, σ faible)
    - Ou marche aléatoire bornée (si σ fort)
    - Robustesse au bruit à tester
    """
    
    def __init__(self, beta: float = 1.0, sigma: float = 0.01, seed: int = None):
        """
        Args:
            beta: Force de saturation (> 0)
            sigma: Amplitude du bruit (≥ 0)
            seed: Graine aléatoire pour reproductibilité
        """
        assert beta > 0, "beta doit être > 0"
        assert sigma >= 0, "sigma doit être ≥ 0"
        
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique saturation + bruit.
        
        Note: Le bruit rend le processus non-déterministe.
        """
        # Saturation
        saturated = np.tanh(self.beta * state)
        
        # Bruit gaussien
        if self.sigma > 0:
            noise = self.rng.randn(*state.shape) * self.sigma
            result = saturated + noise
        else:
            result = saturated
        
        return result
    
    def reset(self):
        """Réinitialise le générateur aléatoire (optionnel)."""
        # Note: ne réinitialise pas vraiment, juste pour API consistente
        pass
    
    def __repr__(self):
        return f"StochasticSaturationGamma(beta={self.beta}, sigma={self.sigma})"


def create_gamma_hyp_009(beta: float = 1.0, sigma: float = 0.01, 
                         seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-009."""
    return StochasticSaturationGamma(beta=beta, sigma=sigma, seed=seed)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 1.0, 'sigma': 0.01, 'seed': 42}
}

PARAM_GRID_PHASE2 = {
    # Saturation faible, bruit faible
    'low_sat_low_noise': {'beta': 1.0, 'sigma': 0.01, 'seed': 42},
    
    # Saturation forte, bruit faible (équilibre possible)
    'high_sat_low_noise': {'beta': 2.0, 'sigma': 0.01, 'seed': 42},
    
    # Saturation faible, bruit fort (marche aléatoire)
    'low_sat_high_noise': {'beta': 1.0, 'sigma': 0.05, 'seed': 42},
    
    # Saturation forte, bruit fort
    'high_sat_high_noise': {'beta': 2.0, 'sigma': 0.05, 'seed': 42},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-009',
    'name': 'Saturation + bruit additif',
    'family': 'stochastic',
    'form': 'T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 1.0,
            'description': 'Force de saturation'
        },
        'sigma': {
            'type': 'float',
            'range': '[0, +∞)',
            'nominal': 0.01,
            'description': 'Amplitude du bruit gaussien'
        },
        'seed': {
            'type': 'int',
            'range': 'N',
            'nominal': 42,
            'description': 'Graine aléatoire (reproductibilité)'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Équilibre stochastique possible',
        'diversity': 'Maintien possible avec σ adéquat',
        'attractors': 'Distribution stationnaire',
        'trivial': False
    },
    'notes': [
        'Processus stochastique (non-déterministe)',
        'Fixer seed pour reproductibilité',
        'Balance déterminisme (β) / exploration (σ)',
        'TEST-UNIV-004 (sensibilité CI) particulièrement pertinent',
        'Moyenner sur plusieurs seeds pour analyses'
    ]
}

"""
operators/gamma_hyp_010.py

HYP-GAM-010: Bruit multiplicatif avec saturation

FORME: T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)

ATTENDU: Amplification sélective, risque avalanche
"""

import numpy as np
from typing import Callable


class MultiplicativeNoiseGamma:
    """
    Γ avec bruit multiplicatif + saturation.
    
    Mécanisme:
    - Bruit multiplicatif: amplifie grandes valeurs
    - Saturation: borne le résultat final
    - Amplification sélective selon valeur initiale
    
    ATTENDU:
    - Amplification différentielle (riches plus riches)
    - Possibilité avalanche si σ trop fort
    - Structures émergentes possibles
    """
    
    def __init__(self, sigma: float = 0.05, seed: int = None):
        """
        Args:
            sigma: Amplitude du bruit multiplicatif (≥ 0)
            seed: Graine aléatoire
        """
        assert sigma >= 0, "sigma doit être ≥ 0"
        self.sigma = sigma
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique bruit multiplicatif puis saturation.
        
        ATTENTION: σ trop grand peut causer explosions.
        """
        if self.sigma > 0:
            # Bruit multiplicatif
            noise = 1.0 + self.sigma * self.rng.randn(*state.shape)
            multiplied = state * noise
        else:
            multiplied = state
        
        # Saturation pour borner
        result = np.tanh(multiplied)
        
        return result
    
    def reset(self):
        """API consistente avec autres Γ stochastiques."""
        pass
    
    def __repr__(self):
        return f"MultiplicativeNoiseGamma(sigma={self.sigma})"


def create_gamma_hyp_010(sigma: float = 0.05, 
                         seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-010."""
    return MultiplicativeNoiseGamma(sigma=sigma, seed=seed)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'sigma': 0.05, 'seed': 42}
}

PARAM_GRID_PHASE2 = {
    'sigma_low': {'sigma': 0.01, 'seed': 42},
    'sigma_nominal': {'sigma': 0.05, 'seed': 42},
    'sigma_high': {'sigma': 0.1, 'seed': 42},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-010',
    'name': 'Bruit multiplicatif',
    'family': 'stochastic',
    'form': 'T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)',
    'parameters': {
        'sigma': {
            'type': 'float',
            'range': '[0, +∞)',
            'nominal': 0.05,
            'description': 'Amplitude du bruit multiplicatif'
        },
        'seed': {
            'type': 'int',
            'range': 'N',
            'nominal': 42,
            'description': 'Graine aléatoire'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': 'Variable (dépend σ)',
        'diversity': 'Possible augmentation (amplification)',
        'attractors': 'Structures amplifiées ou chaos',
        'trivial': False
    },
    'notes': [
        'Bruit multiplicatif: amplifie proportionnellement',
        'Différent de GAM-009 (bruit additif)',
        'Saturation nécessaire pour stabilité',
        'Risque avalanche si σ > 0.2',
        'Fixer seed pour reproductibilité',
        'Intéressant: compare GAM-009 vs GAM-010',
        'Peut créer hétérogénéité (riches plus riches)'
    ]
}

"""
operators/gamma_hyp_012.py

HYP-GAM-012: Préservation symétrie forcée

FORME: T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = saturation

ATTENDU: Réparation artificielle, robustesse au bruit asymétrique
"""

import numpy as np
from typing import Callable


class ForcedSymmetryGamma:
    """
    Γ qui force la symétrie par symétrisation explicite.
    
    Mécanisme:
    1. Applique transformation F (saturation)
    2. Symétrise: (F + F^T) / 2
    
    ATTENDU:
    - Préservation/création symétrie garantie
    - Robustesse au bruit asymétrique
    - Test si forçage artificiel aide non-trivialité
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: Force de saturation dans F
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique saturation puis symétrise.
        
        Applicable: rang 2 uniquement.
        """
        if state.ndim != 2:
            raise ValueError(f"ForcedSymmetryGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        # Saturation
        F = np.tanh(self.beta * state)
        
        # Symétrisation forcée
        result = (F + F.T) / 2.0
        
        return result
    
    def __repr__(self):
        return f"ForcedSymmetryGamma(beta={self.beta})"


def create_gamma_hyp_012(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-012."""
    return ForcedSymmetryGamma(beta=beta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'beta': 2.0}
}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 1.0},
    'beta_nominal': {'beta': 2.0},
    'beta_high': {'beta': 5.0},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-012',
    'name': 'Préservation symétrie forcée',
    'family': 'structural',
    'form': 'T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = tanh(β·)',
    'parameters': {
        'beta': {
            'type': 'float',
            'range': '(0, +∞)',
            'nominal': 2.0,
            'description': 'Force de saturation'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 uniquement
    'expected_behavior': {
        'convergence': 'Similaire GAM-001 mais symétrique',
        'diversity': 'Possible perte comme saturation pure',
        'attractors': 'Symétriques garantis',
        'trivial': 'Possible (comme GAM-001)'
    },
    'notes': [
        'Force symétrie de manière artificielle',
        'TEST-SYM-001 devrait toujours PASS',
        'TEST-SYM-002: peut créer symétrie depuis ASY',
        'Robuste au bruit asymétrique (M1, M2)',
        'Intéressant pour comparer avec GAM-001',
        'Question: forçage aide-t-il non-trivialité ?'
    ]
}

"""
operators/gamma_hyp_013.py

HYP-GAM-013: Renforcement hebbien local

FORME: T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]

ATTENDU: Émergence clusters, croissance non-linéaire
"""

import numpy as np
from typing import Callable


class HebbianReinforcementGamma:
    """
    Γ de renforcement hebbien (apprentissage non-supervisé).
    
    Mécanisme:
    - Renforce connexions selon corrélations locales
    - Produit matriciel T @ T (auto-corrélation)
    - Apprentissage hebbien: "neurons that fire together, wire together"
    
    ATTENDU:
    - Émergence de clusters (renforcement mutuel)
    - Croissance non-linéaire (risque explosion)
    - Structures hiérarchiques possibles
    
    WARNING: Instable sans régulation (saturation recommandée).
    """
    
    def __init__(self, eta: float = 0.01):
        """
        Args:
            eta: Taux d'apprentissage [0, 0.1]
                Valeurs typiques: 0.001-0.05
        """
        assert 0 <= eta <= 0.1, "eta doit être dans [0, 0.1] pour stabilité"
        self.eta = eta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique renforcement hebbien.
        
        Applicable: rang 2 uniquement (produit matriciel).
        
        FORME:
        T_{n+1} = T_n + η·(T_n @ T_n)
        
        où @ est le produit matriciel.
        """
        if state.ndim != 2:
            raise ValueError(f"HebbianReinforcementGamma applicable uniquement rang 2, reçu {state.ndim}")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"HebbianReinforcementGamma nécessite matrice carrée, reçu {state.shape}")
        
        # Produit matriciel (auto-corrélation)
        hebbian_term = state @ state
        
        # Mise à jour
        result = state + self.eta * hebbian_term
        
        return result
    
    def __repr__(self):
        return f"HebbianReinforcementGamma(eta={self.eta})"


def create_gamma_hyp_013(eta: float = 0.01) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-013."""
    return HebbianReinforcementGamma(eta=eta)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'eta': 0.01}
}

PARAM_GRID_PHASE2 = {
    'eta_very_low': {'eta': 0.001},
    'eta_low': {'eta': 0.01},
    'eta_high': {'eta': 0.05},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-013',
    'name': 'Renforcement hebbien local',
    'family': 'structural',
    'form': 'T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]',
    'parameters': {
        'eta': {
            'type': 'float',
            'range': '[0, 0.1]',
            'nominal': 0.01,
            'description': 'Taux d\'apprentissage hebbien'
        }
    },
    'd_applicability': ['SYM', 'ASY'],  # Rang 2 carré uniquement
    'expected_behavior': {
        'convergence': 'Instable (risque explosion)',
        'diversity': 'Augmentation (clusters)',
        'attractors': 'Structures émergentes ou explosion',
        'trivial': False
    },
    'notes': [
        'INSTABLE sans régulation additionnelle',
        'Produit matriciel T @ T (coûteux: O(N³))',
        'Nécessite matrices CARRÉES',
        'Risque explosion si η trop grand ou D mal conditionné',
        'Intéressant si combiné avec saturation (voir GAM-103[R1])',
        'Peut créer structures hiérarchiques',
        'TEST-UNIV-001 (norme) critique pour détecter explosions'
    ]
}

# prc_automation/batch_runner.py
"""
Batch Runner Charter 5.5 - Pipeline exécution complet.

Modes:
- --brut: Collecte données (db_raw)
- --test: Application tests (db_results)
- --verdict: Génération verdicts exploratoires (rapports)
- --all: Pipeline complet
"""

import argparse
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path

from tests.utilities.discovery import discover_active_tests
from tests.utilities.applicability import check as check_applicability
from tests.utilities.test_engine import TestEngine


class CriticalTestError(Exception):
    """Exception pour erreurs critiques nécessitant arrêt."""
    pass


# =============================================================================
# MODE BRUT (collecte données)
# =============================================================================

def run_batch_brut(args):
    """
    Exécute kernel pour toutes configs.
    Stocke dans db_raw uniquement.
    """
    print(f"\n{'='*70}")
    print("MODE BRUT - Collecte données")
    print(f"{'='*70}\n")
    
    gamma_id = args.gamma
    
    # TODO: Implémenter génération configs + exécution kernel
    # Pour l'instant, assume que db_raw existe déjà
    
    print(f"⚠️ Mode --brut assume db_raw existante")
    print(f"  Vérifier executions pour {gamma_id}...")
    
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE gamma_id = ?", (gamma_id,))
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        print(f"\n❌ Aucune exécution trouvée pour {gamma_id} dans db_raw")
        print(f"   Action: Exécuter kernel manuellement ou implémenter génération configs")
        sys.exit(1)
    
    print(f"✓ {count} exécutions trouvées pour {gamma_id}")


# =============================================================================
# MODE TEST (observations)
# =============================================================================

def run_batch_test(args):
    """
    Applique tests sur runs existants.
    Calcule observations.
    Stocke dans db_results.
    """
    print(f"\n{'='*70}")
    print("MODE TEST - Application tests")
    print(f"{'='*70}\n")
    
    gamma_id = args.gamma
    params_config_id = args.params
    
    # Vérifier db_raw
    exec_ids = get_exec_ids_for_gamma(gamma_id)
    if not exec_ids:
        print(f"❌ Aucune exécution trouvée pour {gamma_id} dans db_raw")
        print(f"   Action: Exécuter --brut d'abord")
        sys.exit(1)
    
    print(f"✓ {len(exec_ids)} exécutions trouvées")
    
    # Découvrir tests actifs
    print("\nDécouverte tests...")
    all_tests = discover_active_tests()
    print(f"✓ {len(all_tests)} tests actifs découverts")
    
    # Initialiser engine
    engine = TestEngine()
    
    # Compteurs
    total_observations = 0
    errors = []
    
    # Pour chaque run
    for i, exec_id in enumerate(exec_ids, 1):
        print(f"\n[{i}/{len(exec_ids)}] Processing exec_id={exec_id}")
        
        try:
            # Charger contexte
            context = load_execution_context(exec_id)
            
            # Charger premier snapshot pour state_shape
            first_snapshot = load_first_snapshot(exec_id)
            context['state_shape'] = first_snapshot.shape
            context['exec_id'] = exec_id  # ⚠️ TRAÇABILITÉ
            
            # Filtrer tests applicables
            applicable_tests = {}
            for test_id, test_module in all_tests.items():
                applicable, reason = check_applicability(test_module, context)
                if applicable:
                    applicable_tests[test_id] = test_module
            
            print(f"  {len(applicable_tests)}/{len(all_tests)} tests applicables")
            
            if not applicable_tests:
                continue
            
            # Charger history complète
            history = load_execution_history(exec_id)
            
            # Appliquer chaque test
            for test_id, test_module in applicable_tests.items():
                try:
                    # Phase : Observation
                    observation = engine.execute_test(
                        test_module, context, history, params_config_id
                    )
                    
                    # Stocker observation
                    store_test_observation(exec_id, observation)
                    total_observations += 1
                    
                    status = observation['status']
                    print(f"    ✓ {test_id}: {status}")
                
                except Exception as e:
                    error_msg = f"exec_id={exec_id}, test={test_id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"    ✗ {test_id}: {str(e)}")
        
        except Exception as e:
            error_msg = f"exec_id={exec_id}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ Erreur run: {str(e)}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ MODE TEST")
    print(f"{'='*70}")
    print(f"Observations générées: {total_observations}")
    print(f"Erreurs:               {len(errors)}")
    
    if errors:
        print("\nErreurs détaillées:")
        for err in errors[:10]:  # Limiter affichage
            print(f"  - {err}")


# =============================================================================
# MODE VERDICT (analyse exploratoire)
# =============================================================================

def run_batch_verdict(args):
    """
    Génère verdicts exploratoires sur observations existantes.
    
    Architecture 5.5 (NOUVEAU):
    - verdict_reporter orchestre verdict_engine + gamma_profiling
    - Génération rapports structurés selon Charter R0
    """
    print(f"\n{'='*70}")
    print("MODE VERDICT - Analyse exploratoire R0")
    print(f"{'='*70}\n")
    
    params_config_id = args.params
    verdict_config_id = args.verdict
    
    print(f"Params config:  {params_config_id}")
    print(f"Verdict config: {verdict_config_id}\n")
    
    # Vérifier que observations existent
    n_observations = count_observations(params_config_id)
    if n_observations == 0:
        print(f"❌ Aucune observation trouvée pour params={params_config_id}")
        print(f"   Action: Exécuter --mode test d'abord")
        sys.exit(1)
    
    print(f"✓ {n_observations} observations trouvées\n")
    
    # Import verdict_reporter (NOUVEAU)
    try:
        from tests.utilities.verdict_reporter import generate_verdict_report
    except ImportError as e:
        print(f"❌ Erreur import verdict_reporter: {e}")
        sys.exit(1)
    
    # Exécution pipeline (SIMPLIFIÉ)
    try:
        results = generate_verdict_report(
            params_config_id=params_config_id,
            verdict_config_id=verdict_config_id
        )
        
        # Résumé rapide
        print(f"\n{'='*70}")
        print("✓ VERDICT TERMINÉ")
        print(f"{'='*70}")
        print(f"Gammas profilés : {results['metadata']['data_summary']['n_gammas']}")
        print(f"Tests analysés  : {results['metadata']['data_summary']['n_tests']}")
        print(f"Répertoire      : {Path(results['report_paths']['summary']).parent}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Erreur génération verdict: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# MODE ALL (pipeline complet)
# =============================================================================

def run_batch_all(args):
    """Exécute pipeline complet en une commande."""
    print(f"\n{'#'*70}")
    print("# PIPELINE COMPLET: brut → test → verdict")
    print(f"{'#'*70}\n")
    
    # run_batch_brut(args)  # Si gamma spécifié
    # run_batch_test(args)  # Si gamma spécifié
    run_batch_verdict(args)
    
    print(f"\n{'#'*70}")
    print("# PIPELINE TERMINÉ")
    print(f"{'#'*70}\n")


# =============================================================================
# UTILITAIRES DATABASE
# =============================================================================

def get_exec_ids_for_gamma(gamma_id: str) -> list:
    """Récupère tous exec_ids pour une gamma."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Executions WHERE gamma_id = ?", (gamma_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def count_observations(params_config_id: str) -> int:
    """Compte observations SUCCESS pour une config."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM TestObservations 
        WHERE params_config_id = ? AND status = 'SUCCESS'
    """, (params_config_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def load_execution_context(exec_id: int) -> dict:
    """Charge contexte depuis db_raw."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gamma_id, d_encoding_id, modifier_id, seed, run_id
        FROM Executions WHERE id = ?
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"exec_id={exec_id} non trouvé dans db_raw")
    
    return {
        'gamma_id': row[0],
        'd_encoding_id': row[1],
        'modifier_id': row[2],
        'seed': row[3],
        'run_id': row[4]
    }


def load_first_snapshot(exec_id: int):
    """Charge premier snapshot pour déduire state_shape."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
        LIMIT 1
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"Aucun snapshot pour exec_id={exec_id}")
    
    # Décompresser
    import gzip
    import pickle
    state = pickle.loads(gzip.decompress(row[0]))
    return state


def load_execution_history(exec_id: int) -> list:
    """Charge history complète depuis db_raw."""
    import gzip
    import pickle
    
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT iteration, state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    history = []
    for row in cursor.fetchall():
        state = pickle.loads(gzip.decompress(row[1]))
        history.append(state)
    
    conn.close()
    return history


def store_test_observation(exec_id: int, observation: dict):
    """Stocke observation dans db_results."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    # Extraire stats pour colonnes rapides
    stats = observation.get('statistics', {})
    first_metric = list(stats.keys())[0] if stats else None
    
    if first_metric:
        stat_data = stats[first_metric]
        stat_initial = stat_data.get('initial')
        stat_final = stat_data.get('final')
        stat_min = stat_data.get('min')
        stat_max = stat_data.get('max')
        stat_mean = stat_data.get('mean')
        stat_std = stat_data.get('std')
    else:
        stat_initial = stat_final = stat_min = None
        stat_max = stat_mean = stat_std = None
    
    # Extraire evolution
    evol = observation.get('evolution', {})
    first_evol = list(evol.keys())[0] if evol else None
    
    if first_evol:
        evol_data = evol[first_evol]
        evolution_transition = evol_data.get('transition')
        evolution_trend = evol_data.get('trend')
        evolution_trend_coefficient = evol_data.get('slope')
    else:
        evolution_transition = evolution_trend = None
        evolution_trend_coefficient = None
    
    cursor.execute("""
        INSERT OR REPLACE INTO TestObservations (
            exec_id, test_name, test_category,
            params_config_id,
            applicable, status, message,
            stat_initial, stat_final, stat_min, stat_max, stat_mean, stat_std,
            evolution_transition, evolution_trend, evolution_trend_coefficient,
            observation_data,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exec_id,
        observation['test_name'],
        observation['test_category'],
        observation['config_params_id'],
        observation['status'] not in ['NOT_APPLICABLE', 'SKIPPED'],
        observation['status'],
        observation['message'],
        stat_initial, stat_final, stat_min, stat_max, stat_mean, stat_std,
        evolution_transition, evolution_trend, evolution_trend_coefficient,
        json.dumps(observation),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Runner Charter 5.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode brut (collecte données)
  python -m prc_automation.batch_runner --mode brut --gamma GAM-001
  
  # Mode test (observations)
  python -m prc_automation.batch_runner --mode test --gamma GAM-001 --params params_default_v1
  
  # Mode verdict (analyse exploratoire, configs par défaut)
  python -m prc_automation.batch_runner --mode verdict
  
  # Mode verdict (configs spécifiques)
  python -m prc_automation.batch_runner --mode verdict --params params_default_v1 --verdict verdict_strict_v1
  
  # Pipeline complet
  python -m prc_automation.batch_runner --mode all --gamma GAM-001
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['brut', 'test', 'verdict', 'all'],
                       help="Mode exécution")
    
    parser.add_argument('--gamma', default=None,
                       help="Gamma ID (ex: GAM-001) - Requis pour modes brut/test")
    
    parser.add_argument('--params', default='params_default_v1',
                       help="Params config ID (défaut: params_default_v1)")
    
    parser.add_argument('--verdict', default='verdict_default_v1',
                       help="Verdict config ID (défaut: verdict_default_v1)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validation arguments
    if args.mode in ['brut', 'test'] and not args.gamma:
        print("❌ Erreur: --gamma requis pour modes 'brut' et 'test'")
        sys.exit(1)
    
    # Exécution
    if args.mode == 'brut':
        run_batch_brut(args)
    elif args.mode == 'test':
        run_batch_test(args)
    elif args.mode == 'verdict':
        run_batch_verdict(args)
    elif args.mode == 'all':
        run_batch_all(args)


if __name__ == "__main__":
    main()


# tests/test_graph_001.py
"""
Propriétés graphe (interprétation adjacence).

Objectif :
- Analyser structure connectivité
- Détecter motifs réseau

Métriques :
- density : Densité connexions
- clustering_local : Transitivité locale
- degree_variance : Hétérogénéité degrés

Algorithmes utilisés :
- graph.density : Ratio arêtes/max
- graph.clustering_local : Clustering moyen
- graph.degree_variance : Variance degrés

Exclusions :
- Chemins plus courts : Trop coûteux pour R0
- Communautés : Nécessite algorithmes dédiés
"""

import numpy as np

TEST_ID = "GRA-001"
TEST_CATEGORY = "GRAPH"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   
  
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'density': {
        'registry_key': 'graph.density',
        'default_params': {
            'threshold': 0.1,  # Seuil arêtes
        },
        'post_process': 'round_4',
    },
    
    'clustering_local': {
        'registry_key': 'graph.clustering_local',
        'default_params': {
            'threshold': 0.1,
        },
        'post_process': 'round_4',
    },
    
    'degree_variance': {
        'registry_key': 'graph.degree_variance',
        'default_params': {
            'threshold': 0.1,
            'normalize': True,
        },
        'post_process': 'round_4',
    }
}

# tests/test_pat_001.py
"""
Diversité et concentration distribution.

Objectif :
- Mesurer dispersion valeurs
- Détecter émergence structures concentrées ou uniformes

Métriques :
- diversity_simpson : Indice diversité Simpson
- concentration_top10 : Concentration énergie dans top 10%
- uniformity : Proximité distribution uniforme

Algorithmes utilisés :
- pattern.diversity : Indice Simpson
- pattern.concentration_ratio : Ratio concentration
- pattern.uniformity : Distance à uniforme

Exclusions :
- Entropie Shannon : Redondant avec diversity
- Coefficient Gini : Approximé par concentration_ratio
"""

import numpy as np

TEST_ID = "PAT-001"
TEST_CATEGORY = "PAT"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   

APPLICABILITY_SPEC = {
    "requires_rank": None,  # Tout rang
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'diversity_simpson': {
        'registry_key': 'pattern.diversity',
        'default_params': {
            'bins': 50,
        },
        'post_process': 'round_4',
    },
    
    'concentration_top10': {
        'registry_key': 'pattern.concentration_ratio',
        'default_params': {
            'top_percent': 0.1,
        },
        'post_process': 'round_4',
    },
    
    'uniformity': {
        'registry_key': 'pattern.uniformity',
        'default_params': {
            'bins': 50,
        },
        'post_process': 'round_4',
    }
}

# tests/test_spatial_001.py
"""
Rugosité et lissage spatial.

Objectif :
- Mesurer complexité structure spatiale
- Détecter transitions rugosité/lissage

Métriques :
- gradient_magnitude : Amplitude variations spatiales
- laplacian_energy : Rugosité (courbure locale)
- smoothness : Inverse rugosité normalisé

Algorithmes utilisés :
- spatial.gradient_magnitude : Norme gradient moyen
- spatial.laplacian_energy : Énergie laplacien
- spatial.smoothness : Score lissage

Exclusions :
- Variance locale : Corrélée avec gradient
- Détection contours : Trop spécifique pour R0
"""

import numpy as np

TEST_ID = "SPA-001"
TEST_CATEGORY = "SPATIAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,  # Gradients nécessitent espace
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'gradient_magnitude': {
        'registry_key': 'spatial.gradient_magnitude',
        'default_params': {
            'normalize': True,
        },
        'post_process': 'round_6',
    },
    
    'laplacian_energy': {
        'registry_key': 'spatial.laplacian_energy',
        'default_params': {
            'normalize': True,
        },
        'post_process': 'round_6',
    },
    
    'smoothness': {
        'registry_key': 'spatial.smoothness',
        'default_params': {},
        'post_process': 'round_4',
    }
}

# tests/test_spectral_001.py
"""
Évolution spectre valeurs propres.

Objectif :
- Observer évolution distribution spectrale
- Détecter concentration/dispersion énergie

Métriques :
- eigenvalue_max : Plus grande valeur propre (dominance)
- spectral_gap : Écart λ₁ - λ₂ (séparation)
- spectral_radius : Rayon spectral (stabilité itérations)

Algorithmes utilisés :
- spectral.eigenvalue_max : Calcul λ_max
- spectral.spectral_gap : Écart spectral

Exclusions :
- Entropie spectrale : Redondant avec statistical tests
- Valeurs propres individuelles : Trop détaillé pour R0
"""

import numpy as np

TEST_ID = "SPE-001"
TEST_CATEGORY = "SPECTRAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,  
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'eigenvalue_max': {
        'registry_key': 'spectral.eigenvalue_max',
        'default_params': {
            'absolute': True,
        },
        'post_process': 'round_4',
    },
    
    'spectral_gap': {
        'registry_key': 'spectral.spectral_gap',
        'default_params': {
            'normalize': True,  # Valeur relative
        },
        'post_process': 'round_6',
    }
    
}

# tests/test_spectral_001.py
"""
Évolution spectre valeurs propres.

Objectif :
- Observer évolution distribution spectrale
- Détecter concentration/dispersion énergie

Métriques :
- eigenvalue_max : Plus grande valeur propre (dominance)
- spectral_gap : Écart λ₁ - λ₂ (séparation)
- spectral_radius : Rayon spectral (stabilité itérations)

Algorithmes utilisés :
- spectral.spectral_radius : Rayon spectral

Exclusions :
- Entropie spectrale : Redondant avec statistical tests
- Valeurs propres individuelles : Trop détaillé pour R0
"""

import numpy as np

TEST_ID = "SPE-002"
TEST_CATEGORY = "SPECTRAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 3,  # Gap nécessite ≥3 valeurs propres
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    
    'spectral_radius': {
        'registry_key': 'spectral.spectral_radius',
        'default_params': {},
        'post_process': 'round_4',
    }
}

# prc_framework/tests/test_sym_001.py
"""
Évolution asymétrie matrices.

Objectif :
- Mesurer création/destruction/préservation symétrie

Métriques :
- asymmetry_norm : Discriminant principal (global)
- asymmetry_norm_normalized : Comparable entre tailles différentes

Algorithmes utilisés :
- algebra.matrix_asymmetry : Norme ||A - A^T||, paramétrable

Exclusions :
- Trace asymétrie : Masque patterns spatiaux (trop agrégée)
- Max asymétrie : Sensible outliers, peu robuste
"""

import numpy as np

TEST_ID = "SYM-001"
TEST_CATEGORY = "SYM"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["SYM", "ASY"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'asymmetry_norm': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {
            'norm_type': 'frobenius',
            'normalize': False,
        },
        'post_process': 'round_6',
    },
    
    'asymmetry_norm_normalized': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {
            'norm_type': 'frobenius',
            'normalize': True,
        },
        'post_process': 'round_6',
    }
}

# tests/test_topological_001.py
"""
Invariants topologiques simplifiés.

Objectif :
- Observer changements topologiques
- Détecter création/destruction structures

Métriques :
- connected_components : Fragmentation
- holes_count : Nombre trous
- euler_characteristic : Invariant χ

Algorithmes utilisés :
- topological.connected_components : Comptage composantes
- topological.holes_count : Détection trous
- topological.euler_characteristic : Calcul χ

Exclusions :
- Homologie persistante : Hors scope R0
- Betti numbers : Nécessite bibliothèques spécialisées
"""

import numpy as np

TEST_ID = "TOP-001"
TEST_CATEGORY = "TOPOLOGICAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 10,  # Topologie nécessite espace
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'connected_components': {
        'registry_key': 'topological.connected_components',
        'default_params': {
            'threshold': 0.0,
            'connectivity': 1,
        },
        'post_process': 'round_2',
    },
    
    'holes_count': {
        'registry_key': 'topological.holes_count',
        'default_params': {
            'threshold': 0.0,
            'min_hole_size': 4,
        },
        'post_process': 'round_2',
    },
    
    'euler_characteristic': {
        'registry_key': 'topological.euler_characteristic',
        'default_params': {
            'threshold': 0.0,
        },
        'post_process': 'round_2',
    }
}

# prc_framework/tests/test_univ_001.py
"""
Évolution norme Frobenius.

Objectif :
- Mesurer stabilité globale du tenseur sous action Γ

Métriques :
- frobenius_norm : Discrimine explosions/effondrements/stabilité

Algorithmes utilisés :
- algebra.matrix_norm : Standard, robuste, O(n²)

Exclusions :
- Norme spectrale : Trop coûteuse, peu discriminante ici
- Norme nucléaire : Redondante avec Frobenius pour détection explosions
"""

import numpy as np

TEST_ID = "UNIV-001"
TEST_CATEGORY = "UNIV"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": None,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'frobenius_norm': {
        'registry_key': 'algebra.matrix_norm',
        'default_params': {
            'norm_type': 'frobenius',
        },
        'post_process': 'round_4',
    }
}

# tests/test_univ_002.py
"""
Évolution trace normalisée.

Objectif :
- Mesurer stabilité trace (somme diagonale)
- Détection comportements pathologiques (explosion trace)

Métriques :
- trace_normalized : Trace / dimension (moyenne diagonale)
- trace_variance : Variance trace sur temps

Algorithmes utilisés :
- algebra.trace_value : Calcul standard trace

Exclusions :
- Déterminant : Trop sensible petites variations
- Valeurs propres individuelles : Redondant avec spectral tests
"""

import numpy as np

TEST_ID = "UNIV-002"
TEST_CATEGORY = "UNIV"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'trace_normalized': {
        'registry_key': 'algebra.trace_value',
        'default_params': {
            'normalize': True,  # Diviser par dimension
        },
        'post_process': 'round_4',
    },
    
    'trace_absolute': {
        'registry_key': 'algebra.trace_value',
        'default_params': {
            'normalize': False,
        },
        'post_process': 'round_2',
    }
}

# tests/utilities/HUB/profiling_runner.py
"""
Orchestration profiling multi-axes avec découverte automatique.

RESPONSABILITÉ :
- Exécuter profiling tous axes (test, gamma, modifier, encoding)
- Enrichir axe test (discriminant_power)
- Format retour unifié strict

ARCHITECTURE :
- Délégation profiling_common (profiling individuel)
- Délégation cross_profiling (enrichissements)
- Zéro calcul inline (orchestration pure)

UTILISATEURS :
- verdict_reporter.py (génération rapports complets)
- Scripts analyse (profiling_analysis.py si futur)

INTERDICTIONS (voir PRC_DEPENDENCY_RULES.md) :
- Implémenter profiling (→ profiling_common)
- Implémenter calculs (→ UTIL, registries)
- Modifier observations (lecture seule)
"""

from typing import Dict, List
from collections import defaultdict

# Profiling modules
from ..profiling_common import (
    profile_all_tests,
    profile_all_gammas,
    profile_all_modifiers,
    profile_all_encodings,
    compare_tests_summary,
    compare_gammas_summary,
    compare_modifiers_summary,
    compare_encodings_summary,
)

# Cross-profiling
from ..cross_profiling import compute_all_discriminant_powers


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ordre exécution axes (Charter R5.1-D)
DEFAULT_AXES_ORDER = ['test', 'gamma', 'modifier', 'encoding']


# ============================================================================
# DÉCOUVERTE AUTOMATIQUE (si besoin futur)
# ============================================================================

def discover_profiling_axes() -> List[str]:
    """
    Découvre axes profiling disponibles.
    
    R0 : Retourne liste hardcodée (4 axes)
    R1+ : Introspection profiling_common (detect profile_all_*)
    
    Returns:
        Liste axes disponibles
    """
    # R0 : Hardcodé (4 axes connus)
    return DEFAULT_AXES_ORDER.copy()


# ============================================================================
# ORCHESTRATION PRINCIPALE
# ============================================================================

def run_all_profiling(
    observations: List[dict],
    axes: List[str] = None
) -> Dict[str, Dict]:
    """
    Exécute profiling tous axes demandés.
    
    DÉLÉGATION STRICTE :
    - Profiling individuel → profiling_common.profile_all_{axis}()
    - Comparaisons → profiling_common.compare_{axis}_summary()
    - Enrichissement test → cross_profiling.compute_all_discriminant_powers()
    
    Args:
        observations: Liste observations SUCCESS
        axes: Axes à profiler (None = tous)
    
    Returns:
        {
            'test': {
                'profiles': {...},
                'summary': {...},
                'discriminant_powers': {...},  # Enrichissement spécifique
                'metadata': {...}
            },
            'gamma': {...},
            'modifier': {...},
            'encoding': {...}
        }
    
    Raises:
        ValueError: Si axe inconnu
    
    Examples:
        >>> results = run_all_profiling(observations)
        >>> results['gamma']['profiles']['GAM-001']
        {...}
        
        >>> results_partial = run_all_profiling(observations, axes=['gamma', 'test'])
    """
    if axes is None:
        axes = DEFAULT_AXES_ORDER
    
    # Validation axes
    valid_axes = discover_profiling_axes()
    for axis in axes:
        if axis not in valid_axes:
            raise ValueError(
                f"Axe inconnu '{axis}'. Valides: {valid_axes}"
            )
    
    results = {}
    
    for axis in axes:
        # Appel dynamique fonctions profiling_common
        # profile_all_{axis}() + compare_{axis}_summary()
        
        axis_plural = f"{axis}s" if axis != 'encoding' else 'encodings'
        
        profile_func_name = f"profile_all_{axis_plural}"
        compare_func_name = f"compare_{axis_plural}_summary"
        
        # Récupération fonctions
        profile_func = globals()[profile_func_name]
        compare_func = globals()[compare_func_name]
        
        # Profiling axe
        profiles = profile_func(observations)
        summary = compare_func(profiles)
        
        # Structure retour
        result = {
            'profiles': profiles,
            'summary': summary,
            'metadata': {
                'axis': axis,
                'n_entities': len(profiles),
                'n_observations': len(observations),
                'profiling_version': '6.1',
                'profiling_module': 'profiling_common'
            }
        }
        
        # Enrichissement spécifique axe test
        if axis == 'test':
            result['discriminant_powers'] = compute_all_discriminant_powers(
                profiles, reference_axis='gamma'
            )
        
        results[axis] = result
    
    return results


def run_profiling_single_axis(
    observations: List[dict],
    axis: str
) -> Dict:
    """
    Profiling un seul axe (helper).
    
    Args:
        observations: Liste observations
        axis: Axe à profiler ('test', 'gamma', 'modifier', 'encoding')
    
    Returns:
        Résultat profiling axe
    """
    results = run_all_profiling(observations, axes=[axis])
    return results[axis]


# ============================================================================
# HELPERS EXTRACTION (si besoin reporting)
# ============================================================================

def get_entity_profile(
    profiling_results: Dict,
    axis: str,
    entity_id: str
) -> Dict:
    """
    Extrait profil d'une entité spécifique.
    
    Args:
        profiling_results: Résultats run_all_profiling()
        axis: Axe concerné
        entity_id: ID entité (ex: 'GAM-001', 'SYM-001')
    
    Returns:
        Profil entité (tests + metadata)
    
    Raises:
        KeyError: Si axe ou entité introuvable
    """
    if axis not in profiling_results:
        raise KeyError(f"Axe '{axis}' non trouvé dans résultats")
    
    axis_results = profiling_results[axis]
    profiles = axis_results['profiles']
    
    if entity_id not in profiles:
        raise KeyError(
            f"Entité '{entity_id}' non trouvée dans axe '{axis}'. "
            f"Disponibles: {list(profiles.keys())}"
        )
    
    return profiles[entity_id]


def get_test_profile_for_entity(
    profiling_results: Dict,
    axis: str,
    entity_id: str,
    test_name: str
) -> Dict:
    """
    Extrait profil d'un test spécifique pour une entité.
    
    Args:
        profiling_results: Résultats run_all_profiling()
        axis: Axe concerné
        entity_id: ID entité
        test_name: ID test (ex: 'SYM-001')
    
    Returns:
        Profil test pour entité
    
    Raises:
        KeyError: Si non trouvé
    """
    entity_profile = get_entity_profile(profiling_results, axis, entity_id)
    
    if test_name not in entity_profile['tests']:
        raise KeyError(
            f"Test '{test_name}' non trouvé pour {entity_id}. "
            f"Disponibles: {list(entity_profile['tests'].keys())}"
        )
    
    return entity_profile['tests'][test_name]
    
# tests/utilities/test_engine.py
"""
Test Engine Charter 5.4 - Génération observations pures.
"""

import numpy as np
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional

from .registries.registry_manager import RegistryManager
from .config_loader import get_loader


# =============================================================================
# NOUVEAU : DÉTECTION ÉVÉNEMENTS DYNAMIQUES
# =============================================================================

def detect_dynamic_events(values: np.ndarray) -> dict:
    """
    Détecte événements dynamiques sur trajectoire métrique.
    
    Événements R0 :
    - deviation_onset : |val - initial| > 0.1 * |initial|
    - instability_onset : |diff| > P90(diffs) * 10
    - oscillatory : nb sign_changes > 10% iterations
    - saturation : std(last_20%) / mean(last_20%) < 0.05
    - collapse : any(|last_10| < 1e-10) and max(|all|) > 1.0
    
    Args:
        values: array (n_iterations,) - trajectoire métrique
    
    Returns:
        {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool
        }
    """
    if len(values) < 2:
        return {
            'deviation_onset': None,
            'instability_onset': None,
            'oscillatory': False,
            'saturation': False,
            'collapse': False
        }
    
    # 1. Deviation onset (>10% initial)
    initial = values[0]
    deviations = np.abs(values - initial) / (np.abs(initial) + 1e-10)
    deviation_idx = np.where(deviations > 0.1)[0]
    deviation_onset = int(deviation_idx[0]) if len(deviation_idx) > 0 else None
    
    # 2. Instability onset (|diff| > P90 * 10)
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)
    threshold_instability = np.percentile(abs_diffs, 90) * 10
    instability_idx = np.where(abs_diffs > threshold_instability)[0]
    instability_onset = int(instability_idx[0]) if len(instability_idx) > 0 else None
    
    # 3. Oscillations (>10% sign changes)
    signs = np.sign(diffs)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    oscillatory = bool(sign_changes > len(values) * 0.1)
    
    # 4. Saturation (std(last_20%) / mean < 5%)
    last_20pct = max(int(len(values) * 0.2), 1)
    final_segment = values[-last_20pct:]
    saturation = bool((np.std(final_segment) / (np.abs(np.mean(final_segment)) + 1e-10)) < 0.05)
    
    # 5. Collapse (retour brutal à ~0)
    last_10 = values[-10:] if len(values) >= 10 else values
    collapse = bool(np.any(np.abs(last_10) < 1e-10) and np.max(np.abs(values)) > 1.0)
    
    return {
        'deviation_onset': deviation_onset,
        'instability_onset': instability_onset,
        'oscillatory': oscillatory,
        'saturation': saturation,
        'collapse': collapse
    }


def compute_event_sequence(
    events: dict,
    n_iterations: int
) -> dict:
    """
    Construit séquence ordonnée depuis événements.
    
    Calcule onsets RELATIFS (fraction durée totale) pour timelines.
    
    Args:
        events: Retour de detect_dynamic_events()
        n_iterations: Nombre total itérations (pour calcul relatif)
    
    Returns:
        {
            'sequence': ['deviation', 'instability'],
            'sequence_timing': [3, 7],
            'sequence_timing_relative': [0.015, 0.035],
            'saturation_onset_estimated': bool
        }
    """
    timed_events = []
    saturation_estimated = False
    
    # Événements avec onset ponctuel
    if events['deviation_onset'] is not None:
        onset_abs = events['deviation_onset']
        onset_rel = onset_abs / n_iterations
        timed_events.append(('deviation', onset_abs, onset_rel))
    
    if events['instability_onset'] is not None:
        onset_abs = events['instability_onset']
        onset_rel = onset_abs / n_iterations
        timed_events.append(('instability', onset_abs, onset_rel))
    
    # Saturation : onset estimé à 80% (heuristique R0)
    if events['saturation']:
        onset_abs = int(0.80 * n_iterations)
        onset_rel = 0.80
        timed_events.append(('saturation', onset_abs, onset_rel))
        saturation_estimated = True
    
    # Collapse : onset estimé à 90% (fin de run)
    if events['collapse']:
        onset_abs = int(0.90 * n_iterations)
        onset_rel = 0.90
        timed_events.append(('collapse', onset_abs, onset_rel))
    
    # Oscillatory : pas d'onset (comportement global)
    # Inséré si présent mais pas dans séquence temporelle
    
    # Trier par timing absolu
    timed_events.sort(key=lambda x: x[1])
    
    return {
        'sequence': [name for name, _, _ in timed_events],
        'sequence_timing': [timing_abs for _, timing_abs, _ in timed_events],
        'sequence_timing_relative': [timing_rel for _, _, timing_rel in timed_events],
        'saturation_onset_estimated': saturation_estimated,
        'oscillatory_global': events['oscillatory']
    }
def patch_execute_test_dynamic_events(
    metric_buffers: Dict[str, List[float]],
    n_iterations: int
) -> Tuple[dict, dict]:
    """
    Calcule dynamic_events + timeseries pour tous metrics.
    
    À insérer dans TestEngine.execute_test() après boucle itérations.
    
    Args:
        metric_buffers: {metric_name: [val_0, ..., val_N]}
        n_iterations: Nombre total itérations (len(history))
    
    Returns:
        (dynamic_events, timeseries)
        - dynamic_events: {metric_name: {events + sequence}}
        - timeseries: {metric_name: [val_0, ..., val_N]}
    """
    dynamic_events = {}
    timeseries = {}
    
    for metric_name, values in metric_buffers.items():
        # Stocker timeseries (optionnel, lourd)
        timeseries[metric_name] = list(values)
        
        if len(values) < 2:
            continue
        
        # Détecter événements
        events = detect_dynamic_events(np.array(values))
        
        # Calculer séquence + onsets relatifs
        seq_info = compute_event_sequence(events, n_iterations)
        
        # Fusionner
        dynamic_events[metric_name] = {
            **events,
            **seq_info
        }
    
    return dynamic_events, timeseries
    
    
class TestEngine:
    """
    Moteur exécution tests PRC 5.4.
    
    Responsabilités :
    1. Valider COMPUTATION_SPECS via RegistryManager
    2. Exécuter formules sur tous snapshots
    3. Appliquer post_processors
    4. Calculer statistics/evolution
    5. Retourner dict standardisé avec exec_id
    """
    
    VERSION = "5.5"
    
    def __init__(self):
        self.registry_manager = RegistryManager()
        self.computation_cache: Dict[str, Dict] = {}
        self.config_loader = get_loader()
    
    def execute_test(
        self,
        test_module,
        run_metadata: Dict[str, Any],
        history: List[np.ndarray],
        params_config_id: str
    ) -> Dict[str, Any]:
        """
        Exécute un test.
        
        Args:
            test_module: Module test importé
            run_metadata: {exec_id, gamma_id, d_encoding_id, modifier_id, seed, state_shape}
            history: Liste complète des snapshots
            params_config_id: ID config params
        
        Returns:
            Dict observation format 5.4 avec exec_id
        """
        result = self._init_result(test_module, run_metadata, params_config_id)
        
        try:
            # Charger params YAML
            params = self.config_loader.load(
                config_type='params',
                config_id=params_config_id,
                test_id=test_module.TEST_ID
            )
            
            if not params or not isinstance(params, dict):
                raise ValueError(f"Config {params_config_id} invalide")
            
            # Extraire params
            common_params = params.get('common', {})
            if not common_params:
                raise ValueError(f"Config {params_config_id} manque section 'common'")
            
            category = test_module.TEST_CATEGORY.lower()
            if category in params:
                common_params = {**common_params, **params[category]}
            
            # Préparer computations
            computations = self._prepare_computations(
                test_module.COMPUTATION_SPECS,
                params_config_id
            )
            
            if not computations:
                result['status'] = 'ERROR'
                result['message'] = 'Aucune spécification valide'
                if result['status'] == 'ERROR':
                    print(result.get('message'))
                    print(result.get('traceback'))
                return result
            
            # Buffers
            metric_buffers = {name: [] for name in computations.keys()}
            skipped_iterations = {}
            
            # Exécution
            start_time = time.time()
            
            for iteration, snapshot in enumerate(history):
                for metric_name, computation in computations.items():
                    try:
                        func = computation['function']
                        func_params = computation['params']
                        
                        raw_value = func(snapshot, **func_params)
                        
                        if computation['post_process']:
                            raw_value = computation['post_process'](raw_value)
                        
                        if not np.isfinite(raw_value):
                            raise ValueError(f"Valeur non finie: {raw_value}")
                        
                        metric_buffers[metric_name].append(float(raw_value))
                    
                    except Exception as e:
                        if metric_name not in skipped_iterations:
                            skipped_iterations[metric_name] = []
                        skipped_iterations[metric_name].append({
                            'iteration': iteration,
                            'error': str(e)
                        })
             # NOUVEAU : Calculer événements dynamiques + séquence
            n_iterations = len(history)           
            execution_time = time.time() - start_time
            dynamic_events, timeseries = patch_execute_test_dynamic_events(
                metric_buffers,
                n_iterations
            )
            

            # Compiler résultats
            return self._compile_results(
                result, metric_buffers, skipped_iterations,
                computations, execution_time, common_params,
                dynamic_events, timeseries  # ← AJOUTER
            )
            
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = f"Erreur: {str(e)}"
            result['traceback'] = traceback.format_exc()
            if result['status'] == 'ERROR':
                print(result.get('message'))
                print(result.get('traceback'))
            return result
            
            
    def _init_result(
        self,
        test_module,
        run_metadata: Dict,
        params_config_id: str
    ) -> Dict:
        """Initialise structure résultat avec exec_id."""
        return {
            # ⚠️ TRAÇABILITÉ COMPLÈTE
            'exec_id': run_metadata.get('exec_id'),
            
            'run_metadata': {
                'gamma_id': run_metadata['gamma_id'],
                'd_encoding_id': run_metadata['d_encoding_id'],
                'modifier_id': run_metadata['modifier_id'],
                'seed': run_metadata['seed'],
            },
            
            'test_name': test_module.TEST_ID,
            'test_category': test_module.TEST_CATEGORY,
            'test_version': test_module.TEST_VERSION,
            'config_params_id': params_config_id,
            
            'status': 'PENDING',
            'message': '',
            
            'statistics': {},
            'evolution': {},
            'dynamic_events': {},
            'timeseries': {},
            
            
            'metadata': {
                'engine_version': self.VERSION,
                'computations': {},
            }
        }
    
    def _prepare_computations(
        self,
        specs: Dict,
        config_id: str
    ) -> Dict:
        """Prépare et valide toutes les spécifications."""
        computations = {}
        
        for metric_name, spec in specs.items():
            cache_key = f"{config_id}_{metric_name}_{hash(str(spec))}"
            
            if cache_key in self.computation_cache:
                computations[metric_name] = self.computation_cache[cache_key]
                continue
            
            try:
                prepared = self.registry_manager.validate_computation_spec(spec)
                computations[metric_name] = prepared
                self.computation_cache[cache_key] = prepared
            
            except Exception as e:
                print(f"[TestEngine] Ignoré '{metric_name}': {e}")
                continue
        
        return computations
    
    def _compile_results(
        self,
        result: Dict,
        buffers: Dict,
        skipped: Dict,
        computations: Dict,
        exec_time: float,
        params: dict,
        dynamic_events: dict,  # ← NOUVEAU
        timeseries: dict       # ← NOUVEAU
        ) -> Dict:
        """Compile résultats finaux."""
        for metric_name, values in buffers.items():
            if len(values) < 2:
                continue
            
            # Statistics
            result['statistics'][metric_name] = {
                'initial': values[0],
                'final': values[-1],
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'q1': float(np.percentile(values, 25)),
                'q3': float(np.percentile(values, 75)),
                'n_valid': len(values),
            }
            
            # Evolution
            result['evolution'][metric_name] = self._analyze_evolution(values, params)

            # Metadata
            result['metadata']['computations'][metric_name] = {
                'registry_key': computations[metric_name]['registry_key'],
                'params_used': computations[metric_name]['params'],
                'has_post_process': computations[metric_name]['post_process'] is not None,
            }
        
        # Ajouter dynamic_events + timeseries
        result['dynamic_events'] = dynamic_events
        result['timeseries'] = timeseries  # Optionnel (lourd en stockage)
            
        result['metadata'].update({
            'execution_time_sec': exec_time,
            'num_iterations_processed': len(next(iter(buffers.values()), [])),
            'total_metrics': len(buffers),
            'successful_metrics': sum(1 for v in buffers.values() if len(v) >= 2),
            'skipped_iterations': skipped,
        })
        
        if not result['statistics']:
            result['status'] = 'ERROR'
            result['message'] = 'Aucune métrique valide'
        else:
            result['status'] = 'SUCCESS'
            total_skipped = sum(len(v) for v in skipped.values())
            if total_skipped > 0:
                result['message'] = f"SUCCESS avec {total_skipped} itérations sautées"
            else:
                result['message'] = 'SUCCESS'
        
        return result
    
    def _analyze_evolution(self, values: List[float], params: dict) -> Dict:
        """Analyse évolution série temporelle."""
        if len(values) < 2:
            return {'transition': 'insufficient_data', 'trend': 'unknown'}
        
        explosion_threshold = params.get('explosion_threshold', 1000.0)
        stability_tolerance = params.get('stability_tolerance', 0.1)
        growth_factor = params.get('growth_factor', 1.5)
        shrink_factor = params.get('shrink_factor', 0.5)
        epsilon = params.get('epsilon', 1e-10)
        
        # Tendance
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])
        
        if abs(slope) < epsilon:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Transition
        initial = values[0]
        final = values[-1]
        max_val = max(values)
        relative_change = abs(final - initial) / (abs(initial) + epsilon)
        
        if max_val > explosion_threshold:
            transition = "explosive"
        elif relative_change < stability_tolerance:
            transition = "stable"
        elif final > initial * growth_factor:
            transition = "growing"
        elif final < initial * shrink_factor:
            transition = "shrinking"
        else:
            transition = "oscillating"
        
        volatility = np.std(np.diff(values)) / (np.mean(np.abs(values)) + epsilon)
        
        return {
            'transition': transition,
            'trend': trend,
            'slope': slope,
            'volatility': float(volatility),
            'relative_change': float(relative_change),
        }


# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.5 - Analyse exploratoire générique REFACTORISÉ.

ARCHITECTURE REFACTORISÉE (Phase 2.1) :
- Délégation I/O → data_loading.py
- Délégation filtrage/diagnostics → statistical_utils.py
- Délégation stratification → regime_utils.py
- Cœur métier : analyses statistiques multi-facteurs (variance, interactions)

CORRECTIONS APPLIQUÉES (post-review) :
1. Paires ORIENTÉES : permutations() pas combinations()
2. Interaction = changement d'effet (VR conditionnel >> VR marginal)
3. Filtrage testabilité explicite (min_samples, min_groups)
4. params_config_id exclu interactions (trop corrélé)
5. Drill-down gamma recalculé sur sous-ensemble

RESPONSABILITÉS CONSERVÉES :
- Analyses variance marginale (η²)
- Analyses interactions orientées
- Discrimination métriques
- Corrélations
- Interprétation patterns
- Décision verdict
- Pipeline régime complet
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import warnings
import json

# ============================================================================
# IMPORTS MODULES REFACTORISÉS
# ============================================================================

# I/O et structuration données
from .data_loading import (
    load_all_observations,
    observations_to_dataframe
)

# Filtrage et diagnostics numériques
from .statistical_utils import (
    compute_eta_squared,
    kruskal_wallis_test,
    filter_numeric_artifacts,
    generate_degeneracy_report,
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report
)

# Stratification régimes
from .regime_utils import stratify_by_regime

# Configuration
from .config_loader import get_loader

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Factors expérimentaux analysés
FACTORS = [
    'gamma_id',
    'd_encoding_id',
    'modifier_id',
    'seed',
    'test_name'
    # params_config_id EXCLU : trop corrélé test_name
]

# Projections numériques
PROJECTIONS = [
    'value_final',
    'value_mean',
    'slope',
    'volatility',
    'relative_change'
]

# Seuils testabilité
MIN_SAMPLES_PER_GROUP = 2
MIN_GROUPS = 2
MIN_TOTAL_SAMPLES = 10


# =============================================================================
# ANALYSIS 1 : VARIANCE MARGINALE
# =============================================================================

def analyze_marginal_variance(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str]
) -> pd.DataFrame:
    """
    Analyse variance marginale : chaque facteur pris isolément.
    
    Utilise η² (eta-squared) pour mesurer proportion variance expliquée :
    - η² = SSB / (SSB + SSW)
    - 0 ≤ η² ≤ 1
    - η² proche 1 → facteur très discriminant
    - η² proche 0 → facteur peu informatif
    
    ⚠️ PHASE 2.1 : Correction critique calcul via compute_eta_squared()
    
    Args:
        df: DataFrame observations
        factors: Liste facteurs à analyser
        projections: Liste projections numériques
    
    Returns:
        DataFrame avec colonnes :
        - test_name, metric_name, projection, factor
        - variance_ratio (η²)
        - p_value (Kruskal-Wallis)
        - n_groups, significant
    
    Filtrage testabilité :
        - Contexte global : ≥ MIN_TOTAL_SAMPLES observations
        - Par groupe : ≥ MIN_SAMPLES_PER_GROUP observations
        - Facteur : ≥ MIN_GROUPS niveaux distincts
    """
    results = []
    skipped_testability = 0
    
    for projection in projections:
        for (test_name, metric_name), group in df.groupby(['test_name', 'metric_name']):
            
            # Filtrage testabilité globale
            if len(group) < MIN_TOTAL_SAMPLES:
                skipped_testability += 1
                continue
            
            for factor in factors:
                # Filtrage cardinalité factor
                if group[factor].nunique() < MIN_GROUPS:
                    continue
                
                # Grouper par factor
                factor_groups = [
                    g[projection].dropna().values 
                    for name, g in group.groupby(factor)
                    if len(g.dropna()) >= MIN_SAMPLES_PER_GROUP
                ]
                
                if len(factor_groups) < MIN_GROUPS:
                    continue
                
                # Test Kruskal-Wallis
                try:
                    statistic, p_value = kruskal_wallis_test(factor_groups)
                except (ValueError, Exception):
                    continue
                
                # Calcul η² via fonction utilitaire
                variance_ratio, ssb, ssw = compute_eta_squared(factor_groups)
                
                results.append({
                    'test_name': test_name,
                    'metric_name': metric_name,
                    'projection': projection,
                    'factor': factor,
                    'variance_ratio': variance_ratio,
                    'p_value': p_value,
                    'n_groups': len(factor_groups),
                    'significant': p_value < 0.05
                })
    
    print(f"   ⊘ Skipped {skipped_testability} groups (min_samples)")
    
    return pd.DataFrame(results).sort_values('variance_ratio', ascending=False)


# =============================================================================
# ANALYSIS 2 : INTERACTIONS ORIENTÉES
# =============================================================================

def analyze_oriented_interactions(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str],
    marginal_variance: pd.DataFrame,
    min_interaction_strength: float = 2.0
) -> pd.DataFrame:
    """
    Détecte interactions ORIENTÉES : A|B distinct de B|A.
    
    Interaction vraie détectée si :
    1. VR(A|B=b) significatif (η² > 0.3, p < 0.05)
    2. VR(A|B=b) >> VR(A) marginal (ratio > min_interaction_strength)
    3. Testabilité robuste (≥3 niveaux, ≥5 obs/groupe)
    
    ⚠️ PHASE 2.2 : Critères testabilité renforcés
    - vr_marginal ≥ 0.1 (marginal substantiel)
    - n_groups ≥ 3 (interaction nécessite ≥3 niveaux)
    - min_group_size ≥ 5 (robustesse statistique)
    
    Différence vs combinations() :
    - Paires orientées : (A, B) ET (B, A) testées séparément
    - Total : len(factors) × (len(factors) - 1) paires
    - Interaction orientée : effet de A change selon contexte B
    
    Args:
        df: DataFrame observations
        factors: Liste facteurs
        projections: Projections numériques
        marginal_variance: Résultats analyze_marginal_variance()
        min_interaction_strength: Seuil ratio VR_cond / VR_marg
    
    Returns:
        DataFrame interactions détectées avec colonnes :
        - factor_varying, factor_context, context_value
        - test_name, metric_name, projection
        - vr_conditional, vr_marginal, interaction_strength
        - p_value, n_groups
    """
    results = []
    skipped_testability = 0
    
    # Index variance marginale pour lookup rapide
    marginal_index = {}
    for _, row in marginal_variance.iterrows():
        key = (row['test_name'], row['metric_name'], row['projection'], row['factor'])
        marginal_index[key] = row['variance_ratio']
    
    # PAIRES ORIENTÉES (pas combinations)
    print(f"   Génération {len(factors) * (len(factors) - 1)} paires orientées...")
    
    for factor_varying in factors:
        if factor_varying == 'test_name':
            continue  # test_name n'est jamais un facteur "actif"

        for factor_context in factors:
            if factor_context == factor_varying:
                continue

            for (test_name, metric_name), tm_group in df.groupby(['test_name', 'metric_name']):

                if len(tm_group) < MIN_TOTAL_SAMPLES:
                    continue

                for projection in projections:

                    for context_value, context_group in tm_group.groupby(factor_context):

                        if len(context_group) < MIN_TOTAL_SAMPLES:
                            continue
                        
                        # Variance factor_varying dans ce contexte                        
                        varying_groups = [
                            g[projection].dropna().values
                            for name, g in context_group.groupby(factor_varying)
                            if len(g.dropna()) >= MIN_SAMPLES_PER_GROUP
                        ]
                        
                        if len(varying_groups) < MIN_GROUPS:
                            continue
                        
                        # Test significativité
                        try:
                            statistic, p_value = kruskal_wallis_test(varying_groups)
                        except:
                            continue
                        
                        # Calcul η² conditionnel via fonction utilitaire
                        vr_conditional, ssb, ssw = compute_eta_squared(varying_groups)
                        
                        # Récupérer variance marginale pour comparaison
                        marginal_key = (test_name, metric_name, projection, factor_varying)
                        vr_marginal = marginal_index.get(marginal_key, 0.0)
                        
                        # PHASE 2.2 : Marginal doit être substantiel (≥10%)
                        if vr_marginal < 0.1:
                            continue
                        
                        # Calcul force interaction
                        interaction_strength = vr_conditional / vr_marginal
                        
                        # Interaction = effet change selon contexte
                        # Critères testabilité renforcés (Phase 2.2)
                        min_group_size = min(len(g) for g in varying_groups)
                        
                        if (vr_conditional > 0.3 and 
                            vr_marginal > 0.1 and  # ← Déjà filtré avant, mais explicit
                            p_value < 0.05 and 
                            interaction_strength > min_interaction_strength and
                            len(varying_groups) >= 3 and  # ← Au moins 3 niveaux factor_varying
                            min_group_size >= 5):  # ← Robustesse statistique
                            
                            results.append({
                                'factor_varying': factor_varying,
                                'factor_context': factor_context,
                                'context_value': str(context_value),
                                'test_name': test_name,
                                'metric_name': metric_name,
                                'projection': projection,
                                'vr_conditional': vr_conditional,
                                'vr_marginal': vr_marginal,
                                'interaction_strength': interaction_strength,
                                'p_value': p_value,
                                'n_groups': len(varying_groups)
                            })
    
    print(f"   ⊘ Skipped {skipped_testability} contexts (testability)")
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 3 : DISCRIMINANCE
# =============================================================================

def analyze_metric_discrimination(
    df: pd.DataFrame,
    projections: List[str]
) -> pd.DataFrame:
    """Détecte métriques non discriminantes (CV < 0.1)."""
    results = []
    
    for projection in projections:
        for (test_name, metric_name), group in df.groupby(['test_name', 'metric_name']):
            
            values = group[projection].dropna().values
            
            if len(values) < 5:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if abs(mean_val) < 1e-10:
                cv = np.nan
            else:
                cv = std_val / abs(mean_val)
            
            results.append({
                'test_name': test_name,
                'metric_name': metric_name,
                'projection': projection,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'n_observations': len(values),
                'non_discriminant': cv < 0.1 if not np.isnan(cv) else False
            })
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 4 : CORRÉLATIONS
# =============================================================================

def analyze_metric_correlations(
    df: pd.DataFrame,
    projection: str = 'value_final',
    threshold: float = 0.8
) -> pd.DataFrame:
    """Détecte corrélations fortes entre métriques."""
    results = []
    
    for test_name, test_group in df.groupby('test_name'):
        
        # Pivot
        pivot = test_group.pivot_table(
            index=['gamma_id', 'd_encoding_id', 'modifier_id', 'seed'],
            columns='metric_name',
            values=projection
        )
        
        metrics = pivot.columns.tolist()
        
        if len(metrics) < 2:
            continue
        
        # Corrélations pairwise
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                
                valid = pivot[[metric1, metric2]].dropna()
                
                if len(valid) < 5:
                    continue
                
                try:
                    corr, p_value = spearmanr(valid[metric1], valid[metric2])
                except:
                    continue
                
                if abs(corr) > threshold:
                    results.append({
                        'test_name': test_name,
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_observations': len(valid)
                    })
    
    return pd.DataFrame(results)


# =============================================================================
# INTERPRETATION
# =============================================================================

def interpret_patterns(
    df: pd.DataFrame,
    marginal_variance: pd.DataFrame,
    oriented_interactions: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame
) -> Tuple[dict, dict]:
    """
    Synthèse patterns : GLOBAL + PAR GAMMA.
    
    Returns:
        (patterns_global, patterns_by_gamma)
    """
    patterns_global = {
        'marginal_dominant': [],
        'oriented_interactions': [],
        'non_discriminant': [],
        'redundant': []
    }
    
    # 1. Variance marginale dominante
    dominant = marginal_variance[
        (marginal_variance['variance_ratio'] > 0.5) &
        (marginal_variance['significant'])
    ]
    
    if not dominant.empty:
        for factor in dominant['factor'].unique():
            subset = dominant[dominant['factor'] == factor]
            patterns_global['marginal_dominant'].append({
                'factor': factor,
                'n_metrics': len(subset),
                'projections': subset['projection'].unique().tolist(),
                'max_variance_ratio': float(subset['variance_ratio'].max())
            })
    
    # 2. Interactions orientées (VRAIES)
    if not oriented_interactions.empty:
        # Grouper par paire orientée
        for (fv, fc), group in oriented_interactions.groupby(['factor_varying', 'factor_context']):
            patterns_global['oriented_interactions'].append({
                'interaction': f"{fv} | {fc}",  # Notation orientée
                'n_cases': len(group),
                'contexts_affected': group['context_value'].unique().tolist()[:5],
                'max_strength': float(group['interaction_strength'].max()),
                'examples': group.nlargest(3, 'interaction_strength')[
                    ['test_name', 'metric_name', 'projection', 'context_value', 'interaction_strength']
                ].to_dict('records')
            })
    
    # 3. Non discriminant
    non_disc = discrimination[discrimination['non_discriminant']]
    if not non_disc.empty:
        patterns_global['non_discriminant'].append({
            'n_metrics': len(non_disc),
            'projections': non_disc['projection'].unique().tolist()
        })
    
    # 4. Redondant
    if not correlations.empty:
        patterns_global['redundant'].append({
            'n_pairs': len(correlations)
        })
    
    # =========================================================================
    # PATTERNS PAR GAMMA (drill-down RECALCULÉ)
    # =========================================================================
    
    patterns_by_gamma = {}

    for gamma_id in df['gamma_id'].unique():
    
        # Sous-ensemble STRICT
        gamma_df = df[df['gamma_id'] == gamma_id]
    
        # RECALCUL analyses sur ce gamma uniquement
        gamma_marginal = analyze_marginal_variance(
            gamma_df,
            [f for f in FACTORS if f != 'gamma_id'],  # Exclure gamma
            PROJECTIONS
        )
    
        patterns_gamma = {
            'marginal_dominant': [],
            'oriented_interactions': []
        }
        
        # Dominant dans gamma
        dominant_gamma = gamma_marginal[
            (gamma_marginal['variance_ratio'] > 0.5) &
            (gamma_marginal['significant'])
        ]
        
        if not dominant_gamma.empty:
            for factor in dominant_gamma['factor'].unique():
                subset = dominant_gamma[dominant_gamma['factor'] == factor]
                patterns_gamma['marginal_dominant'].append({
                    'factor': factor,
                    'n_metrics': len(subset)
                })
        
        # Interactions dans gamma (simplifié pour drill-down)
        if not oriented_interactions.empty:
            gamma_interactions = oriented_interactions[
                oriented_interactions['test_name'].isin(gamma_df['test_name'].unique())
            ]
    
            if not gamma_interactions.empty:
                for (fv, fc), group in gamma_interactions.groupby(['factor_varying', 'factor_context']):
                    if fv != 'gamma_id' and fc != 'gamma_id':
                        patterns_gamma['oriented_interactions'].append({
                            'interaction': f"{fv} | {fc}",
                            'n_cases': len(group)
                        })  
        
        patterns_by_gamma[gamma_id] = patterns_gamma
        
    return patterns_global, patterns_by_gamma


# =============================================================================
# VERDICT
# =============================================================================

def decide_verdict(
    patterns_global: dict,
    patterns_by_gamma: dict
) -> Tuple[str, str, dict]:
    """Décision verdict : GLOBAL + PAR GAMMA."""
    
    has_critical = any(len(v) > 0 for v in patterns_global.values())
    
    if not has_critical:
        verdict_global = "SURVIVES[R0]"
        reason_global = "Aucun pattern pathologique systématique détecté."
    else:
        reasons = []
        
        if patterns_global['marginal_dominant']:
            reasons.append(f"{len(patterns_global['marginal_dominant'])} factors dominants")
        
        if patterns_global['oriented_interactions']:
            reasons.append(f"{len(patterns_global['oriented_interactions'])} interactions vraies")
        
        if patterns_global['non_discriminant']:
            p = patterns_global['non_discriminant'][0]
            reasons.append(f"{p['n_metrics']} métriques non discriminantes")
        
        if patterns_global['redundant']:
            p = patterns_global['redundant'][0]
            reasons.append(f"{p['n_pairs']} paires redondantes")
        
        reason_global = " | ".join(reasons)
        verdict_global = "WIP[R0-open]"
    
    # Verdicts par gamma
    verdicts_by_gamma = {}
    
    if patterns_by_gamma:
        for gamma_id, patterns_gamma in patterns_by_gamma.items():
            critical_gamma = sum(len(v) for v in patterns_gamma.values() if v)
    
            if critical_gamma == 0:
                verdict_gamma = "SURVIVES[R0]"
                reason_gamma = "Aucun pattern spécifique."
            else:
                reasons_gamma = []
                if patterns_gamma['marginal_dominant']:
                    reasons_gamma.append(f"{len(patterns_gamma['marginal_dominant'])} factors dominants")
                if patterns_gamma['oriented_interactions']:
                    reasons_gamma.append(f"{len(patterns_gamma['oriented_interactions'])} interactions")
        
                reason_gamma = " | ".join(reasons_gamma)
                verdict_gamma = "WIP[R0-open]"
        
            verdicts_by_gamma[gamma_id] = {
                'verdict': verdict_gamma,
                'reason': reason_gamma,
                'patterns': patterns_gamma
            }
        
    return verdict_global, reason_global, verdicts_by_gamma


# =============================================================================
# PIPELINE RÉGIME
# =============================================================================

def analyze_regime(
    observations: List[dict],
    regime_name: str,
    params_config_id: str,
    verdict_config_id: str
) -> dict:
    """
    Pipeline analyse complet sur une strate.
    
    Identique à pipeline global, retourne structure complète.
    """
    df = observations_to_dataframe(observations)
    
    if len(df) < MIN_TOTAL_SAMPLES:
        return {
            'regime': regime_name,
            'n_observations': len(observations),
            'status': 'INSUFFICIENT_DATA',
            'message': f'Moins de {MIN_TOTAL_SAMPLES} observations'
        }
    
    # Analyses identiques pipeline global
    marginal_variance = analyze_marginal_variance(df, FACTORS, PROJECTIONS)
    oriented_interactions = analyze_oriented_interactions(
        df, FACTORS, PROJECTIONS, marginal_variance
    )
    discrimination = analyze_metric_discrimination(df, PROJECTIONS)
    correlations = analyze_metric_correlations(df)
    
    patterns_global, patterns_by_gamma = interpret_patterns(
        df, marginal_variance, oriented_interactions,
        discrimination, correlations
    )
    
    verdict, reason, verdicts_by_gamma = decide_verdict(
        patterns_global, patterns_by_gamma
    )
    
    return {
        'regime': regime_name,
        'n_observations': len(observations),
        'n_rows_df': len(df),
        'status': 'SUCCESS',
        'marginal_variance': marginal_variance,
        'oriented_interactions': oriented_interactions,
        'discrimination': discrimination,
        'correlations': correlations,
        'patterns_global': patterns_global,
        'patterns_by_gamma': patterns_by_gamma,
        'verdict': verdict,
        'reason': reason,
        'verdicts_by_gamma': verdicts_by_gamma
    }


# =============================================================================
# RAPPORTS STRATIFIÉS
# =============================================================================

def generate_stratified_report(
    params_config_id: str,
    verdict_config_id: str,
    results_global: dict,
    results_stable: dict,
    results_explosif: dict,
    output_dir: str = "reports/verdicts"
) -> None:
    """Génère rapport structuré avec 3 strates."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_stratified_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata commune
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'architecture': 'stratified_parallel_regimes',
        'stratification_threshold': 1e50,
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'regimes': {
            'GLOBAL': {
                'n_observations': results_global['n_observations'],
                'description': 'Baseline complète (toutes observations)'
            },
            'STABLE': {
                'n_observations': results_stable['n_observations'],
                'description': 'Régime non-extrême (|projections| < 1e50)'
            },
            'EXPLOSIF': {
                'n_observations': results_explosif['n_observations'],
                'description': 'Régime magnitude extrême (|projections| >= 1e50)'
            }
        }
    }
    
    with open(report_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Rapport humain structuré
    with open(report_dir / 'summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("VERDICT ANALYSIS - STRATIFICATION PARALLÈLE\n")
        f.write(f"{timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write("  3 analyses parallèles (pipeline identique)\n")
        f.write("  Stratification : |projection| >= 1e50\n")
        f.write("  Aucune donnée filtrée (conservation intégrale)\n\n")
        
        # Résumé par régime
        for regime_name, results in [
            ('GLOBAL', results_global),
            ('STABLE', results_stable),
            ('EXPLOSIF', results_explosif)
        ]:
            f.write("="*80 + "\n")
            f.write(f"RÉGIME {regime_name}\n")
            f.write("="*80 + "\n")
            f.write(f"Observations: {results['n_observations']}\n")
            
            if results['status'] != 'SUCCESS':
                f.write(f"Status: {results['status']}\n")
                f.write(f"Message: {results.get('message', 'N/A')}\n\n")
                continue
            
            f.write(f"Verdict: {results['verdict']}\n")
            f.write(f"Raison:  {results['reason']}\n\n")
            
            f.write("PATTERNS DÉTECTÉS:\n")
            for pattern_type, pattern_list in results['patterns_global'].items():
                if pattern_list:
                    f.write(f"  {pattern_type}: {len(pattern_list)} occurrences\n")
            f.write("\n")
    
    # JSON complet par régime
    for regime_name, results in [
        ('global', results_global),
        ('stable', results_stable),
        ('explosif', results_explosif)
    ]:
        if results['status'] == 'SUCCESS':
            report_json = {
                'regime': results['regime'],
                'n_observations': results['n_observations'],
                'verdict': results['verdict'],
                'reason': results['reason'],
                'patterns_global': results['patterns_global'],
                'patterns_by_gamma': results['patterns_by_gamma']
            }
            
            with open(report_dir / f'analysis_{regime_name}.json', 'w') as f:
                json.dump(report_json, f, indent=2)
    
    # CSVs par régime
    for regime_name, results in [
        ('global', results_global),
        ('stable', results_stable),
        ('explosif', results_explosif)
    ]:
        if results['status'] == 'SUCCESS':
            results['marginal_variance'].to_csv(
                report_dir / f'marginal_variance_{regime_name}.csv',
                index=False
            )
            if not results['oriented_interactions'].empty:
                results['oriented_interactions'].to_csv(
                    report_dir / f'oriented_interactions_{regime_name}.csv',
                    index=False
                )
    
    print(f"\n✓ Rapports stratifiés générés : {report_dir}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None:
    """Pipeline complet verdict exploratoire REFACTORISÉ."""
    
    print(f"\n{'='*70}")
    print(f"VERDICT ANALYSIS - INTERACTIONS ORIENTÉES (REFACTORISÉ)")
    print(f"{'='*70}\n")
    
    n_oriented_pairs = len(FACTORS) * (len(FACTORS) - 1)
    
    print(f"Factors analysés : {', '.join(FACTORS)}")
    print(f"Paires orientées : {n_oriented_pairs}")
    print(f"Projections : {', '.join(PROJECTIONS)}")
    print(f"Testabilité : min_samples={MIN_SAMPLES_PER_GROUP}, min_groups={MIN_GROUPS}\n")
    
    # 1. Config
    print("1. Chargement config...")
    loader = get_loader()
    verdict_config = loader.load('verdict', verdict_config_id)
    
    # 2. Observations (DÉLÉGUÉ → data_loading)
    print("2. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations")
    
    # Filtrage artefacts numériques (DÉLÉGUÉ → statistical_utils)
    observations, rejection_stats = filter_numeric_artifacts(observations)
    
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        print(f"      Détail par test :")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"        {test}: {count} invalides")
    print()
    
    # Diagnostics (DÉLÉGUÉ → statistical_utils)
    print("3. Diagnostics numériques...")
    degeneracy_report = generate_degeneracy_report(observations)
    print_degeneracy_report(degeneracy_report)
    
    scale_report = diagnose_scale_outliers(observations)
    print_scale_outliers_report(scale_report)
    
    # Stratification (DÉLÉGUÉ → regime_utils)
    print("4. Stratification régimes...")
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE    : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF  : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    # Analyses parallèles (CŒUR MÉTIER)
    print("\n5. Analyses statistiques stratifiées...")
    
    print("   [GLOBAL] Baseline complète...")
    results_global = analyze_regime(
        observations, 'GLOBAL', 
        params_config_id, verdict_config_id
    )
    
    print("   [STABLE] Régime non-extrême...")
    results_stable = analyze_regime(
        obs_stable, 'STABLE',
        params_config_id, verdict_config_id
    )
    
    print("   [EXPLOSIF] Régime magnitude extrême...")
    results_explosif = analyze_regime(
        obs_explosif, 'EXPLOSIF',
        params_config_id, verdict_config_id
    )
    
    # Génération rapports stratifiés
    print("\n6. Génération rapports stratifiés...")
    generate_stratified_report(
        params_config_id, verdict_config_id,
        results_global, results_stable, results_explosif
    )
    
    print(f"\n{'='*70}")
    print(f"VERDICT REFACTORISÉ : Analyses complètes sur 3 strates")
    print(f"{'='*70}\n")

# tests/utilities/verdict_reporter.py
"""
Verdict Reporter - Orchestration génération rapports R0.

ARCHITECTURE REFACTORISÉE (Phase 2.3) :
- Délégation I/O → data_loading.py
- Délégation diagnostics → statistical_utils.py
- Délégation stratification → regime_utils.py
- Délégation formatage → report_writers.py
- Cœur métier : orchestration pipeline + compilation résultats

RESPONSABILITÉS CONSERVÉES :
- Orchestration pipeline complet (5 étapes)
- Compilation métadata
- Formatage structures gamma_profiles
- Compilation structural_patterns
- Coordination génération rapports
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================================
# IMPORTS MODULES REFACTORISÉS
# ============================================================================

# I/O et structuration données
from .data_loading import (
    load_all_observations
)

# Filtrage et diagnostics numériques
from .statistical_utils import (
    filter_numeric_artifacts,
    generate_degeneracy_report,
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report
)

# Stratification régimes
from .regime_utils import (
    stratify_by_regime,
    extract_conserved_properties
)

# Analyses verdict et profiling
from .verdict_engine import (
    analyze_regime,
    FACTORS,
    PROJECTIONS,
    MIN_SAMPLES_PER_GROUP,
    MIN_GROUPS,
    MIN_TOTAL_SAMPLES
)

from .gamma_profiling import (
    profile_all_gammas,
    rank_gammas_by_test,
    compare_gammas_summary
)

# Formatage et écriture rapports
from .report_writers import (
    write_json,
    write_header,
    write_regime_synthesis,
    write_dynamic_signatures,
    write_comparisons_enriched,
    write_consultation_footer
)

# Configuration
from .config_loader import get_loader


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    output_dir: str = "reports/verdicts"
) -> dict:
    """
    Pipeline complet génération rapport verdict R0.
    
    ARCHITECTURE REFACTORISÉE :
    1. Chargement observations (data_loading)
    2. Diagnostics numériques (statistical_utils)
    3. Analyses globales stratifiées (verdict_engine)
    4. Profiling gamma (gamma_profiling)
    5. Fusion résultats + génération rapports (report_writers)
    
    Args:
        params_config_id: Config params utilisée (ex: 'params_default_v1')
        verdict_config_id: Config verdict (ex: 'verdict_default_v1')
        output_dir: Répertoire sortie rapports
    
    Returns:
        dict: Résultats complets (pour introspection)
        {
            'metadata': {...},
            'gamma_profiles': {...},
            'structural_patterns': {...},
            'comparisons': {...},
            'report_paths': {...}
        }
    """
    print(f"\n{'='*70}")
    print(f"VERDICT REPORTER R0 - GÉNÉRATION RAPPORTS (REFACTORISÉ)")
    print(f"{'='*70}\n")
    
    print(f"Params config:  {params_config_id}")
    print(f"Verdict config: {verdict_config_id}\n")
    
    # =========================================================================
    # ÉTAPE 1 : CHARGEMENT + DIAGNOSTICS (DÉLÉGUÉ)
    # =========================================================================
    
    print("1. Chargement observations...")
    # ✅ DÉLÉGUÉ → data_loading
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations chargées")
    
    # Filtrage artefacts numériques
    # ✅ DÉLÉGUÉ → statistical_utils
    observations, rejection_stats = filter_numeric_artifacts(observations)
    
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"      {test}: {count} invalides")
    print()
    
    # Diagnostics (informatifs uniquement)
    print("2. Diagnostics numériques...")
    # ✅ DÉLÉGUÉ → statistical_utils
    degeneracy_report = generate_degeneracy_report(observations)
    scale_report = diagnose_scale_outliers(observations)
    print(f"   ✓ Dégénérescences : {degeneracy_report['observations_with_flags']} observations")
    print(f"   ✓ Ruptures échelle : {scale_report['observations_with_outliers']} observations")
    print()
    
    # =========================================================================
    # ÉTAPE 2 : ANALYSES GLOBALES STRATIFIÉES (DÉLÉGUÉ verdict_engine)
    # =========================================================================
    
    print("3. Analyses globales stratifiées...")
    
    # Stratification régimes
    # ✅ DÉLÉGUÉ → regime_utils
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE   : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    # Analyses parallèles (3 strates)
    print("\n   Analyse GLOBAL...")
    # ✅ DÉLÉGUÉ → verdict_engine
    results_global = analyze_regime(
        observations, 'GLOBAL',
        params_config_id, verdict_config_id
    )
    
    print("   Analyse STABLE...")
    if len(obs_stable) > 0:
        results_stable = analyze_regime(
            obs_stable, 'STABLE',
            params_config_id, verdict_config_id
        )
    else:
        results_stable = {
            'regime': 'STABLE',
            'n_observations': 0,
            'status': 'INSUFFICIENT_DATA',
            'message': 'Aucune observation dans strate STABLE'
        }
    
    print("   Analyse EXPLOSIF...")
    if len(obs_explosif) > 0:
        results_explosif = analyze_regime(
            obs_explosif, 'EXPLOSIF',
            params_config_id, verdict_config_id
        )
    else:
        results_explosif = {
            'regime': 'EXPLOSIF',
            'n_observations': 0,
            'status': 'INSUFFICIENT_DATA',
            'message': 'Aucune observation dans strate EXPLOSIF'
        }
    print()
    
    # =========================================================================
    # ÉTAPE 3 : PROFILING GAMMA (DÉLÉGUÉ gamma_profiling)
    # =========================================================================
    
    print("4. Profiling gamma (comportements individuels)...")
    # ✅ DÉLÉGUÉ → gamma_profiling
    gamma_profiles = profile_all_gammas(observations)
    print(f"   ✓ {len(gamma_profiles)} gammas profilés")
    
    # Comparaisons inter-gammas
    comparisons = compare_gammas_summary(gamma_profiles)
    print(f"   ✓ Comparaisons : {len(comparisons['by_test'])} tests analysés")
    print()
    
    # =========================================================================
    # ÉTAPE 4 : FUSION RÉSULTATS (LOCAL - orchestration)
    # =========================================================================
    
    print("5. Fusion résultats...")
    
    # Compiler structure finale
    final_results = {
        'metadata': _compile_metadata(
            params_config_id,
            verdict_config_id,
            observations,
            rejection_stats,
            degeneracy_report,
            scale_report
        ),
        
        'gamma_profiles': _format_gamma_profiles(gamma_profiles),
        
        'structural_patterns': _compile_structural_patterns(
            results_global,
            results_stable,
            results_explosif
        ),
        
        'comparisons': comparisons,
        
        'diagnostics': {
            'numeric_artifacts': rejection_stats,
            'degeneracy': degeneracy_report,
            'scale_outliers': scale_report
        }
    }
    
    print(f"   ✓ Structure compilée")
    print()
    
    # =========================================================================
    # ÉTAPE 5 : GÉNÉRATION RAPPORTS (DÉLÉGUÉ report_writers)
    # =========================================================================
    
    print("6. Génération rapports multi-formats...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_verdict_r0"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # 6a. Metadata
    # ✅ DÉLÉGUÉ → report_writers
    write_json(final_results['metadata'], report_dir / 'metadata.json')
    report_paths['metadata'] = str(report_dir / 'metadata.json')
    
    # 6b. Rapport humain principal
    _write_summary_report(report_dir, final_results)
    report_paths['summary'] = str(report_dir / 'summary.txt')
    
    # 6c. Gamma profiles (JSON + CSV)
    _write_gamma_profiles(report_dir, final_results['gamma_profiles'])
    report_paths['gamma_profiles_json'] = str(report_dir / 'gamma_profiles.json')
    report_paths['gamma_profiles_csv'] = str(report_dir / 'gamma_profiles.csv')
    
    # 6d. Comparaisons inter-gammas
    # ✅ DÉLÉGUÉ → report_writers
    write_json(final_results['comparisons'], report_dir / 'comparisons.json')
    report_paths['comparisons'] = str(report_dir / 'comparisons.json')
    
    # 6e. Structural patterns (analyses globales)
    # ✅ DÉLÉGUÉ → report_writers
    write_json(final_results['structural_patterns'], report_dir / 'structural_patterns.json')
    report_paths['structural_patterns'] = str(report_dir / 'structural_patterns.json')
    
    # 6f. Diagnostics détaillés
    # ✅ DÉLÉGUÉ → report_writers
    write_json(final_results['diagnostics'], report_dir / 'diagnostics.json')
    report_paths['diagnostics'] = str(report_dir / 'diagnostics.json')
    
    # 6g. Analyses stratifiées (CSVs)
    if results_global['status'] == 'SUCCESS':
        results_global['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_global.csv',
            index=False
        )
        report_paths['marginal_variance_global'] = str(report_dir / 'marginal_variance_global.csv')
    
    if results_stable['status'] == 'SUCCESS' and not results_stable['marginal_variance'].empty:
        results_stable['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_stable.csv',
            index=False
        )
        report_paths['marginal_variance_stable'] = str(report_dir / 'marginal_variance_stable.csv')
    
    if results_explosif['status'] == 'SUCCESS' and not results_explosif['marginal_variance'].empty:
        results_explosif['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_explosif.csv',
            index=False
        )
        report_paths['marginal_variance_explosif'] = str(report_dir / 'marginal_variance_explosif.csv')
    
    print(f"   ✓ Rapports générés : {report_dir}")
    print()
    
    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    
    print("="*70)
    print("RAPPORT VERDICT R0 GÉNÉRÉ (REFACTORISÉ)")
    print("="*70)
    print(f"Répertoire : {report_dir}")
    print(f"Gammas     : {len(gamma_profiles)}")
    print(f"Tests      : {len(comparisons['by_test'])}")
    print(f"Fichiers   : {len(report_paths)}")
    print("="*70 + "\n")
    
    final_results['report_paths'] = report_paths
    return final_results


# =============================================================================
# COMPILATION METADATA (LOCAL - orchestration)
# =============================================================================

def _compile_metadata(
    params_config_id: str,
    verdict_config_id: str,
    observations: list,
    rejection_stats: dict,
    degeneracy_report: dict,
    scale_report: dict
) -> dict:
    """Compile métadonnées rapport."""
    
    # Compter gammas/tests uniques
    gammas = set(obs['gamma_id'] for obs in observations)
    tests = set(obs['test_name'] for obs in observations)
    
    return {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'architecture': 'verdict_reporter_r0_refactored',
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'data_summary': {
            'total_observations': len(observations),
            'valid_observations': rejection_stats['valid_observations'],
            'rejected_observations': rejection_stats['rejected_observations'],
            'rejection_rate': rejection_stats['rejection_rate'],
            'n_gammas': len(gammas),
            'n_tests': len(tests),
            'gammas_list': sorted(gammas),
            'tests_list': sorted(tests)
        },
        'quality_flags': {
            'observations_with_degeneracy': degeneracy_report['observations_with_flags'],
            'degeneracy_rate': degeneracy_report['flag_rate'],
            'observations_with_scale_outliers': scale_report['observations_with_outliers'],
            'scale_outlier_rate': scale_report['outlier_rate']
        },
        'analysis_parameters': {
            'factors_analyzed': FACTORS,
            'projections_analyzed': PROJECTIONS,
            'testability_thresholds': {
                'min_samples_per_group': MIN_SAMPLES_PER_GROUP,
                'min_groups': MIN_GROUPS,
                'min_total_samples': MIN_TOTAL_SAMPLES
            }
        }
    }


# =============================================================================
# FORMATAGE GAMMA PROFILES (LOCAL - spécifique verdict)
# =============================================================================

def _format_gamma_profiles(gamma_profiles: dict) -> dict:
    """
    Formate gamma_profiles pour rapport.
    
    Structure Charter R0 :
    {
        'GAM-001': {
            'tests': {
                'SYM-001': {
                    'regime': 'CONSERVES_X',
                    'behavior': 'stable',
                    'timeline': 'early_deviation_then_saturation',
                    'confidence': 'high'
                }
            },
            'summary': {...}
        }
    }
    """
    formatted = {}
    
    for gamma_id, gamma_data in gamma_profiles.items():
        tests_formatted = {}
        
        for test_name, test_profile in gamma_data['tests'].items():
            prc = test_profile['prc_profile']
            
            tests_formatted[test_name] = {
                'regime': prc['regime'],
                'behavior': prc['behavior'],
                'timeline': prc['dominant_timeline']['timeline_compact'],
                'timeline_confidence': prc['dominant_timeline']['confidence'],
                'confidence': prc['confidence'],
                'n_runs': prc['n_runs'],
                'n_valid': prc['n_valid'],
                'pathologies': prc['pathologies'],
                'robustness': prc['robustness']
            }
        
        # Synthèse gamma (régime dominant)
        regime_counts = defaultdict(int)
        for test_prof in tests_formatted.values():
            regime_counts[test_prof['regime']] += 1
        
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'NO_DATA'
        
        formatted[gamma_id] = {
            'tests': tests_formatted,
            'summary': {
                'n_tests': len(tests_formatted),
                'dominant_regime': dominant_regime,
                'regime_distribution': dict(regime_counts)
            }
        }
    
    return formatted


# =============================================================================
# COMPILATION STRUCTURAL PATTERNS (LOCAL - spécifique verdict)
# =============================================================================

def _compile_structural_patterns(
    results_global: dict,
    results_stable: dict,
    results_explosif: dict
) -> dict:
    """Compile patterns structuraux (analyses globales)."""
    
    return {
        'stratification': {
            'GLOBAL': {
                'n_observations': results_global['n_observations'],
                'status': results_global['status'],
                'verdict': results_global.get('verdict', 'N/A'),
                'reason': results_global.get('reason', 'N/A'),
                'patterns': results_global.get('patterns_global', {})
            },
            'STABLE': {
                'n_observations': results_stable['n_observations'],
                'status': results_stable['status'],
                'verdict': results_stable.get('verdict', 'N/A'),
                'reason': results_stable.get('reason', 'N/A'),
                'patterns': results_stable.get('patterns_global', {})
            },
            'EXPLOSIF': {
                'n_observations': results_explosif['n_observations'],
                'status': results_explosif['status'],
                'verdict': results_explosif.get('verdict', 'N/A'),
                'reason': results_explosif.get('reason', 'N/A'),
                'patterns': results_explosif.get('patterns_global', {})
            }
        }
    }


# =============================================================================
# GÉNÉRATION FICHIERS RAPPORTS (PARTIELLEMENT DÉLÉGUÉ)
# =============================================================================

def _write_summary_report(report_dir: Path, results: dict):
    """
    Écrit rapport humain principal (ENRICHI R0+).
    
    ⚠️ PARTIELLEMENT DÉLÉGUÉ : Utilise report_writers pour sections
    """
    
    metadata = results['metadata']
    gamma_profiles = results['gamma_profiles']
    structural = results['structural_patterns']
    comparisons = results['comparisons']
    
    with open(report_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        # ✅ DÉLÉGUÉ → report_writers
        write_header(f, "VERDICT REPORT R0+ - POSTURE NON GAMMA-CENTRIQUE (REFACTORISÉ)")
        
        f.write(f"{metadata['generated_at']}\n\n")
        
        f.write("ARCHITECTURE RAPPORT:\n")
        f.write("  verdict_engine   : Analyses statistiques globales (variance, interactions)\n")
        f.write("  gamma_profiling  : Profils comportementaux individuels (régimes, timelines)\n")
        f.write("  verdict_reporter : Orchestration + génération rapports (REFACTORISÉ)\n")
        f.write("  report_writers   : Formatage sections standardisées\n\n")
        
        # DATA SUMMARY
        f.write("="*80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("="*80 + "\n")
        data = metadata['data_summary']
        f.write(f"Total observations    : {data['total_observations']}\n")
        f.write(f"Valid observations    : {data['valid_observations']}\n")
        f.write(f"Rejected (artifacts)  : {data['rejected_observations']} ({data['rejection_rate']*100:.1f}%)\n")
        f.write(f"Gammas analyzed       : {data['n_gammas']}\n")
        f.write(f"Tests analyzed        : {data['n_tests']}\n\n")
        
        # QUALITY FLAGS
        f.write("QUALITY FLAGS:\n")
        quality = metadata['quality_flags']
        f.write(f"  Dégénérescences détectées : {quality['observations_with_degeneracy']} ({quality['degeneracy_rate']*100:.1f}%)\n")
        f.write(f"  Ruptures échelle          : {quality['observations_with_scale_outliers']} ({quality['scale_outlier_rate']*100:.1f}%)\n\n")
        
        # ✅ DÉLÉGUÉ → report_writers
        write_regime_synthesis(f, gamma_profiles)
        f.write("\n")
        
        # ✅ DÉLÉGUÉ → report_writers
        write_dynamic_signatures(f, gamma_profiles)
        f.write("\n")
        
        # GAMMA PROFILES (résumé par gamma)
        f.write("="*80 + "\n")
        f.write("GAMMA PROFILES (régimes dominants par gamma)\n")
        f.write("="*80 + "\n")
        
        for gamma_id in sorted(gamma_profiles.keys()):
            profile = gamma_profiles[gamma_id]
            summary = profile['summary']
            f.write(f"\n{gamma_id}:\n")
            f.write(f"  Régime dominant : {summary['dominant_regime']}\n")
            f.write(f"  Tests profilés  : {summary['n_tests']}\n")
            f.write(f"  Distribution    : {summary['regime_distribution']}\n")
            
            # ✅ DÉLÉGUÉ → regime_utils
            conserved = extract_conserved_properties(profile)
            if conserved:
                f.write(f"  Propriétés conservées : {', '.join(conserved)}\n")
        
        f.write("\n")
        
        # ✅ DÉLÉGUÉ → report_writers
        write_comparisons_enriched(f, comparisons, gamma_profiles)
        f.write("\n")
        
        # STRUCTURAL PATTERNS (analyses globales)
        f.write("="*80 + "\n")
        f.write("STRUCTURAL PATTERNS (analyses globales stratifiées)\n")
        f.write("="*80 + "\n")
        
        for regime_name in ['GLOBAL', 'STABLE', 'EXPLOSIF']:
            regime_data = structural['stratification'][regime_name]
            f.write(f"\nRÉGIME {regime_name}:\n")
            f.write(f"  Observations : {regime_data['n_observations']}\n")
            f.write(f"  Status       : {regime_data['status']}\n")
            
            if regime_data['status'] == 'SUCCESS':
                f.write(f"  Verdict      : {regime_data['verdict']}\n")
                f.write(f"  Raison       : {regime_data['reason']}\n")
                
                patterns = regime_data['patterns']
                f.write(f"  Patterns détectés:\n")
                for pattern_type, pattern_list in patterns.items():
                    if pattern_list:
                        f.write(f"    {pattern_type}: {len(pattern_list)} occurrences\n")
        
        f.write("\n")
        
        # ✅ DÉLÉGUÉ → report_writers
        write_consultation_footer(f)


def _write_gamma_profiles(report_dir: Path, gamma_profiles: dict):
    """Écrit gamma_profiles.json + CSV."""
    
    # JSON complet
    # ✅ DÉLÉGUÉ → report_writers
    write_json(gamma_profiles, report_dir / 'gamma_profiles.json')
    
    # CSV (vue tabulaire)
    rows = []
    for gamma_id, gamma_data in gamma_profiles.items():
        for test_name, test_data in gamma_data['tests'].items():
            rows.append({
                'gamma_id': gamma_id,
                'test_name': test_name,
                'regime': test_data['regime'],
                'behavior': test_data['behavior'],
                'timeline': test_data['timeline'],
                'timeline_confidence': test_data['timeline_confidence'],
                'confidence': test_data['confidence'],
                'n_runs': test_data['n_runs'],
                'n_valid': test_data['n_valid'],
                'pathology_numeric_instability': test_data['pathologies']['numeric_instability'],
                'pathology_oscillatory': test_data['pathologies']['oscillatory'],
                'pathology_collapse': test_data['pathologies']['collapse'],
                'pathology_trivial': test_data['pathologies']['trivial'],
                'robust_homogeneous': test_data['robustness']['homogeneous'],
                'robust_mixed_behavior': test_data['robustness']['mixed_behavior'],
                'robust_numerically_stable': test_data['robustness']['numerically_stable']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / 'gamma_profiles.csv', index=False)
    
# tests/utilities/registries/algebra_registry.py
"""
Registre des opérations algébriques sur tenseurs.

Fonctions disponibles :
- matrix_asymmetry : Norme partie anti-symétrique
- matrix_norm : Norme tenseur d'ordre N
- trace_value : Trace de matrice
- determinant_value : Déterminant
- frobenius_norm : Norme Frobenius (alias optimisé)
- spectral_norm : Norme spectrale
- condition_number : Conditionnement matrice
- rank_estimate : Estimation rang effectif
"""

import numpy as np
from .base_registry import BaseRegistry, register_function


class AlgebraRegistry(BaseRegistry):
    """Registre des opérations algébriques sur tenseurs."""
    
    REGISTRY_KEY = "algebra"
    
    @register_function("matrix_asymmetry")
    def compute_asymmetry(
        self,
        state: np.ndarray,
        norm_type: str = 'frobenius',
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calcule la norme de la partie anti-symétrique.
        
        Args:
            state: Tenseur rang 2 (matrice carrée)
            norm_type: 'frobenius' | 'spectral' | 'nuclear'
            normalize: Normaliser par norme totale ?
            epsilon: Protection division par zéro
        
        Returns:
            float: Norme de (state - state.T) / 2
        
        Raises:
            ValueError: Si state non 2D ou non carré
        
        Notes:
            - Valeurs élevées indiquent forte asymétrie
            - normalize=True permet comparaison entre matrices tailles différentes
            - Symétrie parfaite → 0.0
            - normalize=True → valeurs dans [0, 1]
        
        Examples:
            >>> state = np.array([[1, 2], [2, 1]])  # Symétrique
            >>> compute_asymmetry(state)
            0.0
            
            >>> state = np.array([[1, 2], [3, 1]])  # Asymétrique
            >>> compute_asymmetry(state) > 0
            True
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        asymmetry = state - state.T
        
        if norm_type == 'frobenius':
            norm = np.linalg.norm(asymmetry, 'fro')
        elif norm_type == 'spectral':
            norm = np.linalg.norm(asymmetry, 2)
        elif norm_type == 'nuclear':
            norm = np.sum(np.linalg.svd(asymmetry, compute_uv=False))
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")
        
        if normalize:
            total_norm = np.linalg.norm(state, 'fro') + epsilon
            norm = norm / total_norm
        
        return float(norm)
    
    @register_function("matrix_norm")
    def compute_norm(
        self,
        state: np.ndarray,
        norm_type: str = 'frobenius'
    ) -> float:
        """
        Norme de tenseur d'ordre quelconque.
        
        Args:
            state: Tenseur de rang N
            norm_type: 'frobenius' | 'spectral' | 'nuclear' | int
        
        Returns:
            float: Norme du tenseur
        
        Raises:
            ValueError: Si norm_type incompatible avec rang
        
        Notes:
            - Frobenius : Généralisable à tout rang, O(n²)
            - Spectral : Matrice 2D uniquement, O(n³)
            - Nuclear : Matrice 2D uniquement, O(n³)
            - int : Norme Lp du vecteur aplati
        
        Examples:
            >>> state = np.ones((10, 10))
            >>> compute_norm(state, 'frobenius')
            10.0
        """
        if norm_type == 'frobenius':
            return float(np.linalg.norm(state, 'fro'))
        
        elif norm_type == 'spectral':
            if state.ndim != 2:
                raise ValueError("Norme spectrale requiert matrice 2D")
            return float(np.linalg.norm(state, 2))
        
        elif norm_type == 'nuclear':
            if state.ndim != 2:
                raise ValueError("Norme nucléaire requiert matrice 2D")
            return float(np.sum(np.linalg.svd(state, compute_uv=False)))
        
        elif isinstance(norm_type, int):
            return float(np.linalg.norm(state.flatten(), norm_type))
        
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")
    
    @register_function("frobenius_norm")
    def compute_frobenius(
        self,
        state: np.ndarray
    ) -> float:
        """
        Norme Frobenius (alias optimisé).
        
        Args:
            state: Tenseur de rang N
        
        Returns:
            float: ||state||_F = sqrt(sum(state**2))
        
        Notes:
            - Généralisation norme euclidienne
            - O(n²) pour matrices n×n
            - Toujours >= 0
        """
        return float(np.linalg.norm(state, 'fro'))
    
    @register_function("trace_value")
    def compute_trace(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Trace de matrice.
        
        Args:
            state: Matrice carrée 2D
            normalize: Diviser par dimension ?
            epsilon: Protection si normalize=True
        
        Returns:
            float: Trace ou trace normalisée
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Trace = somme valeurs propres
            - Trace(AB) = Trace(BA)
            - normalize=True → moyenne diagonale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        trace = np.trace(state)
        
        if normalize:
            trace = trace / (state.shape[0] + epsilon)
        
        return float(trace)
    
    @register_function("determinant_value")
    def compute_determinant(
        self,
        state: np.ndarray,
        log_scale: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Déterminant de matrice.
        
        Args:
            state: Matrice carrée 2D
            log_scale: Retourner log|det| ?
            epsilon: Protection log si det proche de 0
        
        Returns:
            float: det(state) ou log|det(state)|
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Déterminant = produit valeurs propres
            - det = 0 → matrice singulière
            - log_scale utile si det très grand/petit
            - log_scale=True → log(|det| + epsilon)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        det = np.linalg.det(state)
        
        if log_scale:
            return float(np.log(np.abs(det) + epsilon))
        
        return float(det)
    
    @register_function("spectral_norm")
    def compute_spectral_norm(
        self,
        state: np.ndarray
    ) -> float:
        """
        Norme spectrale (plus grande valeur singulière).
        
        Args:
            state: Matrice 2D
        
        Returns:
            float: σ_max (plus grande valeur singulière)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Norme spectrale = ||A||_2
            - σ_max = sqrt(plus grande valeur propre de A^T A)
            - Mesure d'amplification maximale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        return float(np.linalg.norm(state, 2))
    
    @register_function("condition_number")
    def compute_condition(
        self,
        state: np.ndarray,
        norm_type: str = 'spectral',
        epsilon: float = 1e-10
    ) -> float:
        """
        Conditionnement de matrice.
        
        Args:
            state: Matrice carrée 2D
            norm_type: 'spectral' | 'frobenius'
            epsilon: Protection division par zéro
        
        Returns:
            float: κ(A) = ||A|| * ||A⁻¹||
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - κ ≥ 1 toujours
            - κ proche de 1 → bien conditionnée
            - κ >> 1 → mal conditionnée
            - κ = ∞ → singulière
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        if norm_type == 'spectral':
            p = 2
        elif norm_type == 'frobenius':
            p = 'fro'
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")
        
        try:
            cond = np.linalg.cond(state, p=p)
            if np.isinf(cond) or np.isnan(cond):
                return float(1e10)  # Valeur sentinelle
            return float(cond)
        except np.linalg.LinAlgError:
            return float(1e10)
    
    @register_function("rank_estimate")
    def compute_rank(
        self,
        state: np.ndarray,
        tolerance: float = 1e-10
    ) -> float:
        """
        Estimation rang effectif via SVD.
        
        Args:
            state: Matrice 2D
            tolerance: Seuil valeur singulière (σ > tolerance)
        
        Returns:
            float: Nombre de valeurs singulières > tolerance
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Rang numérique vs rang théorique
            - tolerance détermine sensibilité
            - Rang max = min(n, m) pour matrice n×m
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        singular_values = np.linalg.svd(state, compute_uv=False)
        rank = np.sum(singular_values > tolerance)
        
        return float(rank)
        
# prc_framework/tests/utilities/registries/base_registry.py

from abc import ABC
from typing import Dict, Callable

class BaseRegistry(ABC):
    """
    Classe abstraite pour tous les registres.
    
    Contraintes :
    - REGISTRY_KEY doit être défini et unique
    - Fonctions décorées avec @register_function
    - Signature: func(state, **params) -> float
    """
    
    REGISTRY_KEY: str = None  # À définir dans sous-classe
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._register_all_functions()
    
    def _register_all_functions(self):
        """Découvre et enregistre toutes les fonctions décorées."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_registry_metadata'):
                function_name = attr._registry_metadata['name']
                self._functions[function_name] = attr
    
    def get_function(self, registry_key: str) -> Callable:
        """
        Récupère une fonction par sa clé complète.
        
        Args:
            registry_key: Format "registre.fonction"
        
        Returns:
            Callable
        
        Raises:
            KeyError: Si fonction non trouvée
        """
        _, function_name = registry_key.split('.', 1)
        
        if function_name not in self._functions:
            available = list(self._functions.keys())
            raise KeyError(
                f"Fonction '{function_name}' non trouvée dans {self.REGISTRY_KEY}. "
                f"Disponibles: {available}"
            )
        
        return self._functions[function_name]
    
    def list_functions(self) -> Dict[str, str]:
        """Liste toutes les fonctions avec leur documentation."""
        return {
            name: func.__doc__ or "Pas de documentation"
            for name, func in self._functions.items()
        }


def register_function(name: str):
    """
    Décorateur pour enregistrer une fonction dans un registre.
    
    Usage:
        @register_function("matrix_norm")
        def compute_norm(state, norm_type='fro'):
            return np.linalg.norm(state, norm_type)
    """
    def decorator(func):
        func._registry_metadata = {'name': name}
        return func
    return decorator
    
# tests/utilities/registries/graph_registry.py
"""
Registre des métriques graphes (interprétation matrice comme adjacence).

Fonctions disponibles :
- density : Densité graphe
- degree_variance : Variance degrés
- clustering_local : Clustering local moyen
- average_path_length : Longueur chemin moyenne (approximation)
- centrality_concentration : Concentration centralité
- small_world_coefficient : Coefficient petit-monde
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .base_registry import BaseRegistry, register_function


class GraphRegistry(BaseRegistry):
    """Registre des métriques graphes."""
    
    REGISTRY_KEY = "graph"
    
    @register_function("density")
    def compute_density(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Densité du graphe.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes (|w| > threshold)
            epsilon: Protection division
        
        Returns:
            float: Densité = E / E_max dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - E = nombre arêtes
            - E_max = n(n-1)/2 (graphe complet)
            - 1.0 → graphe complet
            - 0.0 → pas d'arêtes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        n = state.shape[0]
        
        # Compter arêtes
        adjacency = np.abs(state) > threshold
        np.fill_diagonal(adjacency, False)  # Pas de boucles
        
        num_edges = np.sum(adjacency) / 2  # Non-orienté
        
        # Max arêtes (graphe complet)
        max_edges = n * (n - 1) / 2
        
        if max_edges < epsilon:
            return 0.0
        
        density = num_edges / max_edges
        
        return float(np.clip(density, 0.0, 1.0))
    
    @register_function("degree_variance")
    def compute_degree_variance(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance des degrés.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            normalize: Normaliser par degré moyen² ?
            epsilon: Protection division
        
        Returns:
            float: Variance degrés ou CV²
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure hétérogénéité degrés
            - Élevé → hubs et nœuds isolés
            - Faible → degrés homogènes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Matrice adjacence
        adjacency = np.abs(state) > threshold
        np.fill_diagonal(adjacency, False)
        
        # Degrés
        degrees = np.sum(adjacency, axis=1)
        
        # Variance
        var = float(np.var(degrees))
        
        if normalize:
            mean_degree = np.mean(degrees) + epsilon
            var = var / (mean_degree ** 2)
        
        return var
    
    @register_function("clustering_local")
    def compute_clustering_local(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient clustering local moyen.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Clustering moyen dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure transitivité locale
            - 1.0 → tous triangles fermés
            - 0.0 → pas de triangles
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Matrice adjacence binaire
        adjacency = (np.abs(state) > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        n = adjacency.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            # Voisins de i
            neighbors = np.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Arêtes entre voisins
            subgraph = adjacency[np.ix_(neighbors, neighbors)]
            num_edges = np.sum(subgraph) / 2
            
            # Clustering local
            max_edges = k * (k - 1) / 2
            if max_edges > epsilon:
                clustering = num_edges / max_edges
                clustering_coeffs.append(clustering)
        
        if not clustering_coeffs:
            return 0.0
        
        return float(np.mean(clustering_coeffs))
    
    @register_function("average_path_length")
    def compute_average_path_length(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        sample_fraction: float = 0.1,
        epsilon: float = 1e-10
    ) -> float:
        """
        Longueur chemin moyenne (approximation échantillonnée).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            sample_fraction: Fraction nœuds échantillonnés
            epsilon: Protection calculs
        
        Returns:
            float: Longueur chemin moyenne
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure compacité graphe
            - Petit → graphe compact
            - Grand → graphe étendu
            - Échantillonné pour performance O(n²) → O(n)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Adjacence
        adjacency = (np.abs(state) > threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        
        # Convertir en distances (1 / poids)
        distances = adjacency.copy()
        distances[distances == 0] = np.inf
        
        # Échantillonner nœuds
        n = adjacency.shape[0]
        n_sample = max(1, int(n * sample_fraction))
        sample_nodes = np.random.choice(n, size=n_sample, replace=False)
        
        # Chemins plus courts (échantillonnés)
        try:
            dist_matrix = shortest_path(
                csr_matrix(distances),
                directed=False,
                indices=sample_nodes
            )
            
            # Moyenne chemins finis
            finite_dists = dist_matrix[np.isfinite(dist_matrix)]
            
            if len(finite_dists) == 0:
                return float('inf')
            
            avg_path = float(np.mean(finite_dists))
        
        except:
            return float('inf')
        
        return avg_path
    
    @register_function("centrality_concentration")
    def compute_centrality_concentration(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Concentration de centralité (degré).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Concentration dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - 1.0 → centralité concentrée (graphe étoile)
            - 0.0 → centralité distribuée
            - Basé sur variance degrés normalisée
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Adjacence
        adjacency = (np.abs(state) > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        # Degrés
        degrees = np.sum(adjacency, axis=1)
        
        if np.sum(degrees) == 0:
            return 0.0
        
        n = len(degrees)
        max_degree = np.max(degrees)
        
        # Concentration (Freeman)
        numerator = np.sum(max_degree - degrees)
        denominator = (n - 1) * (n - 2) + epsilon
        
        concentration = numerator / denominator
        
        return float(np.clip(concentration, 0.0, 1.0))
    
    @register_function("small_world_coefficient")
    def compute_small_world(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient petit-monde (σ = (C/C_rand) / (L/L_rand)).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Coefficient petit-monde
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - σ >> 1 : Propriété petit-monde
            - σ ≈ 1 : Graphe aléatoire
            - σ < 1 : Réseau régulier
            - Approximation via comparaison clustering/path
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Clustering observé
        C = self.compute_clustering_local(state, threshold)
        
        # Path length observé (approximation)
        L = self.compute_average_path_length(state, threshold, sample_fraction=0.1)
        
        if not np.isfinite(L):
            return 1.0
        
        # Densité
        density = self.compute_density(state, threshold)
        
        # Clustering random (approximation)
        C_rand = density + epsilon
        
        # Path random (approximation log(n)/log(k))
        n = state.shape[0]
        k = density * (n - 1)
        if k > 1:
            L_rand = np.log(n) / np.log(k)
        else:
            L_rand = n
        
        # Coefficient
        if L_rand < epsilon or C_rand < epsilon:
            return 1.0
        
        sigma = (C / C_rand) / (L / L_rand + epsilon)
        
        return float(sigma)
        
# tests/utilities/registries/pattern_registry.py
"""
Registre de détection de patterns.

Fonctions disponibles :
- periodicity : Détection périodicité
- symmetry_score : Score symétrie radiale
- clustering_coefficient : Coefficient clustering
- diversity : Diversité valeurs (indice Simpson)
- uniformity : Uniformité distribution
- concentration_ratio : Ratio concentration (Gini)
"""

import numpy as np
from scipy import signal
from .base_registry import BaseRegistry, register_function


class PatternRegistry(BaseRegistry):
    """Registre de détection de patterns."""
    
    REGISTRY_KEY = "pattern"
    
    @register_function("periodicity")
    def compute_periodicity(
        self,
        state: np.ndarray,
        axis: int = 0,
        method: str = 'autocorr'
    ) -> float:
        """
        Détection de périodicité.
        
        Args:
            state: Tenseur rang N
            axis: Axe analyse (si multi-dim)
            method: 'autocorr' | 'fft'
        
        Returns:
            float: Score périodicité dans [0, 1]
        
        Notes:
            - 1.0 → signal parfaitement périodique
            - 0.0 → signal aléatoire
            - autocorr : Basé sur pics autocorrélation
            - fft : Basé sur pics spectre fréquence
        """
        # Prendre slice si multi-dim
        if state.ndim > 1:
            signal_1d = np.mean(state, axis=tuple(i for i in range(state.ndim) if i != axis))
        else:
            signal_1d = state
        
        if method == 'autocorr':
            # Autocorrélation
            autocorr = np.correlate(signal_1d, signal_1d, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normaliser
            
            # Chercher pics
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            
            if len(peaks) > 0:
                # Périodicité = hauteur pic max
                periodicity = float(np.max(autocorr[peaks + 1]))
            else:
                periodicity = 0.0
        
        elif method == 'fft':
            # FFT
            fft = np.fft.fft(signal_1d)
            power = np.abs(fft[:len(fft)//2])**2
            power = power / np.sum(power)
            
            # Périodicité = concentration énergie
            periodicity = float(np.max(power))
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        return np.clip(periodicity, 0.0, 1.0)
    
    @register_function("symmetry_score")
    def compute_symmetry_score(
        self,
        state: np.ndarray,
        symmetry_type: str = 'reflection',
        axis: int = 0
    ) -> float:
        """
        Score symétrie.
        
        Args:
            state: Matrice 2D
            symmetry_type: 'reflection' | 'rotation'
            axis: Axe réflexion (0 ou 1) si reflection
        
        Returns:
            float: Score symétrie dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → parfaitement symétrique
            - 0.0 → totalement asymétrique
            - reflection : Symétrie miroir
            - rotation : Symétrie rotation 180°
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if symmetry_type == 'reflection':
            # Réflexion selon axe
            if axis == 0:
                reflected = np.flipud(state)
            else:
                reflected = np.fliplr(state)
            
            # Corrélation avec réfléchi
            corr = np.corrcoef(state.flatten(), reflected.flatten())[0, 1]
        
        elif symmetry_type == 'rotation':
            # Rotation 180°
            rotated = np.rot90(state, k=2)
            
            # Corrélation avec tourné
            corr = np.corrcoef(state.flatten(), rotated.flatten())[0, 1]
        
        else:
            raise ValueError(f"Type symétrie inconnu: {symmetry_type}")
        
        # Convertir corrélation [-1, 1] en score [0, 1]
        score = (corr + 1.0) / 2.0
        
        if np.isnan(score):
            score = 0.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    @register_function("clustering_coefficient")
    def compute_clustering(
        self,
        state: np.ndarray,
        threshold: float = 0.5,
        normalize: bool = True
    ) -> float:
        """
        Coefficient clustering (valeurs similaires groupées).
        
        Args:
            state: Matrice 2D
            threshold: Seuil similarité (percentile si normalize)
            normalize: Interpréter threshold comme percentile ?
        
        Returns:
            float: Score clustering dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → valeurs similaires fortement groupées
            - 0.0 → distribution homogène
            - Basé sur corrélation pixels voisins
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if normalize:
            threshold = np.percentile(np.abs(state), threshold * 100)
        
        # Masque valeurs élevées
        mask = np.abs(state) > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Compter voisins similaires
        neighbors = 0
        similar_neighbors = 0
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if mask[i, j]:
                    # 4-voisinage
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < state.shape[0] and 0 <= nj < state.shape[1]:
                            neighbors += 1
                            if mask[ni, nj]:
                                similar_neighbors += 1
        
        if neighbors == 0:
            return 0.0
        
        clustering = similar_neighbors / neighbors
        
        return float(clustering)
    
    @register_function("diversity")
    def compute_diversity(
        self,
        state: np.ndarray,
        bins: int = 50,
        epsilon: float = 1e-10
    ) -> float:
        """
        Diversité (indice Simpson).
        
        Args:
            state: Tenseur rang N
            bins: Nombre bins histogramme
            epsilon: Protection division
        
        Returns:
            float: Indice Simpson dans [0, 1]
        
        Notes:
            - 1.0 → distribution uniforme (haute diversité)
            - 0.0 → concentration (basse diversité)
            - D = 1 - Σ p_i²
        """
        # Histogramme
        values = state.flatten()
        hist, _ = np.histogram(values, bins=bins)
        
        # Normaliser
        probs = hist / (np.sum(hist) + epsilon)
        
        # Indice Simpson
        simpson = 1.0 - np.sum(probs ** 2)
        
        return float(simpson)
    
    @register_function("uniformity")
    def compute_uniformity(
        self,
        state: np.ndarray,
        bins: int = 50,
        epsilon: float = 1e-10
    ) -> float:
        """
        Uniformité distribution.
        
        Args:
            state: Tenseur rang N
            bins: Nombre bins histogramme
            epsilon: Protection division
        
        Returns:
            float: Score uniformité dans [0, 1]
        
        Notes:
            - 1.0 → distribution parfaitement uniforme
            - 0.0 → distribution très concentrée
            - Basé sur distance à distribution uniforme
        """
        # Histogramme
        values = state.flatten()
        hist, _ = np.histogram(values, bins=bins)
        
        # Normaliser
        probs = hist / (np.sum(hist) + epsilon)
        
        # Distribution uniforme cible
        uniform = np.ones(bins) / bins
        
        # Distance chi-carré
        chi_sq = np.sum((probs - uniform) ** 2 / (uniform + epsilon))
        
        # Convertir en score [0, 1]
        # chi_sq max ≈ bins (tous points dans 1 bin)
        uniformity = 1.0 - np.clip(chi_sq / bins, 0.0, 1.0)
        
        return float(uniformity)
    
    @register_function("concentration_ratio")
    def compute_concentration(
        self,
        state: np.ndarray,
        top_percent: float = 0.1,
        epsilon: float = 1e-10
    ) -> float:
        """
        Ratio concentration (Gini-like).
        
        Args:
            state: Tenseur rang N
            top_percent: Fraction valeurs top (ex: 0.1 = top 10%)
            epsilon: Protection division
        
        Returns:
            float: Fraction énergie dans top_percent valeurs
        
        Notes:
            - Mesure concentration distribution
            - 1.0 → toute énergie dans quelques valeurs
            - ~top_percent → distribution uniforme
            - Utile détecter activations parcimonieuses
        """
        values = np.abs(state.flatten())
        
        # Trier décroissant
        sorted_values = np.sort(values)[::-1]
        
        # Nombre top valeurs
        n_top = int(len(sorted_values) * top_percent)
        if n_top == 0:
            n_top = 1
        
        # Énergie top vs total
        energy_top = np.sum(sorted_values[:n_top] ** 2)
        energy_total = np.sum(sorted_values ** 2) + epsilon
        
        concentration = energy_top / energy_total
        
        return float(concentration)
        
# prc_framework/tests/utilities/registries/post_processors.py

import numpy as np
from typing import Callable, Dict

POST_PROCESSORS: Dict[str, Callable] = {
    # Identity
    'identity': lambda x: x,
    
    # Rounding
    'round_2': lambda x: round(float(x), 2),
    'round_4': lambda x: round(float(x), 4),
    'round_6': lambda x: round(float(x), 6),
    
    # Absolute value
    'abs': lambda x: abs(float(x)),
    
    # Logarithmic
    'log': lambda x: float(np.log(x + 1e-10)),
    'log10': lambda x: float(np.log10(x + 1e-10)),
    'log1p': lambda x: float(np.log1p(x)),
    
    # Clipping
    'clip_01': lambda x: float(np.clip(x, 0, 1)),
    'clip_positive': lambda x: float(max(0, x)),
    
    # Scientific notation
    'scientific_3': lambda x: float(f"{x:.3e}"),
}


def get_post_processor(key: str) -> Callable:
    """
    Récupère un post-processor par clé.
    
    Args:
        key: Identifiant du post-processor
    
    Returns:
        Callable: Fonction de transformation (float -> float)
    
    Raises:
        KeyError: Si clé inconnue
    """
    if key not in POST_PROCESSORS:
        available = list(POST_PROCESSORS.keys())
        raise KeyError(
            f"Post-processor '{key}' inconnu. "
            f"Disponibles: {available}"
        )
    
    return POST_PROCESSORS[key]


def add_post_processor(key: str, func: Callable) -> None:
    """
    Ajoute un post-processor custom.
    
    Args:
        key: Identifiant unique
        func: Fonction (float -> float)
    
    Raises:
        ValueError: Si clé déjà existante
    """
    if key in POST_PROCESSORS:
        raise ValueError(f"Post-processor '{key}' déjà existant")
    
    POST_PROCESSORS[key] = func
    
# prc_framework/utilities/registries/registry_manager.py

import importlib
import pkgutil
import inspect
from pathlib import Path
from typing import Dict, Any, Callable
import numpy as np

from .base_registry import BaseRegistry
from .post_processors import get_post_processor

class RegistryManager:
    """
    Singleton gérant tous les registres.
    
    Responsabilités :
    1. Charger dynamiquement tous les *_registry.py
    2. Valider registry_key
    3. Fournir fonctions avec paramètres validés
    4. Résoudre post_processors
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registries: Dict[str, BaseRegistry] = {}
        self.function_cache: Dict[str, Callable] = {}
        
        self._load_all_registries()
        self._initialized = True
    
    def _load_all_registries(self) -> None:
        """Charge dynamiquement tous les registres."""
        package_name = "tests.utilities.registries"
        
        try:
            package = importlib.import_module(package_name)
            package_path = Path(package.__file__).parent
            
            for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
                if not is_pkg and module_name.endswith('_registry'):
                    full_name = f"{package_name}.{module_name}"
                    
                    try:
                        module = importlib.import_module(full_name)
                        
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            
                            if (isinstance(attr, type) and 
                                issubclass(attr, BaseRegistry) and 
                                attr != BaseRegistry):
                                
                                instance = attr()
                                
                                if not instance.REGISTRY_KEY:
                                    raise ValueError(f"{attr_name} sans REGISTRY_KEY")
                                
                                if instance.REGISTRY_KEY in self.registries:
                                    raise ValueError(f"REGISTRY_KEY dupliqué: {instance.REGISTRY_KEY}")
                                
                                self.registries[instance.REGISTRY_KEY] = instance
                                print(f"[RegistryManager] Chargé: {instance.REGISTRY_KEY}")
                    
                    except Exception as e:
                        print(f"[RegistryManager] Erreur {module_name}: {e}")
                        continue
        
        except Exception as e:
            print(f"[RegistryManager] Erreur initialisation: {e}")
            raise

			
    def get_function(self, registry_key: str) -> Callable:
        """
        Récupère fonction par clé complète.
        
        Args:
            registry_key: Format "registre.fonction"
        
        Returns:
            Callable
        
        Raises:
            KeyError: Si registre ou fonction non trouvée
        """
        if registry_key in self.function_cache:
            return self.function_cache[registry_key]
        
        if '.' not in registry_key:
            raise KeyError(f"Format invalide: {registry_key}. Attendu: registre.fonction")
        
        registry_name, function_name = registry_key.split('.', 1)
        
        if registry_name not in self.registries:
            available = list(self.registries.keys())
            raise KeyError(f"Registre '{registry_name}' inconnu. Disponibles: {available}")
        
        registry = self.registries[registry_name]
        function = registry.get_function(registry_key)
        
        self.function_cache[registry_key] = function
        
        return function
    
    def validate_computation_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et prépare une spécification de calcul.
        
        Args:
            spec: {
                'registry_key': str,
                'default_params': dict,
                'post_process': str (optionnel),
                'validation': dict (optionnel),
            }
        
        Returns:
            {
                'function': Callable,
                'params': dict (validés),
                'post_process': Callable | None,
                'registry_key': str,
            }
        
        Raises:
            ValueError: Si validation échoue
        """
        # Clés obligatoires
        if 'registry_key' not in spec:
            raise ValueError("Manque 'registry_key'")
        
        if 'default_params' not in spec:
            raise ValueError("Manque 'default_params'")
        
        registry_key = spec['registry_key']
        
        # 1. Récupérer fonction
        function = self.get_function(registry_key)
        
        # 2. Valider paramètres
        validated_params = self._validate_params(function, spec['default_params'])
        
        # 3. Résoudre post_process
        post_process = None
        if 'post_process' in spec and spec['post_process']:
            post_process = get_post_processor(spec['post_process'])
        
        return {
            'function': function,
            'params': validated_params,
            'post_process': post_process,
            'registry_key': registry_key,
        }
    
    def _validate_params(self, function: Callable, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide paramètres contre signature fonction.
        
        Args:
            function: Fonction du registre (méthode bound)
            user_params: Paramètres fournis
        
        Returns:
            Dict paramètres validés et complétés
        
        Raises:
            ValueError: Si paramètre invalide
        """
        sig = inspect.signature(function)
        parameters = sig.parameters
        
        # Debug : afficher paramètres détectés
        param_names = list(parameters.keys())
        
        # Vérifier que 'state' est présent (premier ou après self si bound)
        if 'state' not in param_names:
            raise ValueError(
                f"Fonction doit avoir paramètre 'state'. "
                f"Paramètres détectés : {param_names}"
            )
        
        # Vérifier que state est bien le premier (après self si présent)
        # Note: pour méthode bound, self a disparu de la signature
        first_param = param_names[0]
        if first_param != 'state':
            raise ValueError(
                f"Paramètre 'state' doit être en premier. "
                f"Trouvé : {first_param} en premier. "
                f"Signature complète : {param_names}"
            )
        
        # Valeurs par défaut de la fonction (exclure state)
        defaults = {
            name: param.default
            for name, param in parameters.items()
            if param.default is not inspect.Parameter.empty and name != 'state'
        }
        
        validated = {}
        
        for param_name, param_value in user_params.items():
            if param_name not in parameters:
                raise ValueError(
                    f"Paramètre '{param_name}' invalide. "
                    f"Attendus : {[p for p in param_names if p != 'state']}"
                )
            
            expected_param = parameters[param_name]
            
            # Validation type (basique)
            if expected_param.annotation is not inspect.Parameter.empty:
                expected_type = expected_param.annotation
                
                # Skip validation si type générique (typing.*)
                if hasattr(expected_type, '__origin__'):
                    validated[param_name] = param_value
                    continue
                
                if not isinstance(param_value, expected_type):
                    try:
                        param_value = expected_type(param_value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Paramètre '{param_name}' attend {expected_type.__name__}, "
                            f"reçu {type(param_value).__name__}"
                        )
            
            validated[param_name] = param_value
        
        # Ajouter defaults manquants
        for param_name, default_value in defaults.items():
            if param_name not in validated:
                validated[param_name] = default_value
        
        return validated
    
    def list_available_functions(self) -> Dict[str, list]:
        """Liste toutes les fonctions par registre."""
        return {
            registry_name: list(registry.list_functions().keys())
            for registry_name, registry in self.registries.items()
        }
        
# tests/utilities/registries/spatial_registry.py
"""
Registre des analyses spatiales.

Fonctions disponibles :
- gradient_magnitude : Magnitude gradient moyen
- laplacian_energy : Énergie laplacien
- local_variance : Variance locale moyenne
- edge_density : Densité contours
- spatial_autocorrelation : Autocorrélation spatiale
- smoothness : Mesure lissage
"""

import numpy as np
from scipy import ndimage
from .base_registry import BaseRegistry, register_function


class SpatialRegistry(BaseRegistry):
    """Registre des analyses spatiales."""
    
    REGISTRY_KEY = "spatial"
    
    @register_function("gradient_magnitude")
    def compute_gradient_magnitude(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Magnitude moyenne du gradient.
        
        Args:
            state: Tenseur rang 2 ou 3
            normalize: Normaliser par amplitude état ?
            epsilon: Protection division
        
        Returns:
            float: Moyenne ||∇state||
        
        Raises:
            ValueError: Si rang < 2 ou > 3
        
        Notes:
            - Mesure variation spatiale
            - Élevé → contours nets, transitions brusques
            - Faible → champ lisse, homogène
        """
        if state.ndim not in [2, 3]:
            raise ValueError(f"Attendu 2D ou 3D, reçu {state.ndim}D")
        
        # Gradient selon chaque axe
        gradients = np.gradient(state)
        
        # Magnitude
        magnitude = np.sqrt(sum(g**2 for g in gradients))
        mean_magnitude = float(np.mean(magnitude))
        
        if normalize:
            amplitude = np.max(np.abs(state)) - np.min(np.abs(state))
            mean_magnitude = mean_magnitude / (amplitude + epsilon)
        
        return mean_magnitude
    
    @register_function("laplacian_energy")
    def compute_laplacian_energy(
        self,
        state: np.ndarray,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Énergie du laplacien (rugosité).
        
        Args:
            state: Matrice 2D
            normalize: Normaliser par énergie totale ?
            epsilon: Protection division
        
        Returns:
            float: Somme |∇²state|²
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Laplacien = ∂²/∂x² + ∂²/∂y²
            - Mesure courbure locale
            - Élevé → surface rugueuse
            - Faible → surface lisse
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Laplacien discret
        laplacian = ndimage.laplace(state)
        
        # Énergie
        energy = float(np.sum(laplacian ** 2))
        
        if normalize:
            total_energy = np.sum(state ** 2) + epsilon
            energy = energy / total_energy
        
        return energy
    
    @register_function("local_variance")
    def compute_local_variance(
        self,
        state: np.ndarray,
        window_size: int = 3,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance locale moyenne.
        
        Args:
            state: Matrice 2D
            window_size: Taille fenêtre (impair)
            normalize: Normaliser par variance globale ?
            epsilon: Protection division
        
        Returns:
            float: Moyenne des variances locales
        
        Raises:
            ValueError: Si non 2D ou window_size pair
        
        Notes:
            - Mesure hétérogénéité texture
            - Élevé → texture complexe, hétérogène
            - Faible → texture uniforme, homogène
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if window_size % 2 == 0:
            raise ValueError(f"window_size doit être impair, reçu {window_size}")
        
        # Moyenne locale
        local_mean = ndimage.uniform_filter(state, size=window_size)
        
        # Variance locale = E[X²] - E[X]²
        local_mean_sq = ndimage.uniform_filter(state**2, size=window_size)
        local_var = local_mean_sq - local_mean**2
        
        # Moyenne des variances locales
        mean_local_var = float(np.mean(local_var))
        
        if normalize:
            global_var = np.var(state) + epsilon
            mean_local_var = mean_local_var / global_var
        
        return mean_local_var
    
    @register_function("edge_density")
    def compute_edge_density(
        self,
        state: np.ndarray,
        threshold: float = 0.1,
        method: str = 'sobel'
    ) -> float:
        """
        Densité de contours détectés.
        
        Args:
            state: Matrice 2D
            threshold: Seuil détection contour (relatif au max)
            method: 'sobel' | 'prewitt' | 'laplace'
        
        Returns:
            float: Fraction pixels > seuil dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou méthode inconnue
        
        Notes:
            - Mesure complexité structure
            - Élevé → nombreux contours, structure complexe
            - Faible → peu de contours, structure simple
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Détection contours
        if method == 'sobel':
            sx = ndimage.sobel(state, axis=0)
            sy = ndimage.sobel(state, axis=1)
            edges = np.hypot(sx, sy)
        elif method == 'prewitt':
            sx = ndimage.prewitt(state, axis=0)
            sy = ndimage.prewitt(state, axis=1)
            edges = np.hypot(sx, sy)
        elif method == 'laplace':
            edges = np.abs(ndimage.laplace(state))
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Normaliser
        max_edge = np.max(edges)
        if max_edge == 0:
            return 0.0
        
        edges_normalized = edges / max_edge
        
        # Densité
        density = np.sum(edges_normalized > threshold) / edges.size
        
        return float(density)
    
    @register_function("spatial_autocorrelation")
    def compute_spatial_autocorrelation(
        self,
        state: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Autocorrélation spatiale (Moran's I simplifié).
        
        Args:
            state: Matrice 2D
            lag: Distance pixels voisins
        
        Returns:
            float: Corrélation spatiale
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure similarité pixels voisins
            - > 0 : Agrégation (pixels similaires groupés)
            - ≈ 0 : Distribution aléatoire
            - < 0 : Dispersion (alternance valeurs)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Décaler selon lag
        shifted_h = np.roll(state, lag, axis=0)
        shifted_v = np.roll(state, lag, axis=1)
        
        # Corrélations
        corr_h = np.corrcoef(state.flatten(), shifted_h.flatten())[0, 1]
        corr_v = np.corrcoef(state.flatten(), shifted_v.flatten())[0, 1]
        
        # Moyenne
        if np.isnan(corr_h):
            corr_h = 0.0
        if np.isnan(corr_v):
            corr_v = 0.0
        
        autocorr = (corr_h + corr_v) / 2.0
        
        return float(autocorr)
    
    @register_function("smoothness")
    def compute_smoothness(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Mesure de lissage (inverse rugosité).
        
        Args:
            state: Matrice 2D
            epsilon: Protection division
        
        Returns:
            float: 1 / (1 + variance_gradient)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Valeurs dans [0, 1]
            - 1.0 → parfaitement lisse
            - 0.0 → très rugueux
            - Basé sur variance du gradient
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Gradient
        gx, gy = np.gradient(state)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Variance gradient
        var_gradient = np.var(gradient_magnitude)
        
        # Smoothness
        smoothness = 1.0 / (1.0 + var_gradient + epsilon)
        
        return float(smoothness)
        
# tests/utilities/registries/spectral_registry.py
"""
Registre des analyses spectrales.

Fonctions disponibles :
- eigenvalue_max : Plus grande valeur propre
- eigenvalue_distribution : Statistiques spectre
- spectral_gap : Écart spectral
- fft_power : Puissance fréquentielle
- fft_entropy : Entropie spectrale
- spectral_radius : Rayon spectral
"""

import numpy as np
from .base_registry import BaseRegistry, register_function


class SpectralRegistry(BaseRegistry):
    """Registre des analyses spectrales."""
    
    REGISTRY_KEY = "spectral"
    
    @register_function("eigenvalue_max")
    def compute_eigenvalue_max(
        self,
        state: np.ndarray,
        absolute: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Plus grande valeur propre.
        
        Args:
            state: Matrice carrée 2D
            absolute: Considérer |λ| ou λ ?
            epsilon: Protection si matrice quasi-singulière
        
        Returns:
            float: λ_max ou |λ|_max
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - absolute=True → plus grande magnitude
            - absolute=False → plus grande valeur algébrique
            - Peut être complexe → prend module si absolute=True
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.linalg.eigvals(state)
            
            if absolute:
                return float(np.max(np.abs(eigenvalues)))
            else:
                # Prendre partie réelle si complexe
                if np.iscomplexobj(eigenvalues):
                    eigenvalues = eigenvalues.real
                return float(np.max(eigenvalues))
        
        except np.linalg.LinAlgError:
            return float(epsilon)
    
    @register_function("eigenvalue_distribution")
    def compute_eigenvalue_stats(
        self,
        state: np.ndarray,
        stat: str = 'mean',
        epsilon: float = 1e-10
    ) -> float:
        """
        Statistiques distribution valeurs propres.
        
        Args:
            state: Matrice carrée 2D
            stat: 'mean' | 'std' | 'min' | 'max' | 'range'
            epsilon: Protection calculs
        
        Returns:
            float: Statistique choisie sur |λ_i|
        
        Raises:
            ValueError: Si non 2D, non carrée, ou stat inconnue
        
        Notes:
            - Calcul sur magnitudes (|λ|)
            - 'range' = max - min
            - Robuste aux valeurs propres complexes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.abs(np.linalg.eigvals(state))
            
            if stat == 'mean':
                return float(np.mean(eigenvalues))
            elif stat == 'std':
                return float(np.std(eigenvalues))
            elif stat == 'min':
                return float(np.min(eigenvalues))
            elif stat == 'max':
                return float(np.max(eigenvalues))
            elif stat == 'range':
                return float(np.max(eigenvalues) - np.min(eigenvalues))
            else:
                raise ValueError(f"Stat inconnue: {stat}")
        
        except np.linalg.LinAlgError:
            return float(epsilon)
    
    @register_function("spectral_gap")
    def compute_spectral_gap(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Écart spectral (λ_max - λ_2).
        
        Args:
            state: Matrice carrée 2D
            normalize: Diviser par λ_max ?
            epsilon: Protection division
        
        Returns:
            float: λ_1 - λ_2 ou (λ_1 - λ_2) / λ_1
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure séparation valeur propre dominante
            - Gap grand → convergence rapide
            - normalize=True → valeur relative dans [0, 1]
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.abs(np.linalg.eigvals(state))
            eigenvalues = np.sort(eigenvalues)[::-1]  # Décroissant
            
            if len(eigenvalues) < 2:
                return float(0.0)
            
            gap = eigenvalues[0] - eigenvalues[1]
            
            if normalize and eigenvalues[0] > epsilon:
                gap = gap / eigenvalues[0]
            
            return float(gap)
        
        except np.linalg.LinAlgError:
            return float(0.0)
    
    @register_function("fft_power")
    def compute_fft_power(
        self,
        state: np.ndarray,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Puissance totale spectre FFT.
        
        Args:
            state: Tenseur rang N
            normalize: Normaliser par nombre éléments ?
            epsilon: Protection calculs
        
        Returns:
            float: Somme |FFT(state)|²
        
        Notes:
            - FFT multi-dimensionnelle si rang > 1
            - Mesure énergie fréquentielle
            - normalize=True → comparable entre tailles
        """
        fft_result = np.fft.fftn(state)
        power = np.sum(np.abs(fft_result) ** 2)
        
        if normalize:
            power = power / (state.size + epsilon)
        
        return float(power)
    
    @register_function("fft_entropy")
    def compute_fft_entropy(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Entropie spectrale du spectre FFT.
        
        Args:
            state: Tenseur rang N
            epsilon: Protection log(0)
        
        Returns:
            float: -Σ p_i log(p_i) où p_i = |FFT_i|² / Σ|FFT|²
        
        Notes:
            - Entropie élevée → spectre dispersé
            - Entropie faible → spectre concentré
            - Valeurs dans [0, log(N)] où N = state.size
        """
        fft_result = np.fft.fftn(state)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Normaliser en distribution
        total_power = np.sum(power_spectrum) + epsilon
        probabilities = power_spectrum / total_power
        
        # Entropie de Shannon
        entropy = -np.sum(
            probabilities * np.log(probabilities + epsilon)
        )
        
        return float(entropy)
    
    @register_function("spectral_radius")
    def compute_spectral_radius(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Rayon spectral (max |λ_i|).
        
        Args:
            state: Matrice carrée 2D
            epsilon: Protection si singulière
        
        Returns:
            float: ρ(A) = max_i |λ_i|
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - ρ(A) ≤ ||A|| pour toute norme
            - ρ(A) < 1 → itérations A^n convergent
            - ρ(A) > 1 → potentiellement divergent
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.linalg.eigvals(state)
            return float(np.max(np.abs(eigenvalues)))
        
        except np.linalg.LinAlgError:
            return float(epsilon)
            
# tests/utilities/registries/statistical_registry.py
"""
Registre des analyses statistiques.

Fonctions disponibles :
- entropy : Entropie de Shannon
- kurtosis : Aplatissement distribution
- skewness : Asymétrie distribution
- variance : Variance
- std_normalized : Écart-type normalisé (coefficient variation)
- correlation_mean : Corrélation moyenne
- sparsity : Mesure de parcimonie
"""

import numpy as np
from scipy import stats
from .base_registry import BaseRegistry, register_function


class StatisticalRegistry(BaseRegistry):
    """Registre des analyses statistiques."""
    
    REGISTRY_KEY = "statistical"
    
    @register_function("entropy")
    def compute_entropy(
        self,
        state: np.ndarray,
        bins: int = 50,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Entropie de Shannon de la distribution.
        
        Args:
            state: Tenseur rang N
            bins: Nombre de bins histogramme
            normalize: Normaliser par log(bins) ?
            epsilon: Protection log(0)
        
        Returns:
            float: H = -Σ p_i log(p_i)
        
        Notes:
            - Mesure dispersion distribution
            - normalize=True → valeurs dans [0, 1]
            - Entropie max → distribution uniforme
            - Entropie min → distribution concentrée
        """
        # Aplatir tenseur
        values = state.flatten()
        
        # Histogramme
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Normaliser en probabilités
        hist = hist / (np.sum(hist) + epsilon)
        
        # Entropie
        entropy = -np.sum(hist * np.log(hist + epsilon))
        
        if normalize:
            max_entropy = np.log(bins)
            entropy = entropy / (max_entropy + epsilon)
        
        return float(entropy)
    
    @register_function("kurtosis")
    def compute_kurtosis(
        self,
        state: np.ndarray,
        fisher: bool = True
    ) -> float:
        """
        Aplatissement (kurtosis) de la distribution.
        
        Args:
            state: Tenseur rang N
            fisher: Utiliser définition Fisher (excess kurtosis) ?
        
        Returns:
            float: Kurtosis
        
        Notes:
            - fisher=True → kurtosis - 3 (normale → 0)
            - fisher=False → kurtosis brute (normale → 3)
            - > 0 : Distribution pointue (leptokurtique)
            - < 0 : Distribution aplatie (platykurtique)
        """
        values = state.flatten()
        kurt = float(stats.kurtosis(values, fisher=fisher))
        return kurt
    
    @register_function("skewness")
    def compute_skewness(
        self,
        state: np.ndarray
    ) -> float:
        """
        Asymétrie (skewness) de la distribution.
        
        Args:
            state: Tenseur rang N
        
        Returns:
            float: Coefficient d'asymétrie
        
        Notes:
            - = 0 : Distribution symétrique
            - > 0 : Queue droite (valeurs extrêmes positives)
            - < 0 : Queue gauche (valeurs extrêmes négatives)
        """
        values = state.flatten()
        skew = float(stats.skew(values))
        return skew
    
    @register_function("variance")
    def compute_variance(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance de la distribution.
        
        Args:
            state: Tenseur rang N
            normalize: Diviser par carré de la moyenne ?
            epsilon: Protection division
        
        Returns:
            float: Var(X) ou Var(X) / E[X]²
        
        Notes:
            - Variance = E[(X - E[X])²]
            - normalize=True → coefficient de variation²
            - Mesure dispersion relative
        """
        values = state.flatten()
        var = float(np.var(values))
        
        if normalize:
            mean = np.mean(values)
            var = var / ((mean ** 2) + epsilon)
        
        return var
    
    @register_function("std_normalized")
    def compute_std_normalized(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient de variation (CV).
        
        Args:
            state: Tenseur rang N
            epsilon: Protection division
        
        Returns:
            float: CV = σ / |μ|
        
        Notes:
            - Mesure dispersion relative
            - Sans unité (comparable entre échelles)
            - CV faible → valeurs homogènes
            - CV élevé → valeurs dispersées
        """
        values = state.flatten()
        mean = np.mean(values)
        std = np.std(values)
        
        cv = std / (np.abs(mean) + epsilon)
        
        return float(cv)
    
    @register_function("correlation_mean")
    def compute_correlation_mean(
        self,
        state: np.ndarray,
        axis: int = 0,
        method: str = 'pearson'
    ) -> float:
        """
        Corrélation moyenne entre lignes ou colonnes.
        
        Args:
            state: Matrice 2D
            axis: 0 (lignes) ou 1 (colonnes)
            method: 'pearson' | 'spearman'
        
        Returns:
            float: Moyenne corrélations par paires
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure similarité patterns
            - Valeurs dans [-1, 1]
            - Proche de 1 → patterns similaires
            - Proche de 0 → patterns indépendants
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if axis == 0:
            vectors = state
        else:
            vectors = state.T
        
        n = vectors.shape[0]
        if n < 2:
            return 0.0
        
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                if method == 'pearson':
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                elif method == 'spearman':
                    corr, _ = stats.spearmanr(vectors[i], vectors[j])
                else:
                    raise ValueError(f"Méthode inconnue: {method}")
                
                if np.isfinite(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        return float(np.mean(correlations))
    
    @register_function("sparsity")
    def compute_sparsity(
        self,
        state: np.ndarray,
        threshold: float = 1e-6
    ) -> float:
        """
        Mesure de parcimonie (fraction valeurs proches de 0).
        
        Args:
            state: Tenseur rang N
            threshold: Seuil |x| < threshold → 0
        
        Returns:
            float: Fraction d'éléments "zéro" dans [0, 1]
        
        Notes:
            - 1.0 → tous éléments nuls (très parcimonieux)
            - 0.0 → aucun élément nul (dense)
            - Utile pour détecter structures creuses
        """
        values = state.flatten()
        num_zeros = np.sum(np.abs(values) < threshold)
        sparsity = num_zeros / len(values)
        
        return float(sparsity)
        
# tests/utilities/registries/topological_registry.py
"""
Registre des analyses topologiques simplifiées.

Fonctions disponibles :
- connected_components : Nombre composantes connexes
- euler_characteristic : Caractéristique Euler (2D)
- perimeter_area_ratio : Ratio périmètre/aire
- compactness : Compacité formes
- holes_count : Estimation nombre trous
- fractal_dimension : Dimension fractale (box-counting)
"""

import numpy as np
from scipy import ndimage
from .base_registry import BaseRegistry, register_function


class TopologicalRegistry(BaseRegistry):
    """Registre des analyses topologiques simplifiées."""
    
    REGISTRY_KEY = "topological"
    
    @register_function("connected_components")
    def compute_connected_components(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        connectivity: int = 1
    ) -> float:
        """
        Nombre de composantes connexes.
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            connectivity: 1 (4-connexe) ou 2 (8-connexe)
        
        Returns:
            float: Nombre composantes connexes
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Binarise state > threshold
            - Compte régions connexes
            - Élevé → structure fragmentée
            - Faible → structure cohérente
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser composantes
        labeled, num_components = ndimage.label(binary, structure=np.ones((3, 3)) if connectivity == 2 else None)
        
        return float(num_components)
    
    @register_function("euler_characteristic")
    def compute_euler_characteristic(
        self,
        state: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Caractéristique Euler (χ = V - E + F).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
        
        Returns:
            float: χ ≈ composantes - trous
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Invariant topologique fondamental
            - χ > 0 : Plus de composantes que de trous
            - χ = 0 : Composantes = trous (ex: tore)
            - χ < 0 : Plus de trous
            - Approximation via composantes et contours
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Composantes (objets)
        labeled, num_objects = ndimage.label(binary)
        
        # Trous = composantes dans inverse
        labeled_holes, num_holes = ndimage.label(~binary)
        
        # χ ≈ objets - trous (simplifié)
        # Note: Vrai Euler nécessite analyse plus fine
        euler = num_objects - num_holes
        
        return float(euler)
    
    @register_function("perimeter_area_ratio")
    def compute_perimeter_area_ratio(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Ratio périmètre/aire moyen.
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            epsilon: Protection division
        
        Returns:
            float: Moyenne P/A des composantes
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure complexité frontière
            - Élevé → frontières irrégulières, complexes
            - Faible → formes compactes
            - Cercle: P/A minimal pour aire donnée
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser
        labeled, num_objects = ndimage.label(binary)
        
        if num_objects == 0:
            return 0.0
        
        ratios = []
        for i in range(1, num_objects + 1):
            # Masque objet
            mask = labeled == i
            
            # Aire
            area = np.sum(mask)
            
            if area < 4:  # Trop petit
                continue
            
            # Périmètre (approximation via gradient)
            boundary = mask ^ ndimage.binary_erosion(mask)
            perimeter = np.sum(boundary)
            
            # Ratio
            if area > epsilon:
                ratios.append(perimeter / area)
        
        if not ratios:
            return 0.0
        
        return float(np.mean(ratios))
    
    @register_function("compactness")
    def compute_compactness(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compacité (isoperimetric ratio).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            epsilon: Protection division
        
        Returns:
            float: Moyenne 4πA/P² dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → cercle parfait (compacité max)
            - < 1.0 → forme moins compacte
            - Mesure "circularité"
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser
        labeled, num_objects = ndimage.label(binary)
        
        if num_objects == 0:
            return 0.0
        
        compactness_values = []
        for i in range(1, num_objects + 1):
            mask = labeled == i
            
            area = np.sum(mask)
            if area < 4:
                continue
            
            boundary = mask ^ ndimage.binary_erosion(mask)
            perimeter = np.sum(boundary)
            
            if perimeter > epsilon:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactness = np.clip(compactness, 0.0, 1.0)
                compactness_values.append(compactness)
        
        if not compactness_values:
            return 0.0
        
        return float(np.mean(compactness_values))
    
    @register_function("holes_count")
    def compute_holes_count(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        min_hole_size: int = 4
    ) -> float:
        """
        Estimation nombre trous (régions fermées vides).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            min_hole_size: Taille minimale trou
        
        Returns:
            float: Nombre trous détectés
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Trou = composante connexe de background encerclée
            - Approximation simple via fill_holes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Remplir trous
        filled = ndimage.binary_fill_holes(binary)
        
        # Différence = trous
        holes = filled & ~binary
        
        # Labelliser trous
        labeled_holes, num_holes = ndimage.label(holes)
        
        # Filtrer petits trous
        valid_holes = 0
        for i in range(1, num_holes + 1):
            if np.sum(labeled_holes == i) >= min_hole_size:
                valid_holes += 1
        
        return float(valid_holes)
    
    @register_function("fractal_dimension")
    def compute_fractal_dimension(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        max_box_size: int = None,
        num_scales: int = 10
    ) -> float:
        """
        Dimension fractale (box-counting).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            max_box_size: Taille box max (None = min(shape)/4)
            num_scales: Nombre échelles testées
        
        Returns:
            float: Dimension fractale (1 ≤ D ≤ 2 pour 2D)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure complexité auto-similaire
            - D ≈ 1 : Structure linéaire
            - D ≈ 2 : Remplissage plan
            - D intermédiaire : Structure fractale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        if max_box_size is None:
            max_box_size = min(binary.shape) // 4
        
        # Échelles box
        box_sizes = np.logspace(
            0, np.log2(max_box_size), num=num_scales, base=2
        ).astype(int)
        box_sizes = np.unique(box_sizes)
        box_sizes = box_sizes[box_sizes > 0]
        
        if len(box_sizes) < 2:
            return 1.0
        
        # Compter boxes à chaque échelle
        counts = []
        for box_size in box_sizes:
            # Sous-échantillonner
            if box_size > 1:
                downsampled = binary[::box_size, ::box_size]
            else:
                downsampled = binary
            
            # Compter boxes occupées
            count = np.sum(downsampled)
            counts.append(count)
        
        counts = np.array(counts)
        counts = counts[counts > 0]  # Filtrer zéros
        
        if len(counts) < 2:
            return 1.0
        
        # Régression log-log
        box_sizes = box_sizes[:len(counts)]
        
        coeffs = np.polyfit(
            np.log(box_sizes),
            np.log(counts),
            1
        )
        
        # Dimension = pente
        dimension = -coeffs[0]
        
        # Clipper dans [1, 2]
        dimension = np.clip(dimension, 1.0, 2.0)
        
        return float(dimension)
        
# tests/utilities/aggregation_utils.py
"""
Aggregation Utilities - Agrégations statistiques multi-runs.

RESPONSABILITÉS :
- Agrégations métriques (médiane, q1, q3, cv)
- Détection multimodalité (IQR ratio, bimodal)
- Statistiques descriptives inter-runs

UTILISATEURS :
- gamma_profiling.py (profiling comportemental)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import numpy as np
from typing import List, Dict, Any


# =============================================================================
# AGRÉGATIONS MÉTRIQUES
# =============================================================================

def aggregate_summary_metrics(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège métriques statistiques inter-runs.
    
    CALCULS :
    - final_value : {median, q1, q3, mean, std}
    - initial_value : médiane valeurs initiales
    - mean_value : moyenne des moyennes
    - cv : coefficient variation (std/mean sur finales)
    
    Args:
        observations: Liste observations (même gamma × test)
        metric_name: Nom métrique à agréger
    
    Returns:
        {
            'final_value': {
                'median': float,
                'q1': float,
                'q3': float,
                'mean': float,
                'std': float
            },
            'initial_value': float,
            'mean_value': float,
            'cv': float
        }
        
        Retourne {} si aucune valeur finale disponible
    
    Examples:
        >>> obs = [
        ...     {'observation_data': {'statistics': {
        ...         'asymmetry': {'final': 1.2, 'initial': 1.0, 'mean': 1.1}
        ...     }}},
        ...     {'observation_data': {'statistics': {
        ...         'asymmetry': {'final': 1.5, 'initial': 1.0, 'mean': 1.3}
        ...     }}}
        ... ]
        >>> result = aggregate_summary_metrics(obs, 'asymmetry')
        >>> result['final_value']['median']
        1.35
        >>> result['cv']  # Coefficient variation
        0.157
    
    Notes:
        - cv = std(final) / |mean(final)| (mesure dispersion relative)
        - Protection division par zéro (+ 1e-10)
        - initial_value = médiane (robuste outliers)
    """
    final_values = []
    initial_values = []
    mean_values = []
    
    # Collecter valeurs depuis observations
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        
        final = stats.get('final')
        initial = stats.get('initial')
        mean = stats.get('mean')
        
        if final is not None:
            final_values.append(final)
        if initial is not None:
            initial_values.append(initial)
        if mean is not None:
            mean_values.append(mean)
    
    # Vérifier données disponibles
    if not final_values:
        return {}
    
    # Convertir numpy pour calculs
    final_values = np.array(final_values)
    
    # Statistiques final_value (distribution complète)
    final_stats = {
        'median': float(np.median(final_values)),
        'q1': float(np.percentile(final_values, 25)),
        'q3': float(np.percentile(final_values, 75)),
        'mean': float(np.mean(final_values)),
        'std': float(np.std(final_values))
    }
    
    # Valeurs agrégées simples
    initial_value = float(np.median(initial_values)) if initial_values else 0.0
    mean_value = float(np.mean(mean_values)) if mean_values else 0.0
    
    # Coefficient variation (dispersion relative)
    cv = float(np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10))
    
    return {
        'final_value': final_stats,
        'initial_value': initial_value,
        'mean_value': mean_value,
        'cv': cv
    }


# =============================================================================
# DÉTECTION MULTIMODALITÉ
# =============================================================================

def aggregate_run_dispersion(observations: List[dict], metric_name: str) -> dict:
    """
    Calcule indicateurs multimodalité inter-runs.
    
    INDICATEURS :
    - final_value_iqr_ratio : Q3 / Q1 (détection bimodalité)
    - cv_across_runs : std / |mean| (dispersion relative)
    - bimodal_detected : True si iqr_ratio > 3.0
    
    PRINCIPE BIMODALITÉ :
    - IQR ratio > 3.0 suggère 2+ modes distincts
    - Exemple : Q1=0.1, Q3=0.5 → ratio=5.0 → bimodal probable
    - Heuristique R0 (pas test statistique formel)
    
    Args:
        observations: Liste observations (même gamma × test)
        metric_name: Nom métrique à analyser
    
    Returns:
        {
            'final_value_iqr_ratio': float,
            'cv_across_runs': float,
            'bimodal_detected': bool
        }
        
        Retourne valeurs 0.0/False si < 2 valeurs finales
    
    Examples:
        >>> # Cas homogène
        >>> obs_homo = [
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.0}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.1}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.9}}}}
        ... ]
        >>> result = aggregate_run_dispersion(obs_homo, 'metric')
        >>> result['bimodal_detected']
        False
        >>> result['iqr_ratio']
        1.18  # Q3=1.05, Q1=0.95 → ratio faible
        
        >>> # Cas bimodal
        >>> obs_bimodal = [
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.1}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.2}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.0}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.1}}}}
        ... ]
        >>> result = aggregate_run_dispersion(obs_bimodal, 'metric')
        >>> result['bimodal_detected']
        True
        >>> result['iqr_ratio']
        5.5  # Q3=1.05, Q1=0.15 → ratio élevé
    
    Notes:
        - Seuil bimodal (3.0) heuristique R0
        - Protection division par zéro Q1 (max(Q1, 1e-10))
        - cv_across_runs : même calcul que aggregate_summary_metrics
    """
    final_values = []
    
    # Collecter valeurs finales
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        final = stats.get('final')
        
        if final is not None:
            final_values.append(final)
    
    # Vérifier données suffisantes
    if len(final_values) < 2:
        return {
            'final_value_iqr_ratio': 0.0,
            'cv_across_runs': 0.0,
            'bimodal_detected': False
        }
    
    # Convertir numpy
    final_values = np.array(final_values)
    
    # IQR ratio (Q3 / Q1)
    q1 = np.percentile(final_values, 25)
    q3 = np.percentile(final_values, 75)
    iqr_ratio = q3 / max(q1, 1e-10)
    
    # Coefficient variation
    cv = np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10)
    
    # Détection bimodalité (heuristique)
    bimodal = iqr_ratio > 3.0
    
    return {
        'final_value_iqr_ratio': float(iqr_ratio),
        'cv_across_runs': float(cv),
        'bimodal_detected': bool(bimodal)
    }


# =============================================================================
# HELPERS (Futures Extensions)
# =============================================================================

def compute_dominant_value(
    values: List[Any],
    method: str = 'mode'
) -> Any:
    """
    Calcule valeur dominante (mode, médiane, etc.).
    
    TODO : Implémenter si nécessaire pour analyses catégorielles.
    
    Args:
        values: Liste valeurs
        method: 'mode' | 'median' | 'mean'
    
    Returns:
        Valeur dominante
    
    Examples:
        >>> compute_dominant_value([1, 1, 2, 3], method='mode')
        1
        >>> compute_dominant_value([1, 2, 3], method='median')
        2
    """
    raise NotImplementedError("compute_dominant_value non implémenté (Phase future)")


def aggregate_event_counts(
    observations: List[dict],
    metric_name: str
) -> dict:
    """
    Compte événements booléens inter-runs.
    
    TODO : Implémenter si besoin agrégations événements dynamiques génériques.
    
    Args:
        observations: Liste observations
        metric_name: Nom métrique
    
    Returns:
        {
            'event_name': fraction,
            ...
        }
    
    Notes:
        - Actuellement implémenté dans gamma_profiling.aggregate_dynamic_signatures()
        - Extraction ici si besoin réutilisation ailleurs
    """
    raise NotImplementedError("aggregate_event_counts non implémenté (Phase future)")
    
# tests/utilities/applicability.py

from typing import Tuple, Dict, Any, Callable

# Registre extensible de validators
VALIDATORS: Dict[str, Callable] = {
    # ✅ CORRIGÉ : None signifie "pas de contrainte"
    'requires_rank': lambda run_metadata, expected: 
        expected is None or len(run_metadata['state_shape']) == expected,
    
    'requires_square': lambda run_metadata, required: 
        not required or (
            len(run_metadata['state_shape']) == 2 and 
            run_metadata['state_shape'][0] == run_metadata['state_shape'][1]
        ),
    
    'allowed_d_types': lambda run_metadata, allowed: 
        'ALL' in allowed or run_metadata['d_encoding_id'].split('-')[0] in allowed,
    
    # ✅ CORRIGÉ : False signifie "pas de contrainte"
    'requires_even_dimension': lambda run_metadata, required: 
        not required or all(dim % 2 == 0 for dim in run_metadata['state_shape']),
    
    # ✅ CORRIGÉ : None signifie "pas de contrainte"
    'minimum_dimension': lambda run_metadata, min_dim:
        min_dim is None or all(dim >= min_dim for dim in run_metadata['state_shape']),
}


def check(test_module, run_metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Vérifie applicabilité sur métadonnées uniquement.
    
    Args:
        test_module: Module test importé
        run_metadata: {
            'gamma_id': str,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int,
            'state_shape': tuple,
        }
    
    Returns:
        (applicable: bool, reason: str)
    
    Examples:
        >>> check(test_sym_001, {'state_shape': (10, 10), 'd_encoding_id': 'SYM-001', ...})
        (True, "")
        
        >>> check(test_sym_001, {'state_shape': (10, 20), 'd_encoding_id': 'SYM-001', ...})
        (False, "requires_square = True non satisfait")
    """
    spec = test_module.APPLICABILITY_SPEC
    
    for constraint_name, constraint_value in spec.items():
        if constraint_name not in VALIDATORS:
            return False, f"Contrainte inconnue : {constraint_name}"
        
        validator = VALIDATORS[constraint_name]
        
        try:
            is_valid = validator(run_metadata, constraint_value)
            
            if not is_valid:
                return False, f"{constraint_name} = {constraint_value} non satisfait"
        
        except KeyError as e:
            return False, f"Info manquante : {e}"
        
        except Exception as e:
            return False, f"Erreur validation {constraint_name}: {e}"
    
    return True, ""


def add_validator(name: str, validator: Callable) -> None:
    """
    Ajoute un validator custom.
    
    Args:
        name: Nom unique du validator
        validator: Fonction (run_metadata, constraint_value) -> bool
    
    Raises:
        ValueError: Si nom déjà existant
    """
    if name in VALIDATORS:
        raise ValueError(f"Validator '{name}' déjà existant")
    
    VALIDATORS[name] = validator
    
# tests/utilities/config_loader.py
"""
Utilitaire centralisé chargement configs YAML.

Architecture Charter 5.4 - Section 12.6
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

class ConfigLoader:
    """
    Gestionnaire centralisé configs YAML.
    
    Responsabilités :
    1. Charger n'importe quel type config (params, scoring, thresholds)
    2. Gérer fusion global + specific automatique
    3. Cache pour performance
    4. Validation basique structure
    """
    
    BASE_PATH = Path("tests/config")
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def load(
        self,
        config_type: str,
        config_id: str,
        test_id: Optional[str] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Charge config avec fusion auto global + specific.
        
        Args:
            config_type: 'params' | 'verdict' 
            config_id: Ex 'params_default_v1', 'scoring_conservative_v1'
            test_id: Ex 'UNIV-001' (optionnel, pour override)
            force_reload: Ignorer cache
        
        Returns:
            dict config fusionné
        
        Examples:
            >>> loader = ConfigLoader()
            >>> params = loader.load('params', 'params_default_v1')
            >>> params_univ = loader.load('params', 'params_default_v1', 'UNIV-001')
        
        Raises:
            FileNotFoundError: Si config global absent
            ValueError: Si type config invalide
        """
        # Validation
        valid_types = ['params', 'verdict']
        if config_type not in valid_types:
            raise ValueError(
                f"Type config invalide '{config_type}'. "
                f"Attendu: {valid_types}"
            )
        
        # Clé cache
        cache_key = f"{config_type}:{config_id}:{test_id or 'global'}"
        
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # 1. Charger global (obligatoire)
        global_config = self._load_global(config_type, config_id)
        
        # 2. Charger specific (optionnel)
        if test_id:
            specific_config = self._load_specific(config_type, test_id)
            if specific_config:
                merged = self._merge_configs(global_config, specific_config)
            else:
                merged = global_config
        else:
            merged = global_config
        
        # Cache et retour
        self._cache[cache_key] = merged.copy()
        return merged
    
    def _load_global(self, config_type: str, config_id: str) -> Dict:
        """Charge config global."""
        path = self.BASE_PATH / "global" / f"{config_id}.yaml"
        
        if not path.exists():
            raise FileNotFoundError(
                f"Config global non trouvée: {path}\n"
                f"Vérifier que {config_id}.yaml existe dans tests/config/global/"
            )
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validation basique
        self._validate_config(config, config_type, config_id)
        
        return config
    
    def _load_specific(self, config_type: str, test_id: str) -> Optional[Dict]:
        """Charge config spécifique test (si existe)."""
        test_dir = self.BASE_PATH / "tests" / test_id
        
        if not test_dir.exists():
            return None
        
        # Chercher fichiers matching (ex: params_custom_v1.yaml)
        pattern = f"{config_type}_*.yaml"
        matching_files = list(test_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Si plusieurs, prendre le plus récent (ou alphabétique)
        selected = sorted(matching_files)[-1]
        
        with open(selected, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _merge_configs(self, global_config: Dict, specific_config: Dict) -> Dict:
        """
        Fusionne configs (specific override global).
        
        Stratégie fusion :
        - Clés top-level : specific écrase global
        - Dicts imbriqués : merge récursif
        - Listes : specific remplace global
        """
        merged = global_config.copy()
        
        for key, value in specific_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Merge récursif dicts
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                # Override direct
                merged[key] = value
        
        return merged
    
    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Merge récursif dictionnaires."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict, config_type: str, config_id: str):
        """Validation basique structure."""
        required_keys = ['version', 'config_id', 'description']
        
        missing = [k for k in required_keys if k not in config]
        if missing:
            warnings.warn(
                f"Config {config_id} manque clés: {missing}\n"
                f"Recommandé: ajouter metadata (version, description, etc.)"
            )
        
        # Vérifier cohérence ID
        if 'config_id' in config and config['config_id'] != config_id:
            warnings.warn(
                f"ID incohérent: fichier={config_id}, contenu={config['config_id']}"
            )
    
    def list_available(self, config_type: str) -> Dict[str, list]:
        """
        Liste configs disponibles.
        
        Returns:
            {
                'global': ['params_default_v1', ...],
                'tests': {
                    'UNIV-001': ['params_custom_v1', ...],
                    ...
                }
            }
        """
        result = {'global': [], 'tests': {}}
        
        # Configs globales
        global_dir = self.BASE_PATH / "global"
        if global_dir.exists():
            pattern = f"{config_type}_*.yaml"
            result['global'] = [
                f.stem for f in global_dir.glob(pattern)
            ]
        
        # Configs spécifiques
        tests_dir = self.BASE_PATH / "tests"
        if tests_dir.exists():
            for test_dir in tests_dir.iterdir():
                if test_dir.is_dir():
                    pattern = f"{config_type}_*.yaml"
                    matching = [f.stem for f in test_dir.glob(pattern)]
                    if matching:
                        result['tests'][test_dir.name] = matching
        
        return result
    
    def clear_cache(self):
        """Vide cache (utile après modification configs)."""
        self._cache.clear()


# Instance singleton
_loader = None

def get_loader() -> ConfigLoader:
    """Récupère instance singleton ConfigLoader."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


# tests/utilities/UTIL/cross_profiling.py
"""
Cross Profiling Module - Charter 6.1

RESPONSABILITÉ : Analyses croisées entre axes profiling

ARCHITECTURE (alignée analyse GPT) :
- Sépare calculs structurels (R0) et interprétation qualitative (R1+)
- profiling_common fournit espace vectoriel normalisé
- cross_profiling applique opérateurs sur cet espace

IMPLÉMENTATION R0 :
- Rankings multi-dimensionnels (extension by_test)
- Variance conditionnelle (discriminant power)
- Matrices concordance (helper projections conjointes)

PLACEHOLDERS R1+ :
- Interactions pairwise qualifiées
- Signatures globales avec vocabulaire interprétatif

FRONTIÈRE ÉPISTÉMIQUE :
- R0 : Calcule, mesure, compare (pas d'interprétation causale)
- R1+ : Qualifie, nomme, interprète (INVARIANT, AMPLIFIÉ, etc.)
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict, Counter
from scipy import stats


# ============================================================================
# RANKINGS MULTI-DIMENSIONNELS (R0 complet)
# ============================================================================

def rank_entities_by_metric(
    profiles: dict,
    grouping_dimension: str,
    metric_key: str,
    criterion: str | Callable = 'conservation'
) -> List[Tuple[str, float]]:
    """
    Ranking générique entités par métrique.
    
    REMPLACE : gamma_profiling.rank_gammas_by_test()
    EXTENSION : Critères multiples, tous axes supportés
    
    Args:
        profiles: Résultats profile_all_{axis}() depuis profiling_common
        grouping_dimension: Axe regroupement ('test', 'encoding', 'modifier')
        metric_key: Clé métrique (test_name si grouping='test', etc.)
        criterion: Critère scoring ou callable custom
    
    Returns:
        [(entity_id, score), ...] trié décroissant par score
    
    Critères standards :
        - 'conservation' : CONSERVES_* = 1.0, TRIVIAL = 0.5, autres = 0.0
        - 'stability' : Score basé instability_onset + collapse_fraction
        - 'homogeneity' : Score basé CV + bimodalité
        - callable : Fonction custom(prc_profile, diagnostic_signature) → float
    
    Exemples :
        # Gammas par test (rétrocompatibilité)
        rank_entities_by_metric(gamma_profiles, 'test', 'SYM-001', 'conservation')
        
        # Tests par gamma (inverse)
        rank_entities_by_metric(test_profiles, 'gamma', 'GAM-001', 'stability')
        
        # Modifiers par encoding
        rank_entities_by_metric(modifier_profiles, 'encoding', 'SYM-001', 'homogeneity')
        
        # Critère custom
        rank_entities_by_metric(
            gamma_profiles, 'test', 'TOP-001',
            criterion=lambda prc, diag: prc['confidence'] == 'high'
        )
    """
    scores = []
    
    for entity_id, entity_data in profiles.items():
        # Accéder profil selon grouping_dimension
        if grouping_dimension == 'test':
            # metric_key = test_name
            profile = entity_data['tests'].get(metric_key)
        else:
            # Grouping par autre dimension (encoding, modifier, etc.)
            # Rechercher dans tous tests
            profile = None
            for test_profile in entity_data['tests'].values():
                # Vérifier si test_profile contient metric_key
                # (structure future où tests pourraient grouper par encoding/modifier)
                # Pour R0 : on assume grouping='test' principalement
                pass
            
            if not profile:
                # Fallback : chercher test nommé metric_key
                profile = entity_data['tests'].get(metric_key)
        
        if not profile:
            continue
        
        prc = profile['prc_profile']
        diag = profile.get('diagnostic_signature', {})
        
        # Calcul score selon critère
        if callable(criterion):
            score = float(criterion(prc, diag))
        
        elif criterion == 'conservation':
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            if regime.startswith('CONSERVES_'):
                score = 1.0
            elif regime == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
        
        elif criterion == 'stability':
            dynamic_sig = diag.get('dynamic_signature', {})
            instab = dynamic_sig.get('instability_onset_median') is not None
            collapse = dynamic_sig.get('collapse_fraction', 0.0)
            score = 1.0 - (float(instab) * 0.5 + collapse * 0.5)
        
        elif criterion == 'homogeneity':
            run_disp = diag.get('run_dispersion', {})
            cv = run_disp.get('cv_across_runs', 1.0)
            bimodal = run_disp.get('bimodal_detected', False)
            score = (1.0 - min(cv, 1.0)) * (0.5 if bimodal else 1.0)
        
        else:
            # Critère inconnu
            score = 0.0
        
        scores.append((entity_id, score))
    
    # Tri décroissant
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================================
# VARIANCE CONDITIONNELLE (R0 complet)
# ============================================================================

def compute_discriminant_power(
    profiles: dict,
    test_name: str,
    reference_axis: str = 'gamma'
) -> dict:
    """
    Calcule pouvoir discriminant d'un test cross-entités.
    
    NOUVEAU R0 : Variance conditionnelle inter/intra entités
    EXPLOITABLE : Insights tests sans interprétation qualitative
    
    Mesure capacité test à différencier entités de l'axe de référence.
    
    Args:
        profiles: Résultats profile_all_{axis}() (ex: gamma_profiles)
        test_name: Test analysé (ex: 'SYM-001')
        reference_axis: Axe entités (ex: 'gamma', 'modifier')
    
    Returns:
        {
            'test_name': str,
            'reference_axis': str,
            'n_entities': int,
            'inter_entity_variance': float,     # Variance cross-entités (final_value)
            'intra_entity_variance': float,     # Variance cross-runs agrégée
            'discriminant_ratio': float,        # inter / intra
            'effect_size': float,               # Eta-squared (ANOVA)
            'kruskal_wallis': {
                'statistic': float,
                'p_value': float
            },
            'entities_ranked': [(entity_id, median_final_value), ...],
            'interpretation': str               # R0 : factuel uniquement
        }
    
    Interprétation factuelle R0 :
        - discriminant_ratio > 3.0 : "Variance inter-entités domine"
        - discriminant_ratio < 0.3 : "Variance intra-entité domine"
        - Autre : "Variance mixte"
    
    Note R1+ :
        Vocabulaire qualitatif (DISCRIMINANT, INVARIANT) dans detect_global_signatures()
    """
    # Extraction données
    final_values_by_entity = {}
    all_final_values = []
    
    for entity_id, entity_data in profiles.items():
        test_profile = entity_data['tests'].get(test_name)
        
        if not test_profile:
            continue
        
        instr = test_profile.get('instrumentation', {})
        metrics = instr.get('summary_metrics', {})
        
        # Extraire final_value (structure agrégée)
        final_value_stats = metrics.get('final_value', {})
        median_final = final_value_stats.get('median')
        
        if median_final is None or not np.isfinite(median_final):
            continue
        
        final_values_by_entity[entity_id] = median_final
        all_final_values.append(median_final)
    
    n_entities = len(final_values_by_entity)
    
    if n_entities < 2:
        return {
            'test_name': test_name,
            'reference_axis': reference_axis,
            'n_entities': n_entities,
            'inter_entity_variance': 0.0,
            'intra_entity_variance': 0.0,
            'discriminant_ratio': 0.0,
            'effect_size': 0.0,
            'kruskal_wallis': {'statistic': 0.0, 'p_value': 1.0},
            'entities_ranked': [],
            'interpretation': 'Insufficient entities for analysis'
        }
    
    # Variance inter-entités (variance des médianes)
    inter_entity_variance = float(np.var(list(final_values_by_entity.values())))
    
    # Variance intra-entité agrégée
    # Approximation R0 : moyenne des variances run_dispersion
    intra_variances = []
    for entity_id, entity_data in profiles.items():
        test_profile = entity_data['tests'].get(test_name)
        if not test_profile:
            continue
        
        diag = test_profile.get('diagnostic_signature', {})
        run_disp = diag.get('run_dispersion', {})
        cv = run_disp.get('cv_across_runs', 0.0)
        
        # Approximation variance depuis CV
        # var ≈ (cv * mean)^2
        # Utiliser median comme proxy mean
        median_val = final_values_by_entity.get(entity_id, 0.0)
        approx_var = (cv * median_val) ** 2
        intra_variances.append(approx_var)
    
    intra_entity_variance = float(np.mean(intra_variances)) if intra_variances else 0.0
    
    # Discriminant ratio
    discriminant_ratio = (
        inter_entity_variance / (intra_entity_variance + 1e-10)
        if intra_entity_variance > 0 else float('inf')
    )
    
    # Effect size (Eta-squared)
    # η² = SSB / SST
    grand_mean = np.mean(all_final_values)
    ssb = sum((val - grand_mean) ** 2 for val in final_values_by_entity.values())
    sst = sum((val - grand_mean) ** 2 for val in all_final_values)
    effect_size = ssb / sst if sst > 1e-10 else 0.0
    
    # Kruskal-Wallis test (non-paramétrique)
    # Pour R0 : on n'a que médianes, test approximatif
    # R1+ : utiliser distributions complètes depuis observations
    try:
        # Créer groupes artificiels (répéter médiane comme proxy)
        groups = [[val] * 5 for val in final_values_by_entity.values()]  # Proxy
        h_stat, p_value = stats.kruskal(*groups)
    except:
        h_stat, p_value = 0.0, 1.0
    
    # Ranking entités
    entities_ranked = sorted(
        final_values_by_entity.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Interprétation factuelle R0
    if discriminant_ratio > 3.0:
        interpretation = "Variance inter-entités domine (discriminant fort)"
    elif discriminant_ratio < 0.3:
        interpretation = "Variance intra-entité domine (faible discrimination)"
    else:
        interpretation = "Variance mixte (discrimination modérée)"
    
    return {
        'test_name': test_name,
        'reference_axis': reference_axis,
        'n_entities': n_entities,
        'inter_entity_variance': inter_entity_variance,
        'intra_entity_variance': intra_entity_variance,
        'discriminant_ratio': discriminant_ratio,
        'effect_size': effect_size,
        'kruskal_wallis': {
            'statistic': float(h_stat),
            'p_value': float(p_value)
        },
        'entities_ranked': entities_ranked,
        'interpretation': interpretation
    }


def compute_all_discriminant_powers(
    profiles: dict,
    reference_axis: str = 'gamma'
) -> dict:
    """
    Calcule discriminant power tous tests pour un axe.
    
    Args:
        profiles: Résultats profile_all_{axis}()
        reference_axis: Axe de référence
    
    Returns:
        {
            'test_name_1': {...},  # compute_discriminant_power()
            'test_name_2': {...},
            ...
        }
    """
    # Découvrir tous tests
    all_tests = set()
    for entity_data in profiles.values():
        all_tests.update(entity_data['tests'].keys())
    
    results = {}
    for test_name in all_tests:
        results[test_name] = compute_discriminant_power(
            profiles, test_name, reference_axis
        )
    
    return results


# ============================================================================
# MATRICES CONCORDANCE (R0 helper)
# ============================================================================

def _compute_concordance_matrix(
    profiles_a: dict,
    profiles_b: dict,
    comparison_field: str = 'regime'
) -> dict:
    """
    Calcule matrice concordance entre deux axes.
    
    HELPER R0 : Préparation projections conjointes
    USAGE R1+ : analyze_pairwise_interactions()
    
    Args:
        profiles_a: Profils axe A (ex: gamma_profiles)
        profiles_b: Profils axe B (ex: modifier_profiles)
        comparison_field: Champ comparé ('regime', 'behavior', 'timeline')
    
    Returns:
        {
            'test_name_1': {
                'concordance_rate': float,  # Fraction régimes identiques
                'mismatches': [
                    ('entity_a_id', 'entity_b_id', 'regime_a', 'regime_b'),
                    ...
                ],
                'n_comparisons': int
            },
            ...
        }
    
    Exemple lecture :
        Si concordance_rate = 0.8 pour test SYM-001 :
        → 80% paires (gamma, modifier) ont même régime sous SYM-001
    """
    # Découvrir tests communs
    tests_a = set()
    tests_b = set()
    
    for entity_data in profiles_a.values():
        tests_a.update(entity_data['tests'].keys())
    for entity_data in profiles_b.values():
        tests_b.update(entity_data['tests'].keys())
    
    common_tests = tests_a & tests_b
    
    results = {}
    
    for test_name in common_tests:
        matches = 0
        total = 0
        mismatches = []
        
        for entity_a_id, entity_a_data in profiles_a.items():
            test_profile_a = entity_a_data['tests'].get(test_name)
            if not test_profile_a:
                continue
            
            value_a = test_profile_a['prc_profile'].get(comparison_field)
            
            for entity_b_id, entity_b_data in profiles_b.items():
                test_profile_b = entity_b_data['tests'].get(test_name)
                if not test_profile_b:
                    continue
                
                value_b = test_profile_b['prc_profile'].get(comparison_field)
                
                total += 1
                
                # Comparaison (strip MIXED:: si régime)
                if comparison_field == 'regime':
                    value_a_clean = value_a.split('::')[-1] if value_a else None
                    value_b_clean = value_b.split('::')[-1] if value_b else None
                else:
                    value_a_clean = value_a
                    value_b_clean = value_b
                
                if value_a_clean == value_b_clean:
                    matches += 1
                else:
                    mismatches.append((
                        entity_a_id, entity_b_id,
                        value_a_clean, value_b_clean
                    ))
        
        concordance_rate = matches / total if total > 0 else 0.0
        
        results[test_name] = {
            'concordance_rate': concordance_rate,
            'mismatches': mismatches[:10],  # Limiter aux 10 premiers
            'n_comparisons': total
        }
    
    return results


# ============================================================================
# INTERACTIONS PAIRWISE (R0 placeholder + structure)
# ============================================================================

def analyze_pairwise_interactions(
    profiles_a: dict,
    profiles_b: dict,
    metric: str = 'regime_concordance'
) -> dict:
    """
    Analyse interactions 2-way entre deux axes profiling.
    
    STATUT R0 : Structure + helper concordance fonctionnel
    STATUT R1+ : Interprétation qualitative complète
    
    Args:
        profiles_a: Profils axe A (ex: gamma_profiles)
        profiles_b: Profils axe B (ex: modifier_profiles)
        metric: Métrique interaction
            - 'regime_concordance' : Concordance régimes (R0 implémenté)
            - 'behavior_amplification' : Amplification pathologies (R1+)
            - 'timeline_coupling' : Couplage timelines (R1+)
    
    Returns R0 (metric='regime_concordance'):
        {
            'axis_a': str,  # Détecté depuis metadata profiles
            'axis_b': str,
            'metric': str,
            'concordance_matrix': {...},  # _compute_concordance_matrix()
            'summary': {
                'mean_concordance': float,
                'high_concordance_tests': [test_name, ...],  # > 0.8
                'low_concordance_tests': [test_name, ...],   # < 0.3
                'interaction_detected': bool  # True si variance concordance > seuil
            }
        }
    
    Returns R1+ (autres métriques) :
        {
            'interaction_matrix': np.ndarray,  # Shape: (n_a, n_b)
            'significant_pairs': [
                ('entity_a_id', 'entity_b_id', 'pattern_description'),
                ...
            ],
            'qualitative_labels': {
                ('GAM-004', 'M1'): 'AMPLIFIED',
                ('GAM-009', 'ASY-003'): 'CONTEXTUAL',
                ...
            }
        }
    
    Exemples détection R1+ (vocabulaire interprétatif) :
        - GAM-004 × M1 → OSCILLATORY_SYSTEMATIC (fraction > 0.8)
        - GAM-009 × ASY-003 → CONTEXT_DEPENDENT (variance conditionnelle élevée)
        - Encodings SYM × Tests SYM-* → STRUCTURALLY_ALIGNED
    """
    # R0 : Implémentation minimale pour regime_concordance
    if metric == 'regime_concordance':
        # Détection axes (depuis metadata premier profil)
        axis_a = 'unknown'
        axis_b = 'unknown'
        
        if profiles_a:
            first_entity_a = list(profiles_a.values())[0]
            if 'tests' in first_entity_a and first_entity_a['tests']:
                first_test_a = list(first_entity_a['tests'].values())[0]
                # Chercher clé *_id
                for key in first_test_a.keys():
                    if key.endswith('_id') and key != 'test_name':
                        axis_a = key.replace('_id', '')
                        break
        
        if profiles_b:
            first_entity_b = list(profiles_b.values())[0]
            if 'tests' in first_entity_b and first_entity_b['tests']:
                first_test_b = list(first_entity_b['tests'].values())[0]
                for key in first_test_b.keys():
                    if key.endswith('_id') and key != 'test_name':
                        axis_b = key.replace('_id', '')
                        break
        
        # Calcul matrice concordance
        concordance_matrix = _compute_concordance_matrix(
            profiles_a, profiles_b, comparison_field='regime'
        )
        
        # Synthèse
        concordance_rates = [
            data['concordance_rate']
            for data in concordance_matrix.values()
        ]
        
        mean_concordance = float(np.mean(concordance_rates)) if concordance_rates else 0.0
        
        high_concordance_tests = [
            test_name for test_name, data in concordance_matrix.items()
            if data['concordance_rate'] > 0.8
        ]
        
        low_concordance_tests = [
            test_name for test_name, data in concordance_matrix.items()
            if data['concordance_rate'] < 0.3
        ]
        
        # Interaction détectée si variance concordance élevée
        variance_concordance = float(np.var(concordance_rates)) if concordance_rates else 0.0
        interaction_detected = variance_concordance > 0.1
        
        return {
            'axis_a': axis_a,
            'axis_b': axis_b,
            'metric': metric,
            'concordance_matrix': concordance_matrix,
            'summary': {
                'mean_concordance': mean_concordance,
                'variance_concordance': variance_concordance,
                'high_concordance_tests': high_concordance_tests,
                'low_concordance_tests': low_concordance_tests,
                'interaction_detected': interaction_detected
            }
        }
    
    else:
        # R1+ : Métriques non implémentées
        raise NotImplementedError(
            f"R1+ : Métrique '{metric}' nécessite interprétation qualitative. "
            f"Implémentation différée après validation R0."
        )


# ============================================================================
# INTERACTIONS MULTIWAY (R1+ placeholder)
# ============================================================================

def analyze_multiway_interactions(
    all_profiles: dict,
    combination: Optional[List[str]] = None
) -> dict:
    """
    Analyse interactions n-way (3+ axes).
    
    STATUT R0 : Placeholder complet (docstring détaillé)
    STATUT R1+ : Implémentation après validation pairwise
    
    Args:
        all_profiles: Tous résultats profiling
            {
                'gamma': gamma_profiles,
                'modifier': modifier_profiles,
                'encoding': encoding_profiles,
                'test': test_profiles
            }
        combination: Sous-ensemble axes analysés ou None (tous)
    
    Returns R1+ :
        {
            'combination': ['gamma', 'modifier', 'encoding'],
            'triplets': [
                {
                    'entities': ('GAM-004', 'M1', 'ASY-003'),
                    'tests_affected': ['TOP-001', 'SPE-002'],
                    'emergent_pattern': 'UNIQUE_BEHAVIOR',
                    'description': "Comportement impossible prédire depuis pairwise"
                },
                ...
            ],
            'summary': {
                'n_triplets_analyzed': int,
                'n_emergent_patterns': int,
                'interaction_strength': float
            }
        }
    
    Méthode anticipée :
        1. Énumérer combinaisons n-uplets
        2. Pour chaque n-uplet, comparer :
           - Comportement observé (profil conjoint)
           - Comportement prédit (agrégation pairwise)
        3. Écart > seuil → émergence détectée
    
    Exemple détection :
        GAM-004 × M1 → OSCILLATORY (pairwise)
        GAM-004 × ASY-003 → STABLE (pairwise)
        M1 × ASY-003 → STABLE (pairwise)
        
        Mais GAM-004 × M1 × ASY-003 → COLLAPSE (unique, non prédictible)
    """
    raise NotImplementedError(
        "R1+ : Analyse interactions multiway nécessite validation pairwise d'abord. "
        "Implémentation différée."
    )


# ============================================================================
# SIGNATURES GLOBALES (R1+ placeholder)
# ============================================================================

def detect_global_signatures(all_profiles: dict) -> dict:
    """
    Détection patterns émergents cross-axes avec vocabulaire interprétatif.
    
    STATUT R0 : Placeholder complet (vocabulaire défini)
    STATUT R1+ : Implémentation complète avec labels qualitatifs
    
    Args:
        all_profiles: Tous résultats profiling
    
    Returns R1+ :
        {
            'modifier_signatures': {
                'M0': {
                    'label': 'INVARIANT',
                    'pattern': 'Baseline neutre tous contextes',
                    'confidence': 'high'
                },
                'M1': {
                    'label': 'AMPLIFIED',
                    'pattern': 'Amplifie instabilités GAM-004/GAM-009',
                    'contexts_affected': ['GAM-004', 'GAM-009'],
                    'tests_systematic': ['TOP-001', 'SPE-002'],
                    'confidence': 'high'
                },
                'M2': {
                    'label': 'CONDITIONAL',
                    'pattern': 'Effet dépend encodings',
                    'sensitive_encodings': ['ASY-003'],
                    'confidence': 'medium'
                }
            },
            'encoding_signatures': {
                'R3-001': {
                    'label': 'INVARIANT',
                    'pattern': 'Comportement homogène tous gammas',
                    'confidence': 'high'
                },
                'ASY-003': {
                    'label': 'STRUCTURING',
                    'pattern': 'Induit patterns spécifiques',
                    'affected_gammas': ['GAM-007'],
                    'confidence': 'medium'
                }
            },
            'test_signatures': {
                'SYM-001': {
                    'label': 'DISCRIMINANT',
                    'pattern': 'Distingue encodings SYM vs ASY',
                    'discriminant_power': 0.85,
                    'confidence': 'high'
                },
                'PAT-001': {
                    'label': 'INVARIANT',
                    'pattern': 'Non discriminant (variance faible)',
                    'discriminant_power': 0.12,
                    'confidence': 'high'
                }
            },
            'cross_signatures': {
                'gamma_modifier': [
                    {
                        'pattern': 'GAM-004 sensible perturbations',
                        'systematic_modifiers': ['M1', 'M2'],
                        'label': 'FRAGILE'
                    }
                ],
                'encoding_test': [
                    {
                        'pattern': 'Tests SYM-* alignés encodings SYM',
                        'label': 'STRUCTURAL_ALIGNMENT'
                    }
                ]
            }
        }
    
    Vocabulaire qualitatif R1+ :
        - INVARIANT : Entité stable tous contextes
        - AMPLIFIED : Interaction renforce effet
        - SUPPRESSED : Interaction masque effet
        - CONDITIONAL : Effet dépend contexte
        - CONTEXTUAL : Émergence spécifique combinaison
        - DISCRIMINANT : Forte capacité différenciation
        - FRAGILE : Sensible perturbations
        - ROBUST : Résistant perturbations
        - STRUCTURAL_ALIGNMENT : Cohérence structure/observation
    
    Méthode anticipée :
        1. Analyser discriminant_power tous tests
        2. Analyser concordance pairwise tous axes
        3. Détecter patterns récurrents (heuristiques)
        4. Attribuer labels qualitatifs
        5. Calculer confidence (robustesse pattern)
    """
    raise NotImplementedError(
        "R1+ : Détection signatures globales avec vocabulaire interprétatif. "
        "Implémentation différée après validation analyses quantitatives R0."
    )
    
# tests/utilities/data_loading.py
"""
Data Loading Utilities - I/O Observations depuis DBs.

RESPONSABILITÉS :
- Connexions DB (prc_r0_results.db + prc_r0_raw.db)
- Fusion observations + métadonnées runs
- Conversion observations → DataFrame normalisé
- Cache observations (futur)

ARCHITECTURE :
- load_all_observations() : Double connexion (TestObservations + Executions)
- observations_to_dataframe() : Normalisation pour analyses stats

UTILISATEURS :
- verdict_engine.py (compute_verdict)
- verdict_reporter.py (generate_verdict_report)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def load_all_observations(
    params_config_id: str,
    db_results_path: str = './prc_automation/prc_database/prc_r0_results.db',
    db_raw_path: str = './prc_automation/prc_database/prc_r0_raw.db'
) -> List[Dict]:
    """
    Charge observations SUCCESS avec métadonnées runs.
    
    DOUBLE CONNEXION :
    - db_results : TestObservations (observation_data, status)
    - db_raw : Executions (gamma_id, d_encoding_id, modifier_id, seed)
    
    Args:
        params_config_id: ID config params (ex: 'params_default_v1')
        db_results_path: Chemin DB résultats tests
        db_raw_path: Chemin DB exécutions brutes
    
    Returns:
        List[dict]: Observations avec métadonnées fusionnées
        Format :
        {
            'observation_id': int,
            'exec_id': int,
            'run_id': str,
            'gamma_id': str,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int,
            'test_name': str,
            'params_config_id': str,
            'observation_data': dict,
            'computed_at': str
        }
    
    Raises:
        ValueError: Si aucune observation SUCCESS trouvée
    
    Examples:
        >>> obs = load_all_observations('params_default_v1')
        >>> len(obs)
        4320
        >>> obs[0]['gamma_id']
        'GAM-001'
    """
    # 1. Charger observations depuis db_results
    conn_results = sqlite3.connect(db_results_path)
    conn_results.row_factory = sqlite3.Row
    cursor_results = conn_results.cursor()
    
    cursor_results.execute("""
        SELECT 
            observation_id,
            exec_id,
            test_name,
            params_config_id,
            status,
            observation_data,
            computed_at
        FROM TestObservations
        WHERE params_config_id = ?
          AND status = 'SUCCESS'
    """, (params_config_id,))
    
    obs_rows = cursor_results.fetchall()
    conn_results.close()
    
    if not obs_rows:
        raise ValueError(
            f"Aucune observation SUCCESS pour params={params_config_id}"
        )
    
    # 2. Extraire exec_ids uniques
    exec_ids = list(set(row['exec_id'] for row in obs_rows))
    
    # 3. Charger métadonnées Executions depuis db_raw
    conn_raw = sqlite3.connect(db_raw_path)
    conn_raw.row_factory = sqlite3.Row
    cursor_raw = conn_raw.cursor()
    
    placeholders = ','.join('?' * len(exec_ids))
    cursor_raw.execute(f"""
        SELECT 
            id,
            run_id,
            gamma_id,
            d_encoding_id,
            modifier_id,
            seed
        FROM Executions
        WHERE id IN ({placeholders})
    """, exec_ids)
    
    exec_rows = cursor_raw.fetchall()
    conn_raw.close()
    
    # 4. Index executions par id
    executions_by_id = {
        row['id']: {
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_encoding_id': row['d_encoding_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed']
        }
        for row in exec_rows
    }
    
    # 5. Fusionner observations + métadonnées
    observations = []
    for row in obs_rows:
        exec_id = row['exec_id']
        
        if exec_id not in executions_by_id:
            print(f"⚠️ Skip observation {row['observation_id']}: "
                  f"exec_id={exec_id} introuvable dans db_raw")
            continue
        
        exec_meta = executions_by_id[exec_id]
        
        try:
            obs_data = json.loads(row['observation_data'])
            
            observations.append({
                'observation_id': row['observation_id'],
                'exec_id': exec_id,
                'run_id': exec_meta['run_id'],
                'gamma_id': exec_meta['gamma_id'],
                'd_encoding_id': exec_meta['d_encoding_id'],
                'modifier_id': exec_meta['modifier_id'],
                'seed': exec_meta['seed'],
                'test_name': row['test_name'],
                'params_config_id': row['params_config_id'],
                'observation_data': obs_data,
                'computed_at': row['computed_at']
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Skip observation {row['observation_id']}: {e}")
            continue
    
    return observations


def observations_to_dataframe(observations: List[Dict]) -> pd.DataFrame:
    """
    Convertit observations → DataFrame normalisé pour analyses stats.
    
    PROJECTIONS EXTRAITES :
    - value_final, value_initial, value_mean, value_std, value_min, value_max
    - slope, volatility, relative_change
    - transition, trend (catégorielles)
    
    Args:
        observations: Liste observations (retour load_all_observations)
    
    Returns:
        DataFrame avec colonnes :
        - Identifiants : gamma_id, d_encoding_id, modifier_id, seed, 
                        test_name, params_config_id, metric_name
        - Projections numériques : value_*, slope, volatility, relative_change
        - Catégorielles : transition, trend
    
    Notes:
        - Filtre lignes avec NaN dans TOUTES projections numériques
        - Une ligne par (observation, metric)
    
    Examples:
        >>> df = observations_to_dataframe(obs)
        >>> df.columns
        ['gamma_id', 'test_name', 'value_final', 'slope', ...]
        >>> df.shape
        (8640, 17)  # 4320 obs × 2 métriques moyennes
    """
    rows = []
    
    for obs in observations:
        gamma_id = obs['gamma_id']
        d_encoding_id = obs['d_encoding_id']
        modifier_id = obs['modifier_id']
        seed = obs['seed']
        test_name = obs['test_name']
        params_config_id = obs['params_config_id']
        
        obs_data = obs['observation_data']
        
        if 'statistics' not in obs_data or 'evolution' not in obs_data:
            continue
        
        stats = obs_data['statistics']
        evolution = obs_data['evolution']
        
        for metric_name in stats.keys():
            if metric_name not in evolution:
                continue
            
            metric_stats = stats[metric_name]
            metric_evol = evolution[metric_name]
            
            rows.append({
                # Identifiants
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id,
                'seed': seed,
                'test_name': test_name,
                'params_config_id': params_config_id,
                'metric_name': metric_name,
                
                # Projections numériques
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                
                # Catégorielles
                'transition': metric_evol.get('transition', 'unknown'),
                'trend': metric_evol.get('trend', 'unknown'),
            })
    
    df = pd.DataFrame(rows)
    
    # Nettoyer NaN (lignes sans aucune projection valide)
    numeric_cols = [
        'value_final', 'value_initial', 'value_mean', 'value_std',
        'slope', 'volatility', 'relative_change'
    ]
    df = df.dropna(subset=numeric_cols, how='all')
    
    return df


def cache_observations(
    observations: List[Dict],
    cache_path: str = './cache/observations.pkl'
) -> None:
    """
    Cache observations sur disque (pickle).
    
    FUTUR : Optimisation chargement répété.
    
    Args:
        observations: Liste observations
        cache_path: Chemin cache
    
    Examples:
        >>> cache_observations(obs, './cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(observations, f)
    
    print(f"✓ Cache observations : {cache_path}")


def load_cached_observations(cache_path: str) -> List[Dict]:
    """
    Charge observations depuis cache.
    
    Args:
        cache_path: Chemin cache
    
    Returns:
        Liste observations
    
    Raises:
        FileNotFoundError: Si cache absent
    
    Examples:
        >>> obs = load_cached_observations('./cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache non trouvé : {cache_path}")
    
    with open(cache_file, 'rb') as f:
        observations = pickle.load(f)
    
    print(f"✓ Chargé cache : {cache_path} ({len(observations)} observations)")
    return observations
    
# tests/utilities/discovery.py
"""
Discovery automatique tests actifs.

Architecture Charter 5.5 - Section 12.10
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict
import warnings

# Attributs requis architecture 5.5
REQUIRED_ATTRIBUTES = [
    'TEST_ID',
    'TEST_CATEGORY',
    'TEST_VERSION',
    'APPLICABILITY_SPEC',
    'COMPUTATION_SPECS',
]

def discover_active_tests() -> Dict[str, object]:
    """
    Découvre tous les tests actifs (non _deprecated).
    
    Returns:
        dict {test_id: module}
    
    Examples:
        >>> tests = discover_active_tests()
        >>> print(list(tests.keys()))
        ['UNIV-001', 'SYM-001', ...]
    """
    tests_dir = Path(__file__).parent.parent
    test_files = tests_dir.glob('test_*.py')
    
    active_tests = {}
    
    for test_file in test_files:
        # Skip deprecated
        if '_deprecated' in test_file.stem:
            continue
        
        # Charger module
        module_name = f'tests.{test_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed to import {module_name}: {e}")
            continue
        
        # Valider structure
        try:
            validate_test_structure(module)
            test_id = module.TEST_ID
            active_tests[test_id] = module
        except AssertionError as e:
            warnings.warn(f"Invalid test structure {module_name}: {e}")
            continue
    
    return active_tests


def validate_test_structure(module) -> None:
    """
    Valide qu'un module test a structure 5.4 conforme.
    
    Args:
        module: Module test importé
    
    Raises:
        AssertionError: Si structure invalide
    """
    # 1. Attributs requis
    for attr in REQUIRED_ATTRIBUTES:
        assert hasattr(module, attr), f"Missing required attribute: {attr}"
    
    # 2. Types corrects
    assert isinstance(module.TEST_ID, str), "TEST_ID must be str"
    assert isinstance(module.TEST_CATEGORY, str), "TEST_CATEGORY must be str"
    assert isinstance(module.TEST_VERSION, str), "TEST_VERSION must be str"
    assert isinstance(module.APPLICABILITY_SPEC, dict), "APPLICABILITY_SPEC must be dict"
    assert isinstance(module.COMPUTATION_SPECS, dict), "COMPUTATION_SPECS must be dict"
    
    # 3. Version 5.4 obligatoire
    assert module.TEST_VERSION == "5.5", f"TEST_VERSION must be '5.5', got '{module.TEST_VERSION}'"
    
    # 4. TEST_ID format CAT-NNN
    import re
    assert re.match(r'^[A-Z]{3,4}-\d{3}$', module.TEST_ID), \
        f"TEST_ID invalid format: {module.TEST_ID} (expected CAT-NNN)"
    
    # 5. COMPUTATION_SPECS non vide
    assert len(module.COMPUTATION_SPECS) > 0, "COMPUTATION_SPECS must not be empty"
    
    # 6. COMPUTATION_SPECS entre 1 et 5 métriques
    assert 1 <= len(module.COMPUTATION_SPECS) <= 5, \
        f"COMPUTATION_SPECS must have 1-5 metrics, got {len(module.COMPUTATION_SPECS)}"
    
    # 7. Chaque métrique a registry_key et default_params
    for metric_name, spec in module.COMPUTATION_SPECS.items():
        assert 'registry_key' in spec, \
            f"Metric '{metric_name}' missing 'registry_key'"
        assert 'default_params' in spec, \
            f"Metric '{metric_name}' missing 'default_params'"
        assert '.' in spec['registry_key'], \
            f"Metric '{metric_name}' registry_key must be 'registry.function' format"
    
    # 8. Pas de FORMULAS legacy
    assert not hasattr(module, 'FORMULAS'), \
        "FORMULAS attribute is obsolete in 5.4, use COMPUTATION_SPECS"
    
    # 9. Pas de is_applicable/compute_metric legacy
    assert not hasattr(module, 'is_applicable'), \
        "is_applicable() is obsolete in 5.5, use APPLICABILITY_SPEC"
    assert not hasattr(module, 'compute_metric'), \
        "compute_metric() is obsolete in 5.5, use COMPUTATION_SPECS with registries"
        
# tests/utilities/UTIL/profiling_common.py
"""
Profiling Common Module - Charter 6.1

ARCHITECTURE UNIFIÉE :
- Moteur générique profiling tous axes (gamma, modifier, encoding, test)
- API publique conventionnelle (profile_all_{axis}, compare_{axis}_summary)
- Délégation timeline_utils, aggregation_utils, regime_utils
- Zéro duplication code inter-axes

RESPONSABILITÉS :
- Agrégation signatures dynamiques
- Calcul profil PRC complet
- Profiling entités × tests (moteur générique)
- Comparaisons inter-entités
- API publique découvrable (8 fonctions : 4 axes × 2 fonctions)

AXES SUPPORTÉS :
- test (test_name) : Observations, pouvoir discriminant via cross_profiling
- gamma (gamma_id) : Mécanismes Γ
- modifier (modifier_id) : Perturbations D
- encoding (d_encoding_id) : Structure D

PHILOSOPHIE R0 :
- Format retour unifié strict (profiles, summary, metadata)
- Aucune extension spécifique axe (cross_profiling pour enrichissements)
- Découverte dynamique entités depuis observations
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# ============================================================================
# IMPORTS UTILITIES
# ============================================================================

# Timelines et événements dynamiques
from .timeline_utils import (
    extract_dynamic_events,
    compute_timeline_descriptor,
    TIMELINE_THRESHOLDS
)

# Agrégations statistiques
from .aggregation_utils import (
    aggregate_summary_metrics,
    aggregate_run_dispersion
)

# Classification régimes
from .regime_utils import (
    classify_regime
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Mapping axes → clés DB (normalisation stricte)
ENTITY_KEY_MAP = {
    'test': 'test_name',        # DB utilise 'test_name'
    'gamma': 'gamma_id',        # DB utilise 'gamma_id'
    'modifier': 'modifier_id',  # DB utilise 'modifier_id'
    'encoding': 'd_encoding_id' # DB utilise 'd_encoding_id' (pas d'alias)
}


# ============================================================================
# FONCTIONS COMMUNES (extraction gamma_profiling)
# ============================================================================

def aggregate_dynamic_signatures(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège signatures événements + timelines compositionnels.
    
    EXTRACTION : Identique gamma_profiling.aggregate_dynamic_signatures()
    RÉUTILISABLE : Tous axes profiling
    
    Utilise compute_timeline_descriptor() pour chaque run,
    puis agrège par Counter.
    
    Args:
        observations: Liste observations pour (entity, test) fixés
        metric_name: Nom métrique principale
    
    Returns:
        {
            'dynamic_signature': {
                'deviation_onset_median': float | None,
                'instability_onset_median': float | None,
                'oscillatory_fraction': float,
                'saturation_fraction': float,
                'collapse_fraction': float
            },
            'timeline_distribution': {
                'dominant_timeline': str,
                'timeline_confidence': float,
                'timeline_variants': {timeline: count, ...}
            }
        }
    """
    n_runs = len(observations)
    
    # Compteurs événements booléens
    oscillatory_count = 0
    saturation_count = 0
    collapse_count = 0
    
    # Onsets (pour médiane)
    deviation_onsets = []
    instability_onsets = []
    
    # Timelines compositionnels
    timelines = []
    
    for obs in observations:
        events = extract_dynamic_events(obs, metric_name)
        
        # Booléens
        if events.get('oscillatory'):
            oscillatory_count += 1
        if events.get('saturation'):
            saturation_count += 1
        if events.get('collapse'):
            collapse_count += 1
        
        # Onsets
        if events.get('deviation_onset') is not None:
            deviation_onsets.append(events['deviation_onset'])
        if events.get('instability_onset') is not None:
            instability_onsets.append(events['instability_onset'])
        
        # Timeline descriptor
        timeline_desc = compute_timeline_descriptor(
            events['sequence'],
            events['sequence_timing_relative'],
            events.get('oscillatory_global', False)
        )
        timelines.append(timeline_desc['timeline_compact'])
    
    # Fractions
    oscillatory_frac = oscillatory_count / n_runs if n_runs > 0 else 0.0
    saturation_frac = saturation_count / n_runs if n_runs > 0 else 0.0
    collapse_frac = collapse_count / n_runs if n_runs > 0 else 0.0
    
    # Onsets médianes (absolus, pour info)
    deviation_onset_median = float(np.median(deviation_onsets)) if deviation_onsets else None
    instability_onset_median = float(np.median(instability_onsets)) if instability_onsets else None
    
    # Timeline dominante
    counter = Counter(timelines)
    if counter:
        dominant, count = counter.most_common(1)[0]
        confidence = count / n_runs
    else:
        dominant = 'no_significant_dynamics'
        confidence = 0.0
    
    return {
        'dynamic_signature': {
            'deviation_onset_median': deviation_onset_median,
            'instability_onset_median': instability_onset_median,
            'oscillatory_fraction': oscillatory_frac,
            'saturation_fraction': saturation_frac,
            'collapse_fraction': collapse_frac
        },
        'timeline_distribution': {
            'dominant_timeline': dominant,
            'timeline_confidence': confidence,
            'timeline_variants': dict(counter)
        }
    }


def compute_prc_profile(
    metrics: dict,
    dynamic_sig: dict,
    timeline_dist: dict,
    dispersion: dict,
    n_runs: int,
    n_valid: int,
    test_name: str
) -> dict:
    """
    Génère profil PRC complet avec confidence heuristique.
    
    EXTRACTION : Identique gamma_profiling.compute_prc_profile()
    MODIFICATION : Suppression parameter extensions (pas de cas d'usage R0)
    RÉUTILISABLE : Tous axes profiling
    
    Args:
        metrics: Métriques agrégées (aggregate_summary_metrics)
        dynamic_sig: Signature dynamique (aggregate_dynamic_signatures)
        timeline_dist: Distribution timelines (aggregate_dynamic_signatures)
        dispersion: Dispersion inter-runs (aggregate_run_dispersion)
        n_runs: Nombre total runs
        n_valid: Nombre runs SUCCESS
        test_name: ID test
    
    Returns:
        {
            'regime': str,
            'behavior': str,
            'dominant_timeline': {...},
            'robustness': {...},
            'pathologies': {...},
            'n_runs': int,
            'n_valid': int,
            'confidence': str,
            'confidence_metadata': {...}
        }
    """
    if not metrics:
        return {
            'regime': 'NO_DATA',
            'behavior': 'unknown',
            'n_runs': n_runs,
            'n_valid': n_valid,
            'confidence': 'none',
            'confidence_metadata': {}
        }
    
    # Régime (avec qualificatif MIXED si applicable)
    regime = classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name)
    
    # Behavior
    base_regime = regime.split('::')[-1] if '::' in regime else regime
    
    if base_regime.startswith('CONSERVES_'):
        behavior = 'stable'
    elif base_regime in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE']:
        behavior = 'unstable'
    elif base_regime == 'DEGRADING':
        behavior = 'degrading'
    elif base_regime == 'TRIVIAL':
        behavior = 'stable'  # Techniquement stable (pas de dynamique)
    elif base_regime == 'SATURATES_HIGH':
        behavior = 'stable'  # Converge vers plateau
    else:
        behavior = 'uncategorized'
    
    # Qualificatif MIXED
    if regime.startswith('MIXED::'):
        behavior = 'mixed'
    
    # Timeline dominante
    dominant_timeline = timeline_dist['dominant_timeline']
    timeline_confidence = timeline_dist['timeline_confidence']
    
    # Robustness
    robustness = {
        'homogeneous': not dispersion['bimodal_detected'] and dispersion['cv_across_runs'] < 0.5,
        'mixed_behavior': dispersion['bimodal_detected'],
        'numerically_stable': dynamic_sig['collapse_fraction'] < 0.1 and 
                              dynamic_sig.get('instability_onset_median') is None
    }
    
    # Pathologies
    pathologies = {
        'numeric_instability': 'NUMERIC_INSTABILITY' in regime,
        'oscillatory': dynamic_sig['oscillatory_fraction'] > 0.3,
        'collapse': dynamic_sig['collapse_fraction'] > 0.1,
        'trivial': base_regime == 'TRIVIAL',
        'degrading': base_regime == 'DEGRADING'
    }
    
    # Confidence heuristique
    confidence_criteria = {
        'min_runs_high': 20,
        'min_runs_medium': 10,
        'max_cv_homogeneous': 0.5,
        'min_timeline_confidence': 0.7
    }
    
    if (n_valid >= confidence_criteria['min_runs_high'] and 
        not dispersion['bimodal_detected'] and
        timeline_confidence >= confidence_criteria['min_timeline_confidence']):
        confidence = 'high'
        rationale = f"n_valid={n_valid}, bimodal=False, timeline_conf={timeline_confidence:.2f}"
    elif n_valid >= confidence_criteria['min_runs_medium']:
        confidence = 'medium'
        rationale = f"n_valid={n_valid}"
    else:
        confidence = 'low'
        rationale = f"n_valid={n_valid} < {confidence_criteria['min_runs_medium']}"
    
    return {
        'regime': regime,
        'behavior': behavior,
        'dominant_timeline': {
            'timeline_compact': dominant_timeline,
            'confidence': timeline_confidence,
            'variants': timeline_dist['timeline_variants']
        },
        'robustness': robustness,
        'pathologies': pathologies,
        'n_runs': n_runs,
        'n_valid': n_valid,
        'confidence': confidence,
        'confidence_metadata': {
            'level': confidence,
            'criteria': confidence_criteria,
            'rationale': rationale
        }
    }


# ============================================================================
# MOTEUR GÉNÉRIQUE (privé)
# ============================================================================

def _profile_test_for_entity(
    observations: List[dict],
    test_name: str,
    entity_id: str,
    axis: str
) -> dict:
    """
    Profil UN test sous UNE entité.
    
    GÉNÉRALISATION : gamma_profiling.profile_test_for_gamma()
    MOTEUR INTERNE : Logique partagée tous axes
    
    Args:
        observations: Observations pour (entity_id, test_name) fixés
        test_name: ID test (ex: 'SYM-001')
        entity_id: ID entité (ex: 'GAM-001', 'M1', 'SYM-001')
        axis: Nom axe ('gamma', 'modifier', 'encoding', 'test')
    
    Returns:
        {
            'test_name': str,
            '{axis}_id': str,  # Clé dynamique selon axe
            'prc_profile': {...},
            'diagnostic_signature': {...},
            'instrumentation': {...}
        }
    """
    if not observations:
        return {
            'test_name': test_name,
            f'{axis}_id': entity_id,
            'prc_profile': {
                'regime': 'NO_DATA',
                'behavior': 'unknown',
                'n_runs': 0,
                'n_valid': 0,
                'confidence': 'none'
            },
            'diagnostic_signature': {},
            'instrumentation': {}
        }
    
    # Métrique principale
    first_obs = observations[0]
    obs_data = first_obs.get('observation_data', {})
    stats = obs_data.get('statistics', {})
    
    metric_name = list(stats.keys())[0] if stats else 'unknown'
    
    n_runs = len(observations)
    n_valid = len([o for o in observations if o.get('status') == 'SUCCESS'])
    
    # Vérifier présence dynamic_events
    dynamic_events_present = 'dynamic_events' in obs_data
    timeseries_present = 'timeseries' in obs_data
    
    # NIVEAU 3 : Instrumentation
    summary_metrics = aggregate_summary_metrics(observations, metric_name)
    
    instrumentation = {
        'metric_name': metric_name,
        'summary_metrics': summary_metrics,
        'data_completeness': {
            'dynamic_events_present': dynamic_events_present,
            'timeseries_present': timeseries_present,
            'fallback_used': not dynamic_events_present
        },
        'computation_metadata': {
            'profiling_version': '6.1',
            'profiling_module': 'profiling_common',
            'profiling_axis': axis,
            'timeline_architecture': 'compositional_relative',
            'n_runs': n_runs,
            'n_valid': n_valid
        }
    }
    
    # NIVEAU 2 : Diagnostic signature
    run_dispersion = aggregate_run_dispersion(observations, metric_name)
    event_aggregates = aggregate_dynamic_signatures(observations, metric_name)
    
    diagnostic_signature = {
        'dynamic_signature': event_aggregates['dynamic_signature'],
        'timeline_distribution': event_aggregates['timeline_distribution'],
        'run_dispersion': run_dispersion,
        'thresholds_used': {
            'timeline_early': TIMELINE_THRESHOLDS['early'],
            'timeline_mid': TIMELINE_THRESHOLDS['mid'],
            'timeline_late': TIMELINE_THRESHOLDS['late'],
            'instability_detection': 'P90 * 10 (relative)',
            'oscillatory_threshold': '10% sign changes',
            'saturation_cv': '5%',
            'bimodal_iqr_ratio': '3.0'
        }
    }
    
    # NIVEAU 1 : PRC Profile
    prc_profile = compute_prc_profile(
        summary_metrics,
        event_aggregates['dynamic_signature'],
        event_aggregates['timeline_distribution'],
        run_dispersion,
        n_runs,
        n_valid,
        test_name
    )
    
    return {
        'test_name': test_name,
        f'{axis}_id': entity_id,
        'prc_profile': prc_profile,
        'diagnostic_signature': diagnostic_signature,
        'instrumentation': instrumentation
    }


def _profile_entity_axis(
    observations: List[dict],
    axis: str,
    entity_key: str
) -> dict:
    """
    Moteur générique profiling tous axes.
    
    GÉNÉRALISATION : gamma_profiling.profile_all_gammas()
    CŒUR ARCHITECTURE : Point central unification
    
    Args:
        observations: Toutes observations (découverte dynamique entités)
        axis: Nom axe ('gamma', 'modifier', 'encoding', 'test')
        entity_key: Clé DB selon axe (depuis ENTITY_KEY_MAP)
    
    Returns:
        {
            'entity_id_1': {
                'tests': {
                    'test_name_1': {...},  # Résultat _profile_test_for_entity
                    'test_name_2': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            'entity_id_2': {...},
            ...
        }
    """
    profiles = {}
    
    # Groupement observations par entité (découverte dynamique)
    obs_by_entity = defaultdict(list)
    for obs in observations:
        entity_id = obs.get(entity_key)
        if entity_id:
            obs_by_entity[entity_id].append(obs)
    
    # Profiling chaque entité
    for entity_id, entity_obs in obs_by_entity.items():
        # Groupement observations par test
        obs_by_test = defaultdict(list)
        for obs in entity_obs:
            test_name = obs.get('test_name')
            if test_name:
                obs_by_test[test_name].append(obs)
        
        # Profiling chaque test
        test_profiles = {}
        for test_name, test_obs in obs_by_test.items():
            profile = _profile_test_for_entity(
                test_obs, test_name, entity_id, axis
            )
            test_profiles[test_name] = profile
        
        profiles[entity_id] = {
            'tests': test_profiles,
            'n_tests': len(test_profiles),
            'n_total_runs': len(entity_obs)
        }
    
    return profiles


def _compare_entities_summary(profiles: dict, axis: str) -> dict:
    """
    Comparaisons cross-entities.
    
    GÉNÉRALISATION : gamma_profiling.compare_gammas_summary()
    
    Args:
        profiles: Résultats _profile_entity_axis()
        axis: Nom axe
    
    Returns:
        {
            'by_regime': {
                'regime_name': ['entity_id_1', 'entity_id_2', ...],
                ...
            },
            'by_test': {
                'test_name': {
                    'best_conservation': 'entity_id',
                    'worst_conservation': 'entity_id',
                    'ranking': ['entity_id_1', ...]
                },
                ...
            }
        }
    """
    by_regime = defaultdict(list)
    
    # Groupement par régime dominant
    for entity_id, entity_data in profiles.items():
        regime_counts = Counter()
        for test_profile in entity_data['tests'].values():
            regime = test_profile['prc_profile']['regime']
            regime_counts[regime] += 1
        
        if regime_counts:
            dominant_regime = regime_counts.most_common(1)[0][0]
            by_regime[dominant_regime].append(entity_id)
    
    # Rankings par test
    by_test = {}
    
    all_tests = set()
    for entity_data in profiles.values():
        all_tests.update(entity_data['tests'].keys())
    
    for test_name in all_tests:
        # Scoring conservation simple (aligné gamma_profiling)
        scores = []
        
        for entity_id, entity_data in profiles.items():
            test_profile = entity_data['tests'].get(test_name)
            
            if not test_profile:
                continue
            
            prc = test_profile['prc_profile']
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            
            # Score conservation
            if regime.startswith('CONSERVES_'):
                score = 1.0
            elif regime == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
            
            scores.append((entity_id, score))
        
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            
            by_test[test_name] = {
                'best_conservation': scores[0][0],
                'worst_conservation': scores[-1][0],
                'ranking': [eid for eid, _ in scores]
            }
    
    return {
        'by_regime': dict(by_regime),
        'by_test': by_test
    }


# ============================================================================
# API PUBLIQUE (conventions naming - découvrable)
# ============================================================================

def profile_all_tests(observations: List[dict]) -> dict:
    """
    Profil comportemental tests individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'test_id': {
                'tests': {...},  # Note: "tests" contient ici entities observées
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    
    Note R0:
        Pouvoir discriminant tests calculable via cross_profiling
        (variance inter-entities, effect_size, etc.)
        Pas d'enrichissement spécifique axe test pour R0.
    """
    return _profile_entity_axis(observations, 'test', 'test_name')


def compare_tests_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-tests.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_tests()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'test')


def profile_all_gammas(observations: List[dict]) -> dict:
    """
    Profil comportemental gammas individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'GAM-001': {
                'tests': {
                    'SYM-001': {...},
                    'SPE-001': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    """
    return _profile_entity_axis(observations, 'gamma', 'gamma_id')


def compare_gammas_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-gammas.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_gammas()
    
    Returns:
        {
            'by_regime': {
                'CONSERVES_SYMMETRY': ['GAM-001', 'GAM-004'],
                ...
            },
            'by_test': {
                'SYM-001': {
                    'best_conservation': 'GAM-001',
                    'worst_conservation': 'GAM-013',
                    'ranking': ['GAM-001', ...]
                },
                ...
            }
        }
    """
    return _compare_entities_summary(profiles, 'gamma')


def profile_all_modifiers(observations: List[dict]) -> dict:
    """
    Profil comportemental modifiers individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'M0': {
                'tests': {
                    'SYM-001': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            'M1': {...},
            'M2': {...}
        }
    """
    return _profile_entity_axis(observations, 'modifier', 'modifier_id')


def compare_modifiers_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-modifiers.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_modifiers()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'modifier')


def profile_all_encodings(observations: List[dict]) -> dict:
    """
    Profil comportemental encodings individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'SYM-001': {
                'tests': {...},
                'n_tests': int,
                'n_total_runs': int
            },
            'ASY-001': {...},
            'R3-001': {...},
            ...
        }
    """
    return _profile_entity_axis(observations, 'encoding', 'd_encoding_id')


def compare_encodings_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-encodings.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_encodings()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'encoding')
    
# tests/utilities/regime_utils.py
"""
Regime Utilities - Stratification et classification régimes.

RESPONSABILITÉS :
- Stratification observations (stable/explosif)
- Classification régime comportemental
- Détection propriétés conservées
- Taxonomie régimes (CONSERVES_X, pathologies)

ARCHITECTURE :
- stratify_by_regime() : Séparation stable/explosif
- classify_regime() : Régime spécifique par test
- detect_conserved_property() : Déduction propriété

PRINCIPE R0 :
- Régimes SPÉCIFIQUES (CONSERVES_SYMMETRY vs CONSERVES_NORM)
- Qualificatif MIXED:: pour multimodalité
- Taxonomie extensible (nouveaux régimes faciles à ajouter)

UTILISATEURS :
- verdict_engine.py (stratification globale)
- gamma_profiling.py (classification individuelle)
"""

import numpy as np
from typing import List, Dict, Tuple


# =============================================================================
# STRATIFICATION STABLE/EXPLOSIF
# =============================================================================

def stratify_by_regime(
    observations: List[Dict],
    threshold: float = 1e50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratifie observations en régimes stable/explosif.
    
    Critère : présence valeurs >threshold dans projections exploitées.
    Conserve TOUTES observations (aucun filtrage).
    
    Args:
        observations: Liste observations complètes
        threshold: Seuil magnitude extrême (défaut 1e50)
    
    Returns:
        (obs_stable, obs_explosif)
    
    Examples:
        >>> stable, explosif = stratify_by_regime(observations)
        >>> len(stable) / len(observations)
        0.85  # 85% stable
        >>> len(explosif) / len(observations)
        0.15  # 15% explosif
    """
    stable = []
    explosif = []
    
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {})
        evol = obs_data.get('evolution', {})
        
        has_extreme = False
        
        # Vérifier toutes projections exploitées
        for metric_stats in stats.values():
            for key in ['initial', 'final', 'mean', 'max']:
                val = metric_stats.get(key)
                if val is not None and abs(val) > threshold:
                    has_extreme = True
                    break
            if has_extreme:
                break
        
        if not has_extreme:
            for metric_evol in evol.values():
                for key in ['slope', 'relative_change']:
                    val = metric_evol.get(key)
                    if val is not None and abs(val) > threshold:
                        has_extreme = True
                        break
                if has_extreme:
                    break
        
        if has_extreme:
            explosif.append(obs)
        else:
            stable.append(obs)
    
    return stable, explosif


# =============================================================================
# CLASSIFICATION RÉGIME
# =============================================================================

def classify_regime(
    metrics: Dict,
    dynamic_sig: Dict,
    timeline_dist: Dict,
    dispersion: Dict,
    test_name: str
) -> str:
    """
    Classification régime R0 avec régimes SPÉCIFIQUES.
    
    Au lieu de CONSERVES_X générique, on détecte :
    - CONSERVES_SYMMETRY (SYM-*)
    - CONSERVES_NORM (SPE-*, UNIV-*)
    - CONSERVES_PATTERN (PAT-*)
    - CONSERVES_TOPOLOGY (TOP-*)
    - CONSERVES_GRADIENT (GRA-*)
    - CONSERVES_SPECTRUM (SPA-*)
    
    Régimes pathologiques :
    - NUMERIC_INSTABILITY, OSCILLATORY_UNSTABLE, TRIVIAL, 
      DEGRADING, SATURATES_HIGH, UNCATEGORIZED
    
    Si bimodal détecté → MIXED::{régime_base}
    
    Args:
        metrics: Retour aggregate_summary_metrics()
        dynamic_sig: Signatures dynamiques
        timeline_dist: Distribution timelines
        dispersion: Retour aggregate_run_dispersion()
        test_name: Nom du test (pour déduire propriété)
    
    Returns:
        str: Régime (ex: 'CONSERVES_SYMMETRY', 'MIXED::CONSERVES_NORM')
    
    Examples:
        >>> regime = classify_regime(metrics, dyn_sig, timeline, disp, 'SYM-001')
        >>> regime
        'CONSERVES_SYMMETRY'
        
        >>> regime = classify_regime(metrics, dyn_sig, timeline, disp, 'SPE-002')
        >>> regime
        'MIXED::CONSERVES_NORM'  # Multimodal
    """
    if not metrics:
        return "NO_DATA"
    
    final = metrics['final_value']['median']
    initial = metrics['initial_value']
    cv = metrics['cv']
    
    instability_onset = dynamic_sig.get('instability_onset_median')
    
    # PATHOLOGIES (prioritaires)
    if instability_onset is not None and instability_onset < 20 and final > 1e20:
        base_regime = "NUMERIC_INSTABILITY"
    elif dynamic_sig['oscillatory_fraction'] > 0.3:
        base_regime = "OSCILLATORY_UNSTABLE"
    elif cv < 0.01:
        base_regime = "TRIVIAL"
    
    # CONSERVATION (dépend du test)
    elif final < 2 * initial and cv < 0.1:
        base_regime = detect_conserved_property(test_name)
    
    # SATURATION
    elif 'saturation' in timeline_dist.get('dominant_timeline', '') and dynamic_sig['saturation_fraction'] > 0.7:
        if final > 10 * initial:
            base_regime = "SATURATES_HIGH"
        else:
            # Saturation mais pas croissance → Conservation
            base_regime = detect_conserved_property(test_name)
    
    # DEGRADING
    elif final < 0.5 * initial and dynamic_sig['collapse_fraction'] < 0.1:
        base_regime = "DEGRADING"
    
    else:
        base_regime = "UNCATEGORIZED"
    
    # Qualificatif multimodalité
    if dispersion['bimodal_detected']:
        return f"MIXED::{base_regime}"
    
    return base_regime


def detect_conserved_property(test_name: str) -> str:
    """
    Détermine propriété conservée selon préfixe test.
    
    Args:
        test_name: ID test (ex: 'SYM-001', 'SPE-002')
    
    Returns:
        str: Régime conservation spécifique
    
    Mapping :
        SYM-* → CONSERVES_SYMMETRY
        SPE-*, UNIV-* → CONSERVES_NORM
        PAT-* → CONSERVES_PATTERN
        TOP-* → CONSERVES_TOPOLOGY
        GRA-* → CONSERVES_GRADIENT
        SPA-* → CONSERVES_SPECTRUM
        Autre → CONSERVES_PROPERTY (fallback générique)
    
    Examples:
        >>> detect_conserved_property('SYM-001')
        'CONSERVES_SYMMETRY'
        >>> detect_conserved_property('UNIV-002')
        'CONSERVES_NORM'
    """
    if test_name.startswith('SYM-'):
        return "CONSERVES_SYMMETRY"
    elif test_name.startswith('SPE-') or test_name.startswith('UNIV-'):
        return "CONSERVES_NORM"
    elif test_name.startswith('PAT-'):
        return "CONSERVES_PATTERN"
    elif test_name.startswith('TOP-'):
        return "CONSERVES_TOPOLOGY"
    elif test_name.startswith('GRA-'):
        return "CONSERVES_GRADIENT"
    elif test_name.startswith('SPA-'):
        return "CONSERVES_SPECTRUM"
    else:
        return "CONSERVES_PROPERTY"  # Fallback générique


# =============================================================================
# EXTRACTION PROPRIÉTÉS CONSERVÉES
# =============================================================================

def extract_conserved_properties(profile: Dict) -> List[str]:
    """
    Extrait propriétés conservées depuis profil gamma.
    
    Args:
        profile: Profil gamma (avec tests)
    
    Returns:
        list[str]: Propriétés conservées
    
    Examples:
        >>> props = extract_conserved_properties(gamma_profile)
        >>> props
        ['Symétrie', 'Norme', 'Pattern']
    """
    properties = set()
    
    for test_data in profile.get('tests', {}).values():
        regime = test_data.get('regime', '')
        
        if 'CONSERVES_SYMMETRY' in regime:
            properties.add('Symétrie')
        elif 'CONSERVES_NORM' in regime:
            properties.add('Norme')
        elif 'CONSERVES_PATTERN' in regime:
            properties.add('Pattern')
        elif 'CONSERVES_TOPOLOGY' in regime:
            properties.add('Topologie')
        elif 'CONSERVES_GRADIENT' in regime:
            properties.add('Gradient')
        elif 'CONSERVES_SPECTRUM' in regime:
            properties.add('Spectre')
    
    return sorted(properties)


# =============================================================================
# TAXONOMIE RÉGIMES (RÉFÉRENCE)
# =============================================================================

REGIME_TAXONOMY = {
    # Conservation (sains)
    'CONSERVES_SYMMETRY': {
        'family': 'conservation',
        'description': 'Asymétrie finale < 1e-6',
        'tests': ['SYM-*']
    },
    'CONSERVES_NORM': {
        'family': 'conservation',
        'description': 'Norme finale < 2× initiale',
        'tests': ['SPE-*', 'UNIV-*']
    },
    'CONSERVES_PATTERN': {
        'family': 'conservation',
        'description': 'Diversity/uniformity stables',
        'tests': ['PAT-*']
    },
    'CONSERVES_TOPOLOGY': {
        'family': 'conservation',
        'description': 'Euler characteristic stable',
        'tests': ['TOP-*']
    },
    'CONSERVES_GRADIENT': {
        'family': 'conservation',
        'description': 'Structure gradients conservée',
        'tests': ['GRA-*']
    },
    'CONSERVES_SPECTRUM': {
        'family': 'conservation',
        'description': 'Spectre valeurs propres stable',
        'tests': ['SPA-*']
    },
    
    # Pathologies
    'NUMERIC_INSTABILITY': {
        'family': 'pathology',
        'description': 'Instability onset < 20 && final > 1e20',
        'tests': ['Tous']
    },
    'OSCILLATORY_UNSTABLE': {
        'family': 'pathology',
        'description': 'Oscillatory fraction > 30%',
        'tests': ['Tous']
    },
    'TRIVIAL': {
        'family': 'pathology',
        'description': 'CV < 1% (aucune variation)',
        'tests': ['Tous']
    },
    'DEGRADING': {
        'family': 'pathology',
        'description': 'Final < 0.5 × initial (sans collapse)',
        'tests': ['Tous']
    },
    
    # Autres
    'SATURATES_HIGH': {
        'family': 'saturation',
        'description': 'Saturation + final > 10 × initial',
        'tests': ['Tous']
    },
    'UNCATEGORIZED': {
        'family': 'other',
        'description': 'Comportement non classifié',
        'tests': ['Tous']
    }
}


def get_regime_family(regime: str) -> str:
    """
    Retourne famille d'un régime.
    
    Args:
        regime: Nom régime (avec ou sans MIXED::)
    
    Returns:
        'conservation' | 'pathology' | 'saturation' | 'other'
    
    Examples:
        >>> get_regime_family('CONSERVES_SYMMETRY')
        'conservation'
        >>> get_regime_family('MIXED::CONSERVES_NORM')
        'conservation'
    """
    # Strip qualificatif MIXED::
    base_regime = regime.split('::')[-1] if '::' in regime else regime
    
    return REGIME_TAXONOMY.get(base_regime, {}).get('family', 'other')
    
# tests/utilities/report_writers.py
"""
Report Writers - Formatage et écriture rapports structurés.

RESPONSABILITÉS :
- Écriture JSON structurés
- Formatage TXT lisibles humains
- Génération CSVs analyse
- Helpers formatage (sections, tableaux)

ARCHITECTURE :
- write_json() : JSON avec indent
- write_summary_section() : Formatage sections TXT
- write_regime_synthesis() : Synthèse régimes
- write_timeline_signatures() : Signatures dynamiques

PRINCIPE R0 :
- Séparation calcul/formatting (modules analytiques ≠ writers)
- Réutilisabilité formatters (verdict, modifier, test profiling)
- Structure rapports standardisée

UTILISATEURS :
- verdict_reporter.py (rapports complets)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


# =============================================================================
# ÉCRITURE JSON
# =============================================================================

def write_json(
    data: Dict,
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Écrit dict → JSON formaté.
    
    Args:
        data: Données à sérialiser
        filepath: Chemin fichier sortie
        indent: Indentation (défaut 2)
    
    Examples:
        >>> write_json({'key': 'value'}, Path('output.json'))
    """
    # Conversion tuples → strings pour sérialisation
    data_serializable = _make_json_serializable(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_serializable, f, indent=indent)


def _make_json_serializable(obj):
    """
    Convertit récursivement tuples en strings pour JSON.
    
    Problème : Clés tuple (test, metric, proj) non JSON-serializable.
    Solution : tuple → "test|metric|proj" string.
    """
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): _make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return "|".join(str(x) for x in obj)
    else:
        return obj


# =============================================================================
# FORMATAGE SECTIONS TXT
# =============================================================================

def write_header(
    f,
    title: str,
    width: int = 80,
    char: str = '='
) -> None:
    """
    Écrit header section.
    
    Args:
        f: File handle
        title: Titre section
        width: Largeur ligne
        char: Caractère bordure
    
    Examples:
        >>> with open('report.txt', 'w') as f:
        ...     write_header(f, 'METADATA')
    """
    f.write(char * width + "\n")
    f.write(f"{title}\n")
    f.write(char * width + "\n\n")


def write_subheader(
    f,
    title: str,
    width: int = 80,
    char: str = '-'
) -> None:
    """
    Écrit sous-header section.
    
    Args:
        f: File handle
        title: Titre sous-section
        width: Largeur ligne
        char: Caractère bordure
    """
    f.write(f"{title}\n")
    f.write(char * width + "\n")


def write_key_value(
    f,
    key: str,
    value: Any,
    indent: int = 0
) -> None:
    """
    Écrit paire clé-valeur.
    
    Args:
        f: File handle
        key: Clé
        value: Valeur
        indent: Niveau indentation (espaces)
    
    Examples:
        >>> write_key_value(f, 'Total observations', 4320)
        Total observations    : 4320
    """
    prefix = " " * indent
    f.write(f"{prefix}{key:25s} : {value}\n")


# =============================================================================
# SYNTHÈSE RÉGIMES (TRANSVERSALE)
# =============================================================================

def write_regime_synthesis(
    f,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit synthèse régimes transversale.
    
    Agrège régimes sur tous gammas × tests, groupés par famille.
    
    Args:
        f: File handle
        gamma_profiles: Profils gammas formatés
        width: Largeur section
    
    Structure :
        CONSERVATION (régimes sains):
          CONSERVES_SYMMETRY : 120 (12.5%)
          ...
        PATHOLOGIES:
          NUMERIC_INSTABILITY : 45 (4.7%)
          ...
    """
    write_header(f, "SYNTHÈSE RÉGIMES (vue transversale)", width)
    
    # Compter régimes globalement
    regime_counter = defaultdict(int)
    for gamma_data in gamma_profiles.values():
        for test_data in gamma_data.get('tests', {}).values():
            regime = test_data.get('regime', '')
            regime_counter[regime] += 1
    
    total = sum(regime_counter.values())
    
    f.write(f"Total profils : {total}\n\n")
    
    # Grouper par famille
    conservation = {k: v for k, v in regime_counter.items() if 'CONSERVES_' in k}
    pathologies = {k: v for k, v in regime_counter.items() if k in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE', 'TRIVIAL', 'DEGRADING']}
    mixed = {k: v for k, v in regime_counter.items() if k.startswith('MIXED::')}
    other = {k: v for k, v in regime_counter.items() if k not in conservation and k not in pathologies and k not in mixed}
    
    if conservation:
        f.write("CONSERVATION (régimes sains):\n")
        for regime, count in sorted(conservation.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total conservation : {sum(conservation.values())} ({sum(conservation.values())/total*100:.1f}%)\n\n")
    
    if pathologies:
        f.write("PATHOLOGIES:\n")
        for regime, count in sorted(pathologies.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total pathologies : {sum(pathologies.values())} ({sum(pathologies.values())/total*100:.1f}%)\n\n")
    
    if mixed:
        f.write("MULTIMODALITÉ (MIXED::X):\n")
        for regime, count in sorted(mixed.items(), key=lambda x: -x[1])[:5]:
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total multimodal : {sum(mixed.values())} ({sum(mixed.values())/total*100:.1f}%)\n\n")


# =============================================================================
# SIGNATURES DYNAMIQUES
# =============================================================================

def write_dynamic_signatures(
    f,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit signatures dynamiques par gamma.
    
    Args:
        f: File handle
        gamma_profiles: Profils gammas formatés
        width: Largeur section
    
    Structure :
        GAM-001:
          Timeline dominante : early_deviation_then_saturation (75% tests)
          Variantes (3 timelines distinctes):
            - mid_instability_then_collapse (8 tests)
    """
    write_header(f, "SIGNATURES DYNAMIQUES PAR GAMMA", width)
    
    for gamma_id in sorted(gamma_profiles.keys()):
        gamma_data = gamma_profiles[gamma_id]
        
        # Compter timelines dominantes
        timeline_counter = defaultdict(int)
        for test_data in gamma_data.get('tests', {}).values():
            timeline = test_data.get('timeline', 'unknown')
            timeline_counter[timeline] += 1
        
        # Timeline dominante
        if timeline_counter:
            dominant_timeline, count = max(timeline_counter.items(), key=lambda x: x[1])
            confidence = count / len(gamma_data.get('tests', {}))
            
            f.write(f"\n{gamma_id}:\n")
            f.write(f"  Timeline dominante : {dominant_timeline} ({confidence*100:.0f}% tests)\n")
            
            # Diversité timelines
            if len(timeline_counter) > 1:
                f.write(f"  Variantes ({len(timeline_counter)} timelines distinctes):\n")
                for tl, cnt in sorted(timeline_counter.items(), key=lambda x: -x[1])[:3]:
                    if tl != dominant_timeline:
                        f.write(f"    - {tl} ({cnt} tests)\n")
    
    f.write("\n")


# =============================================================================
# COMPARAISONS ENRICHIES (PAR PROPRIÉTÉ)
# =============================================================================

def write_comparisons_enriched(
    f,
    comparisons: Dict,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit comparaisons enrichies avec contexte propriétés.
    
    Args:
        f: File handle
        comparisons: Retour compare_gammas_summary()
        gamma_profiles: Profils gammas (pour extraction propriétés)
        width: Largeur section
    
    Structure :
        SYMÉTRIE:
          SYM-001:
            Meilleur : GAM-001
            Pire     : GAM-013
            Classement : GAM-001, GAM-004, GAM-007...
    """
    write_header(f, "COMPARISONS INTER-GAMMAS (par propriété)", width)
    
    # Grouper tests par propriété
    tests_by_property = {
        'Symétrie': ['SYM-001'],
        'Norme': ['SPE-001', 'SPE-002', 'UNIV-001', 'UNIV-002'],
        'Pattern': ['PAT-001'],
        'Topologie': ['TOP-001'],
        'Gradient': ['GRA-001'],
        'Spectre': ['SPA-001']
    }
    
    by_test = comparisons.get('by_test', {})
    
    for property_name, test_list in tests_by_property.items():
        tests_in_data = [t for t in test_list if t in by_test]
        
        if not tests_in_data:
            continue
        
        f.write(f"\n{property_name.upper()}:\n")
        
        for test_name in tests_in_data:
            comp = by_test[test_name]
            f.write(f"\n  {test_name}:\n")
            f.write(f"    Meilleur : {comp['best_conservation']}\n")
            f.write(f"    Pire     : {comp['worst_conservation']}\n")
            f.write(f"    Classement : {', '.join(comp['ranking'][:5])}...\n")
    
    f.write("\n")


# =============================================================================
# FOOTER CONSULTATION
# =============================================================================

def write_consultation_footer(
    f,
    width: int = 80,
    char: str = '='
) -> None:
    """
    Écrit footer avec fichiers consultation.
    
    Args:
        f: File handle
        width: Largeur section
        char: Caractère bordure
    """
    f.write(char * width + "\n")
    f.write("CONSULTATION DÉTAILLÉE\n")
    f.write(char * width + "\n")
    f.write("gamma_profiles.json       : Profils complets tous gammas × tests\n")
    f.write("gamma_profiles.csv        : Vue tabulaire pour analyse\n")
    f.write("comparisons.json          : Classements inter-gammas\n")
    f.write("structural_patterns.json  : Analyses globales (variance, interactions)\n")
    f.write("diagnostics.json          : Diagnostics numériques détaillés\n")
    f.write("marginal_variance_*.csv   : Données brutes analyses (3 strates)\n")
    f.write(char * width + "\n")
    
# tests/utilities/statistical_utils.py
"""
Statistical Utilities - Outils statistiques réutilisables.

RESPONSABILITÉS :
- Calculs variance (η², SSB/SSW)
- Filtrage artefacts numériques (inf/nan)
- Diagnostics dégénérescences projections
- Diagnostics ruptures échelle relatives
- Tests statistiques standards (Kruskal-Wallis)

UTILISATEURS :
- verdict_engine.py (analyses globales)
- gamma_profiling.py (agrégations)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# CALCULS VARIANCE (η²)
# =============================================================================

def compute_eta_squared(groups: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Calcule eta-squared (η²) : proportion variance expliquée par groupes.
    
    FORMULE :
    η² = SSB / (SSB + SSW)
    - SSB (Sum of Squares Between) : variance entre groupes
    - SSW (Sum of Squares Within) : variance intra-groupes
    - SST (Sum of Squares Total) : SSB + SSW
    
    Args:
        groups: Liste tableaux numpy (un par groupe)
    
    Returns:
        (eta2, ssb, ssw)
        - eta2 : proportion variance expliquée [0, 1]
        - ssb : somme carrés entre groupes
        - ssw : somme carrés intra-groupes
    
    Examples:
        >>> g1 = np.array([1, 2, 3])
        >>> g2 = np.array([4, 5, 6])
        >>> eta2, ssb, ssw = compute_eta_squared([g1, g2])
        >>> eta2  # Proche de 1.0 (groupes bien séparés)
        0.95
    
    Notes:
        - Retourne (0.0, 0.0, 0.0) si données insuffisantes
        - Protection division par zéro (sst < 1e-10)
        - Filtre groupes vides automatiquement
    """
    # Filtrer groupes vides
    groups_valid = [g for g in groups if len(g) > 0]
    
    if len(groups_valid) < 2:
        return 0.0, 0.0, 0.0
    
    # Concaténer toutes valeurs
    all_values = np.concatenate(groups_valid)
    
    if len(all_values) < 2:
        return 0.0, 0.0, 0.0
    
    # Grand mean (moyenne totale)
    grand_mean = np.mean(all_values)
    
    # SSB : variance expliquée par appartenance groupe
    ssb = sum(
        len(g) * (np.mean(g) - grand_mean)**2
        for g in groups_valid
    )
    
    # SSW : variance résiduelle intra-groupes
    ssw = sum(
        np.sum((g - np.mean(g))**2)
        for g in groups_valid
    )
    
    # SST : variance totale
    sst = ssb + ssw
    
    # η² : proportion variance expliquée
    if sst > 1e-10:
        eta2 = ssb / sst
    else:
        eta2 = 0.0
    
    return eta2, ssb, ssw


def kruskal_wallis_test(groups: List[np.ndarray]) -> Tuple[float, float]:
    """
    Test Kruskal-Wallis avec gestion erreurs.
    
    Args:
        groups: Liste tableaux numpy (un par groupe)
    
    Returns:
        (statistic, p_value)
    
    Raises:
        ValueError: Si moins de 2 groupes ou données insuffisantes
    
    Examples:
        >>> g1 = np.array([1, 2, 3])
        >>> g2 = np.array([4, 5, 6])
        >>> stat, pval = kruskal_wallis_test([g1, g2])
        >>> pval < 0.05  # Différence significative
        True
    """
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis nécessite au moins 2 groupes")
    
    # Filtrer groupes vides
    groups_valid = [g for g in groups if len(g) > 0]
    
    if len(groups_valid) < 2:
        raise ValueError("Moins de 2 groupes non vides")
    
    try:
        statistic, p_value = kruskal(*groups_valid)
        return float(statistic), float(p_value)
    except (ValueError, Exception) as e:
        raise ValueError(f"Erreur Kruskal-Wallis: {e}")


# =============================================================================
# FILTRAGE ARTEFACTS NUMÉRIQUES
# =============================================================================

def is_numeric_valid(obs: dict) -> bool:
    """
    Détecte artefacts numériques (inf/nan) dans projections exploitées.
    
    VÉRIFIE :
    - statistics : initial, final, mean, std, min, max
    - evolution : slope, volatility, relative_change
    
    Args:
        obs: Observation dict avec observation_data
    
    Returns:
        True si aucun artefact détecté
    
    Notes:
        - Fonction privée (usage interne filter_numeric_artifacts)
        - Vérifie TOUTES projections utilisées par verdict_engine
    """
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    # Vérifier statistics
    for metric_stats in statistics.values():
        values_to_check = [
            metric_stats.get('initial'),
            metric_stats.get('final'),
            metric_stats.get('mean'),
            metric_stats.get('std'),
            metric_stats.get('min'),
            metric_stats.get('max'),
        ]
        
        for v in values_to_check:
            if v is not None:
                if np.isinf(v) or np.isnan(v):
                    return False
    
    # Vérifier evolution
    for metric_evol in evolution.values():
        values_to_check = [
            metric_evol.get('slope'),
            metric_evol.get('volatility'),
            metric_evol.get('relative_change'),
        ]
        
        for v in values_to_check:
            if v is not None:
                if np.isinf(v) or np.isnan(v):
                    return False
    
    return True


def filter_numeric_artifacts(observations: List[dict]) -> Tuple[List[dict], dict]:
    """
    Filtre observations avec artefacts numériques.
    
    Log rejets pour traçabilité (par test).
    
    Args:
        observations: Liste observations
    
    Returns:
        (valid_obs, rejection_stats)
        
        rejection_stats :
        {
            'total_observations': int,
            'valid_observations': int,
            'rejected_observations': int,
            'rejection_rate': float,
            'rejected_by_test': dict
        }
    
    Examples:
        >>> valid, stats = filter_numeric_artifacts(observations)
        >>> stats['rejection_rate']
        0.05  # 5% observations rejetées
        >>> stats['rejected_by_test']
        {'TOP-001': 12, 'SPE-001': 3}
    """
    valid = []
    rejected_by_test = {}
    
    for obs in observations:
        if is_numeric_valid(obs):
            valid.append(obs)
        else:
            test_name = obs.get('test_name', 'UNKNOWN')
            rejected_by_test[test_name] = rejected_by_test.get(test_name, 0) + 1
    
    total_rejected = len(observations) - len(valid)
    
    stats = {
        'total_observations': len(observations),
        'valid_observations': len(valid),
        'rejected_observations': total_rejected,
        'rejection_rate': total_rejected / len(observations) if observations else 0,
        'rejected_by_test': rejected_by_test,
    }
    
    return valid, stats


# =============================================================================
# DIAGNOSTICS DÉGÉNÉRESCENCES
# =============================================================================

def diagnose_numeric_degeneracy(obs: dict) -> List[str]:
    """
    Détecte dégénérescences numériques sur projections exploitées.
    
    INSPECTE :
    - value_final, value_mean, slope, volatility, relative_change
    
    FLAGS DÉTECTÉS (non exclusifs) :
    - INFINITE_PROJECTION : inf détecté
    - NAN_PROJECTION : nan détecté
    - EXTREME_MAGNITUDE : |valeur| > 1e50
    
    Args:
        obs: Observation dict
    
    Returns:
        Liste flags format "metric:projection:flag_type"
        Exemple : ['asymmetry:value_final:EXTREME_MAGNITUDE']
    
    Notes:
        - Flags par projection (pas global)
        - inf/nan normalement filtrés avant (filter_numeric_artifacts)
        - EXTREME_MAGNITUDE : valeurs très grandes mais finies
    """
    flags = []
    
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    for metric_name in statistics.keys():
        # Récupérer projections exploitées
        stat = statistics.get(metric_name, {})
        evol = evolution.get(metric_name, {})
        
        projections = {
            'value_final': stat.get('final'),
            'value_mean': stat.get('mean'),
            'slope': evol.get('slope'),
            'volatility': evol.get('volatility'),
            'relative_change': evol.get('relative_change'),
        }
        
        # Inspecter chaque projection
        for proj_name, value in projections.items():
            if value is None:
                continue
            
            # Flags artefacts
            if np.isinf(value):
                flags.append(f"{metric_name}:{proj_name}:INFINITE_PROJECTION")
                continue
            
            if np.isnan(value):
                flags.append(f"{metric_name}:{proj_name}:NAN_PROJECTION")
                continue
            
            # Flag magnitude extrême (> 1e50 mais < inf)
            if abs(value) > 1e50:
                flags.append(f"{metric_name}:{proj_name}:EXTREME_MAGNITUDE")
    
    return flags


def generate_degeneracy_report(observations: List[dict]) -> dict:
    """
    Génère rapport diagnostique dégénérescences.
    
    AGRÉGATIONS :
    - Comptage flags par type
    - Comptage observations flaggées
    - Répartition par test
    - Répartition par projection
    
    Args:
        observations: Liste observations (non filtrées)
    
    Returns:
        {
            'total_observations': int,
            'observations_with_flags': int,
            'flag_rate': float,
            'flag_counts': dict,
            'flags_by_test': dict,
            'flags_by_projection': dict
        }
    
    Examples:
        >>> report = generate_degeneracy_report(observations)
        >>> report['flag_rate']
        0.12  # 12% observations avec flags
        >>> report['flags_by_projection']['value_final']
        45  # 45 occurrences flags sur value_final
    """
    flag_counts = defaultdict(int)
    obs_with_flags = 0
    flags_by_test = defaultdict(lambda: defaultdict(int))
    flags_by_projection = defaultdict(int)
    
    for obs in observations:
        flags = diagnose_numeric_degeneracy(obs)
        
        if flags:
            obs_with_flags += 1
            test_name = obs.get('test_name', 'UNKNOWN')
            
            for flag in flags:
                flag_counts[flag] += 1
                
                # Parser flag: metric:projection:flag_type
                parts = flag.split(':')
                if len(parts) >= 3:
                    projection = parts[1]
                    flag_type = parts[2]
                    flags_by_projection[projection] += 1
                    flags_by_test[test_name][flag_type] += 1
    
    report = {
        'total_observations': len(observations),
        'observations_with_flags': obs_with_flags,
        'flag_rate': obs_with_flags / len(observations) if observations else 0,
        'flag_counts': dict(flag_counts),
        'flags_by_test': {k: dict(v) for k, v in flags_by_test.items()},
        'flags_by_projection': dict(flags_by_projection),
    }
    
    return report


def print_degeneracy_report(report: dict) -> None:
    """
    Affiche rapport diagnostique dégénérescences (stdout).
    
    FORMAT :
    - Header avec totaux
    - Dégénérescences par projection (5 projections analysées)
    - Top 5 flags les plus fréquents
    - Détail par test
    
    Args:
        report: Retour generate_degeneracy_report()
    """
    total = report['total_observations']
    flagged = report['observations_with_flags']
    rate = report['flag_rate']
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC DÉGÉNÉRESCENCES NUMÉRIQUES (projections exploitées)")
    print("=" * 80)
    print(f"Total observations:        {total}")
    print(f"Observations flaggées:     {flagged} ({rate*100:.1f}%)")
    print()
    
    if flagged == 0:
        print("✓ Aucune dégénérescence détectée\n")
        return
    
    # Flags par projection
    print("Dégénérescences par projection (variables analysées):")
    print("-" * 80)
    
    projections = report['flags_by_projection']
    if projections:
        for proj_name in ['value_final', 'value_mean', 'slope', 'volatility', 'relative_change']:
            count = projections.get(proj_name, 0)
            if count > 0:
                percentage = (count / total) * 100
                print(f"  {proj_name:20s} : {count:5d} occurrences ({percentage:5.1f}%)")
    
    print()
    
    # Top flags globaux
    print("Flags les plus fréquents:")
    print("-" * 80)
    
    # Agréger par type de flag
    flag_type_counts = defaultdict(int)
    for flag, count in report['flag_counts'].items():
        parts = flag.split(':')
        flag_type = parts[-1] if parts else flag
        flag_type_counts[flag_type] += count
    
    for flag_type, count in sorted(flag_type_counts.items(), key=lambda x: -x[1])[:5]:
        percentage = (count / total) * 100
        print(f"  {flag_type:25s} : {count:5d} occurrences ({percentage:5.1f}%)")
    
    print()
    
    # Détail par test
    print("Dégénérescences par test:")
    print("-" * 80)
    
    for test_name in sorted(report['flags_by_test'].keys()):
        flags = report['flags_by_test'][test_name]
        total_flags = sum(flags.values())
        print(f"\n{test_name}: {total_flags} flags")
        for flag_type, count in sorted(flags.items(), key=lambda x: -x[1]):
            print(f"  {flag_type:25s} : {count:5d}")
    
    print("\n" + "=" * 80 + "\n")


# =============================================================================
# DIAGNOSTICS RUPTURES ÉCHELLE
# =============================================================================

def diagnose_scale_outliers(observations: List[dict]) -> dict:
    """
    Détecte ruptures d'échelle relatives par contexte (test×metric×projection).
    
    CRITÈRE :
    - Valeur > P90 + 5 décades (facteur 1e5)
    - Raisonnement relatif (pas seuil absolu)
    - Contextuel (P90 calculé par test/métrique/projection)
    
    PROJECTIONS ANALYSÉES :
    - value_final, value_mean, slope, relative_change
    
    Args:
        observations: Liste observations
    
    Returns:
        {
            'total_observations': int,
            'observations_with_outliers': int,
            'outlier_rate': float,
            'contexts_analyzed': int,
            'contexts_with_outliers': int,
            'outliers_by_context': dict,
            'thresholds': dict
        }
    
    Notes:
        - Min 10 observations par contexte pour calculer P90
        - Outliers stockent gap en décades (log10)
        - Clés contexte : (test_name, metric_name, proj_name)
    """
    projections = ['value_final', 'value_mean', 'slope', 'relative_change']
    
    # Collecter valeurs par contexte
    by_context = defaultdict(list)
    obs_indices = defaultdict(list)
    
    for i, obs in enumerate(observations):
        test_name = obs.get('test_name')
        obs_data = obs.get('observation_data', {})
        statistics = obs_data.get('statistics', {})
        evolution = obs_data.get('evolution', {})
        
        for metric_name in statistics.keys():
            stat = statistics.get(metric_name, {})
            evol = evolution.get(metric_name, {})
            
            values = {
                'value_final': stat.get('final'),
                'value_mean': stat.get('mean'),
                'slope': evol.get('slope'),
                'relative_change': evol.get('relative_change'),
            }
            
            for proj_name, value in values.items():
                if value and np.isfinite(value) and abs(value) > 1e-10:
                    key = (test_name, metric_name, proj_name)
                    log_val = np.log10(abs(value))
                    by_context[key].append(log_val)
                    obs_indices[key].append((i, value))
    
    # Calculer P90 par contexte (min 10 observations)
    thresholds = {}
    for key, log_values in by_context.items():
        if len(log_values) >= 10:
            p90 = np.percentile(log_values, 90)
            thresholds[key] = p90
    
    # Détecter outliers (+5 décades au-dessus P90)
    outliers = defaultdict(list)
    outlier_obs_ids = set()
    
    for key, threshold in thresholds.items():
        for obs_idx, value in obs_indices[key]:
            log_val = np.log10(abs(value))
            gap = log_val - threshold
            
            if gap > 5.0:
                outliers[key].append({
                    'obs_idx': obs_idx,
                    'value': value,
                    'log_value': log_val,
                    'gap_decades': gap,
                })
                outlier_obs_ids.add(obs_idx)
    
    report = {
        'total_observations': len(observations),
        'observations_with_outliers': len(outlier_obs_ids),
        'outlier_rate': len(outlier_obs_ids) / len(observations) if observations else 0,
        'contexts_analyzed': len(thresholds),
        'contexts_with_outliers': len(outliers),
        'outliers_by_context': dict(outliers),
        'thresholds': thresholds,
    }
    
    return report


def print_scale_outliers_report(report: dict) -> None:
    """
    Affiche rapport ruptures d'échelle (stdout).
    
    FORMAT :
    - Header avec totaux
    - Top 10 contextes avec le plus d'outliers
    - Pour chaque contexte : stats gaps + pires cas
    
    Args:
        report: Retour diagnose_scale_outliers()
    """
    total = report['total_observations']
    flagged = report['observations_with_outliers']
    rate = report['outlier_rate']
    contexts_analyzed = report['contexts_analyzed']
    contexts_with_outliers = report['contexts_with_outliers']
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC RUPTURES D'ÉCHELLE RELATIVES")
    print("=" * 80)
    print(f"Total observations:              {total}")
    print(f"Contextes analysés (test×métrique×proj): {contexts_analyzed}")
    print(f"Observations avec outliers:      {flagged} ({rate*100:.1f}%)")
    print(f"Contextes ayant outliers:        {contexts_with_outliers}")
    print()
    
    if flagged == 0:
        print("✓ Aucune rupture d'échelle détectée\n")
        return
    
    # Top contextes
    print("Contextes avec ruptures d'échelle (>P90 + 5 décades):")
    print("-" * 80)
    
    outliers_by_context = report['outliers_by_context']
    
    # Trier par nombre d'outliers
    sorted_contexts = sorted(
        outliers_by_context.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for key, outlier_list in sorted_contexts[:10]:
        test_name, metric_name, proj_name = key
        count = len(outlier_list)
        percentage = (count / total) * 100
        
        # Stats gaps
        gaps = [o['gap_decades'] for o in outlier_list]
        max_gap = max(gaps)
        mean_gap = np.mean(gaps)
        
        print(f"\n{test_name}/{metric_name} [{proj_name}]:")
        print(f"  {count:4d} outliers ({percentage:5.2f}%)")
        print(f"  Gap max: +{max_gap:.1f} décades, moyen: +{mean_gap:.1f} décades")
        
        # Pires cas (3 exemples)
        worst_cases = sorted(outlier_list, key=lambda x: x['gap_decades'], reverse=True)[:3]
        if worst_cases:
            print(f"  Pires cas:")
            for case in worst_cases:
                print(f"    obs#{case['obs_idx']:5d}: {case['value']:.2e} "
                      f"(+{case['gap_decades']:.1f} décades)")
    
    print("\n" + "=" * 80 + "\n")
    
# tests/utilities/timeline_utils.py
"""
Timeline Utilities - Construction timelines dynamiques compositionnels.

RESPONSABILITÉS :
- Classification timing (early/mid/late)
- Composition descripteurs {timing}_{event}_then_{event}
- Extraction séquences depuis dynamic_events
- Seuils globaux relatifs (pas absolus)

ARCHITECTURE :
- classify_timing() : early/mid/late selon seuils globaux
- compute_timeline_descriptor() : Composition automatique phases
- extract_dynamic_events() : Parsing observation_data

PRINCIPE R0 :
- Toute notion temporelle est RELATIVE (jamais absolue)
- Seuils globaux uniques (pas de variation par test)
- Composition automatique : {timing}_{event}_then_{event}
- Descriptif pas causal ("then" pas "causes")

UTILISATEURS :
- gamma_profiling.py (timelines gamma)
- Futurs : analyses temporelles avancées
"""

from typing import Dict, List, Tuple


# =============================================================================
# CONFIGURATION TIMELINES (globale, unique, documentée)
# =============================================================================

TIMELINE_THRESHOLDS = {
    'early': 0.20,  # onset < 20% durée
    'mid':   0.60,  # 20% ≤ onset ≤ 60%
    'late':  0.60   # onset > 60%
}

"""
PRINCIPE TIMELINES R0 :
- Toute notion temporelle est RELATIVE (jamais absolue)
- Seuils globaux uniques (pas de variation par test)
- Composition automatique : {timing}_{event}_then_{event}
- Descriptif pas causal ("then" pas "causes")

Exemples :
- early_instability_then_collapse
- mid_deviation_then_saturation
- late_instability_then_plateau

Structure intermédiaire disponible pour exploitation :
{
    'phases': [
        {'event': 'instability', 'timing': 'early', 'onset_relative': 0.05},
        {'event': 'collapse', 'timing': 'late', 'onset_relative': 0.85}
    ],
    'timeline_compact': 'early_instability_then_collapse',
    'n_phases': 2
}
"""


# =============================================================================
# CLASSIFICATION TIMING
# =============================================================================

def classify_timing(onset_relative: float) -> str:
    """
    Classifie timing selon seuils globaux.
    
    Args:
        onset_relative: Onset normalisé [0, 1]
    
    Returns:
        'early' | 'mid' | 'late'
    
    Examples:
        >>> classify_timing(0.05)
        'early'
        >>> classify_timing(0.45)
        'mid'
        >>> classify_timing(0.85)
        'late'
    """
    if onset_relative < TIMELINE_THRESHOLDS['early']:
        return 'early'
    elif onset_relative <= TIMELINE_THRESHOLDS['mid']:
        return 'mid'
    else:
        return 'late'


# =============================================================================
# COMPOSITION TIMELINES
# =============================================================================

def compute_timeline_descriptor(
    sequence: List[str],
    sequence_timing_relative: List[float],
    oscillatory_global: bool = False
) -> Dict:
    """
    Génère descriptor timeline compositionnel.
    
    PRINCIPE :
    - Composition automatique : {timing}_{event}_then_{event}
    - Pas de patterns hardcodés
    - Structure intermédiaire pour exploitation
    
    Args:
        sequence: ['deviation', 'saturation']
        sequence_timing_relative: [0.05, 0.80]
        oscillatory_global: Comportement oscillatoire global
    
    Returns:
        {
            'phases': [
                {'event': 'deviation', 'timing': 'early', 'onset_relative': 0.05},
                {'event': 'saturation', 'timing': 'late', 'onset_relative': 0.80}
            ],
            'timeline_compact': 'early_deviation_then_saturation',
            'n_phases': 2,
            'oscillatory_global': False
        }
    
    Cas spéciaux :
    - Aucun événement → 'no_significant_dynamics'
    - 1 événement → '{timing}_{event}_only'
    - 2+ événements → '{timing1}_{event1}_then_{event2}'
    - Oscillatoire global → préfixe 'oscillatory_'
    
    Examples:
        >>> desc = compute_timeline_descriptor(
        ...     ['deviation', 'saturation'],
        ...     [0.05, 0.80]
        ... )
        >>> desc['timeline_compact']
        'early_deviation_then_saturation'
        
        >>> desc = compute_timeline_descriptor([], [], True)
        >>> desc['timeline_compact']
        'no_significant_dynamics'
    """
    # Vérifier cohérence listes
    if not sequence or not sequence_timing_relative:
        return {
            'phases': [],
            'timeline_compact': 'no_significant_dynamics',
            'n_phases': 0,
            'oscillatory_global': oscillatory_global
        }
    
    # Vérifier longueurs égales
    if len(sequence) != len(sequence_timing_relative):
        return {
            'phases': [],
            'timeline_compact': 'no_significant_dynamics',
            'n_phases': 0,
            'oscillatory_global': oscillatory_global
        }
    
    # Construire phases structurées
    phases = []
    for event, onset_rel in zip(sequence, sequence_timing_relative):
        timing = classify_timing(onset_rel)
        phases.append({
            'event': event,
            'timing': timing,
            'onset_relative': float(onset_rel)
        })
    
    # Composition timeline_compact
    if len(phases) == 1:
        # Format : {timing}_{event}_only
        p = phases[0]
        compact = f"{p['timing']}_{p['event']}_only"
    
    elif len(phases) == 2:
        # Format : {timing1}_{event1}_then_{event2}
        p1, p2 = phases[0], phases[1]
        compact = f"{p1['timing']}_{p1['event']}_then_{p2['event']}"
    
    else:
        # 3+ phases : simplifier
        p1 = phases[0]
        pN = phases[-1]
        compact = f"{p1['timing']}_{p1['event']}_to_{pN['event']}_complex"
    
    # Préfixe oscillatoire si global
    if oscillatory_global:
        compact = f"oscillatory_{compact}"
    
    return {
        'phases': phases,
        'timeline_compact': compact,
        'n_phases': len(phases),
        'oscillatory_global': oscillatory_global
    }


# =============================================================================
# EXTRACTION ÉVÉNEMENTS
# =============================================================================

def extract_dynamic_events(observation: Dict, metric_name: str) -> Dict:
    """
    Extrait événements dynamiques depuis observation.
    
    Lit depuis observation_data['dynamic_events'][metric_name].
    
    Returns:
        {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool,
            'sequence': [...],
            'sequence_timing': [...],
            'sequence_timing_relative': [...],
            'saturation_onset_estimated': bool,
            'oscillatory_global': bool
        }
    
    Notes:
        - Retourne valeurs par défaut si dynamic_events absent
        - Compatible fallback (test_engine avant enrichissement)
    
    Examples:
        >>> events = extract_dynamic_events(obs, 'asymmetry_norm')
        >>> events['sequence']
        ['deviation', 'instability', 'saturation']
        >>> events['sequence_timing_relative']
        [0.05, 0.15, 0.80]
    """
    obs_data = observation.get('observation_data', {})
    dynamic_events = obs_data.get('dynamic_events', {})
    
    events = dynamic_events.get(metric_name, {})
    
    # Valeurs par défaut si absentes
    return {
        'deviation_onset': events.get('deviation_onset'),
        'instability_onset': events.get('instability_onset'),
        'oscillatory': events.get('oscillatory', False),
        'saturation': events.get('saturation', False),
        'collapse': events.get('collapse', False),
        'sequence': events.get('sequence', []),
        'sequence_timing': events.get('sequence_timing', []),
        'sequence_timing_relative': events.get('sequence_timing_relative', []),
        'saturation_onset_estimated': events.get('saturation_onset_estimated', False),
        'oscillatory_global': events.get('oscillatory_global', False)
    }


def extract_metric_timeseries(
    observation: Dict,
    metric_name: str
) -> Tuple[List[float], bool]:
    """
    Extrait série temporelle avec marqueur fallback.
    
    Returns:
        (values, is_fallback)
        - values: liste valeurs ou None
        - is_fallback: True si proxy linéaire utilisé
    
    Notes:
        - Fallback : linspace(initial, final) si timeseries absent
        - Utilisé pour visualisations/analyses temporelles avancées
    
    Examples:
        >>> values, is_fallback = extract_metric_timeseries(obs, 'asymmetry_norm')
        >>> is_fallback
        False  # Timeseries disponible
        >>> len(values)
        200  # n_iterations
    """
    obs_data = observation.get('observation_data', {})
    timeseries = obs_data.get('timeseries', {})
    
    values = timeseries.get(metric_name)
    
    if values is not None:
        return list(values), False
    
    # Fallback : proxy linéaire depuis statistics
    stats = obs_data.get('statistics', {}).get(metric_name, {})
    initial = stats.get('initial')
    final = stats.get('final')
    
    if initial is not None and final is not None:
        import numpy as np
        n_iterations = obs_data.get('metadata', {}).get('n_iterations', 200)
        return list(np.linspace(initial, final, n_iterations)), True
    
    return None, True
    
