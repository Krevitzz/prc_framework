"""
prc.featuring.registries.universal_lite

Responsabilité : Features universelles — tout tenseur, tout rank

Layer : universal (applicability: [])

Architecture :
    _to_matrix   : reshape tenseur → matrice 2D pour SVD (unfolding mode-0)
    _svd_values  : calcule valeurs singulières (configurable n_components)
    Features     : euclidean_norm, entropy (fix collapse), mean_value, std_value,
                   range_value, cv, singular_value_max, singular_value_min,
                   singular_value_spread, condition_number_svd, nuclear_norm,
                   effective_rank

Notes :
    - Toutes fonctions : state np.ndarray (*dims) → float
    - SVD via _to_matrix → np.linalg.svd(compute_uv=False)
    - NaN/Inf retournés comme signal physique (P2 charter), jamais silencieux
    - n_components=None → tous les σ, n_components=k → top-k
"""

import numpy as np


# =============================================================================
# HELPERS INTERNES
# =============================================================================

def _to_matrix(state: np.ndarray) -> np.ndarray:
    """
    Reshape tenseur quelconque → matrice 2D pour SVD.

    Args:
        state : np.ndarray (*dims)

    Returns:
        np.ndarray (dims[0], prod(dims[1:]))
        Rank 2 → identique (view).
        Rank 1 → (1, n).
        Rank 3+ → unfolding mode-0.

    Notes:
        - Unfolding mode-0 : chaque "slice" devient une ligne
        - Pas de copie inutile si déjà 2D
    """
    if state.ndim == 1:
        return state.reshape(1, -1)
    if state.ndim == 2:
        return state
    # Rank 3+ : (dims[0], prod(dims[1:]))
    return state.reshape(state.shape[0], -1)


def _svd_values(state: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Calcule valeurs singulières σ via SVD.

    Args:
        state        : np.ndarray (*dims)
        n_components : None → tous les σ | int k → top-k σ

    Returns:
        np.ndarray σ trié décroissant (longueur min(m,n) ou k)
        Array vide si échec numérique.

    Notes:
        - compute_uv=False : ne calcule pas U et V (plus rapide)
        - LinAlgError → array vide (signal en amont via nan dans features)
    """
    matrix = _to_matrix(state)

    try:
        sigmas = np.linalg.svd(matrix, compute_uv=False)  # Trié décroissant
    except np.linalg.LinAlgError:
        return np.array([])

    if n_components is not None and n_components > 0:
        sigmas = sigmas[:n_components]

    return sigmas


# =============================================================================
# FEATURES EXISTANTES
# =============================================================================

def euclidean_norm(state: np.ndarray) -> float:
    """
    Norme euclidienne L2 — flatten automatique.

    Returns:
        float — ||state||₂
    """
    return float(np.linalg.norm(state))


def entropy(state: np.ndarray, bins: int = 50) -> float:
    """
    Entropie Shannon sur distribution des valeurs.

    FIX collapse : n_bins adaptatif = min(bins, max(2, n_valeurs_uniques)).
    Évite "Too many bins for data range" sur états quasi-constants.
    Entropie ≈ 0 sur état collapsed est une information physique, pas une erreur.

    Args:
        state : np.ndarray (*dims)
        bins  : nombre bins cible (défaut 50)

    Returns:
        float — entropie Shannon ≥ 0
        0.0 si état constant (collapsed) ou vide

    Notes:
        - Flatten automatique
        - Protection log(0) via filtrage hist > 0
        - n_bins = min(bins, max(2, n_unique)) — adaptatif
    """
    flat = state.flatten()

    if len(flat) == 0:
        return 0.0

    # Court-circuit : état constant → entropie = 0 (collapsed)
    # Évite "Too many bins for data range" même avec bins=2
    if np.max(flat) == np.min(flat):
        return 0.0

    # Nombre de valeurs uniques (approximé par arrondi à la précision float)
    n_unique = len(np.unique(np.round(flat, decimals=10)))

    # Bins adaptatif — évite erreur si n_unique < bins
    n_bins = max(2, min(bins, n_unique))

    hist, _ = np.histogram(flat, bins=n_bins)

    hist = hist.astype(float)
    hist_sum = hist.sum()

    if hist_sum == 0:
        return 0.0

    hist = hist / hist_sum
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    return float(-np.sum(hist * np.log(hist)))


def mean_value(state: np.ndarray) -> float:
    """
    Moyenne des valeurs — flatten automatique.

    Returns:
        float — mean(state)
    """
    return float(np.mean(state))


def std_value(state: np.ndarray) -> float:
    """
    Écart-type des valeurs — flatten automatique.

    Returns:
        float — std(state)
    """
    return float(np.std(state))


# =============================================================================
# FEATURES NOUVELLES GÉNÉRIQUES
# =============================================================================

def range_value(state: np.ndarray) -> float:
    """
    Étendue brute : max - min — flatten automatique.

    Returns:
        float — max(state) - min(state)
        0.0 si état constant
    """
    flat = state.flatten()
    return float(np.max(flat) - np.min(flat))


def cv(state: np.ndarray) -> float:
    """
    Coefficient de variation : std / (|mean| + ε).

    Mesure de variabilité relative, scale-invariant.

    Returns:
        float — std / (|mean| + 1e-10)
        Grand si mean ≈ 0 avec variance non nulle → normal, signal physique
    """
    flat = state.flatten()
    return float(np.std(flat) / (np.abs(np.mean(flat)) + 1e-10))


# =============================================================================
# FEATURES SVD-BASED
# =============================================================================

def singular_value_max(state: np.ndarray, n_components: int = None) -> float:
    """
    Valeur singulière maximale σ₁.

    Généralise eigenvalue_max à tout tenseur.
    Mesure l'énergie dominante de la structure linéaire.

    Args:
        n_components : None → tous les σ | k → top-k avant max

    Returns:
        float — σ₁
        np.nan si SVD échoue (signal physique)
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) == 0:
        return np.nan
    return float(sigmas[0])


def singular_value_min(state: np.ndarray, n_components: int = None) -> float:
    """
    Valeur singulière minimale σₙ.

    Mesure la sensibilité : proche de 0 si quasi-singulier.

    Args:
        n_components : None → tous les σ | k → top-k avant min

    Returns:
        float — σₙ
        np.nan si SVD échoue
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) == 0:
        return np.nan
    return float(sigmas[-1])


def singular_value_spread(state: np.ndarray, n_components: int = None) -> float:
    """
    Étendue spectrale : σ₁ - σₙ.

    Généralise eigenvalue_spread à tout tenseur.
    Grande valeur → structure hiérarchisée. Petite → isotrope.

    Args:
        n_components : None → tous les σ

    Returns:
        float — σ₁ - σₙ
        np.nan si SVD échoue ou < 2 σ
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) < 2:
        return np.nan
    return float(sigmas[0] - sigmas[-1])


def condition_number_svd(state: np.ndarray, n_components: int = None) -> float:
    """
    Conditionnement via SVD : σ₁ / (σₙ + ε).

    Généralise condition_number à tout tenseur.
    np.inf si σₙ = 0 → signal physique (matrice singulière).

    Args:
        n_components : None → tous les σ

    Returns:
        float — σ₁ / (σₙ + ε)
        np.nan si SVD échoue
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) == 0:
        return np.nan
    if len(sigmas) == 1:
        return float(sigmas[0] / (sigmas[0] + 1e-10))

    sigma_min = sigmas[-1]
    if sigma_min < 1e-15:
        return np.inf

    return float(sigmas[0] / sigma_min)


def nuclear_norm(state: np.ndarray, n_components: int = None) -> float:
    """
    Norme nucléaire : Σσᵢ.

    Somme des valeurs singulières — généralise trace(√(AᵀA)).
    Mesure l'énergie totale distribuée dans la structure.

    Args:
        n_components : None → tous les σ | k → top-k seulement

    Returns:
        float — Σσᵢ
        np.nan si SVD échoue
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) == 0:
        return np.nan
    return float(np.sum(sigmas))


def effective_rank(state: np.ndarray, n_components: int = None) -> float:
    """
    Rang effectif : exp(H(p)) où p = σᵢ / Σσᵢ.

    Mesure de complexité structurelle via entropie de la distribution des σ.
    - Toutes σ égales → entropie max → rang effectif = n (structure riche)
    - Une σ domine → entropie ≈ 0 → rang effectif ≈ 1 (structure appauvrie)

    Args:
        n_components : None → tous les σ

    Returns:
        float — exp(H(p)) ∈ [1, n]
        1.0 si un seul σ non nul
        np.nan si SVD échoue ou Σσ = 0

    Notes:
        - Référence : Roy & Vetterli (2007)
        - Normalisé : rang effectif ∈ [1, rank(matrix)]
    """
    sigmas = _svd_values(state, n_components)
    if len(sigmas) == 0:
        return np.nan

    sigma_sum = np.sum(sigmas)
    if sigma_sum < 1e-15:
        return np.nan

    # Distribution normalisée
    p = sigmas / sigma_sum

    # Filtrer zéros (éviter log(0))
    p = p[p > 1e-15]

    if len(p) == 0:
        return 1.0

    # Entropie Shannon sur σ normalisés
    h = -np.sum(p * np.log(p))

    return float(np.exp(h))


# =============================================================================
# FEATURES DELTA (comparaison initial → final)
# =============================================================================

def compute_delta_features(features: dict) -> dict:
    """
    Calcule features de variation initial → final depuis dict features en RAM.

    Contrairement aux autres fonctions (state → float), celle-ci opère sur
    les scalaires déjà calculés — elle est appelée par hub_featuring après
    la boucle d'extraction.

    Features produites :
        norm_ratio           : euclidean_norm_final / (norm_initial + ε)
                               → régime de transformation (< 1 = contraction)
        entropy_delta        : entropy_final - entropy_initial
                               → gamma concentre (< 0) ou diffuse (> 0)
        effective_rank_delta : effective_rank_final - effective_rank_initial
                               → gamma enrichit (> 0) ou appauvrit (< 0) la structure
        log_condition_delta  : log(cond_final + ε) - log(cond_initial + ε)
                               → gamma déstabilise (> 0) ou stabilise (< 0)
        spread_ratio         : singular_value_spread_final / (spread_initial + ε)
                               → gamma hiérarchise (> 1) ou aplatit (< 1) le spectre

    Args:
        features : Dict scalaires déjà calculés (output extract_features)

    Returns:
        Dict features delta — clés préfixées par le nom de la feature
        np.nan si feature source absente, NaN ou Inf

    Notes:
        - Guard systématique : source NaN/Inf → delta NaN (jamais silencieux)
        - ε = 1e-10 pour éviter division par zéro
        - log(x + ε) pour condition_number qui peut valoir 0
    """
    EPS = 1e-10
    deltas = {}

    def _safe_get(key: str):
        """Retourne valeur ou np.nan si absente/non finie."""
        v = features.get(key)
        if v is None:
            return np.nan
        v = float(v)
        if not np.isfinite(v):
            return np.nan
        return v

    # norm_ratio : euclidean_norm_final / norm_initial
    n_init = _safe_get('euclidean_norm_initial')
    n_fin  = _safe_get('euclidean_norm_final')
    if np.isnan(n_init) or np.isnan(n_fin):
        deltas['norm_ratio'] = np.nan
    else:
        deltas['norm_ratio'] = float(n_fin / (n_init + EPS))

    # entropy_delta : entropy_final - entropy_initial
    e_init = _safe_get('entropy_initial')
    e_fin  = _safe_get('entropy_final')
    deltas['entropy_delta'] = float(e_fin - e_init) if not (np.isnan(e_init) or np.isnan(e_fin)) else np.nan

    # effective_rank_delta : effective_rank_final - effective_rank_initial
    r_init = _safe_get('effective_rank_initial')
    r_fin  = _safe_get('effective_rank_final')
    deltas['effective_rank_delta'] = float(r_fin - r_init) if not (np.isnan(r_init) or np.isnan(r_fin)) else np.nan

    # log_condition_delta : log(cond_final + ε) - log(cond_initial + ε)
    c_init = _safe_get('condition_number_svd_initial')
    c_fin  = _safe_get('condition_number_svd_final')
    if np.isnan(c_init) or np.isnan(c_fin):
        deltas['log_condition_delta'] = np.nan
    else:
        deltas['log_condition_delta'] = float(np.log(c_fin + EPS) - np.log(c_init + EPS))

    # spread_ratio : singular_value_spread_final / (spread_initial + ε)
    s_init = _safe_get('singular_value_spread_initial')
    s_fin  = _safe_get('singular_value_spread_final')
    if np.isnan(s_init) or np.isnan(s_fin):
        deltas['spread_ratio'] = np.nan
    else:
        deltas['spread_ratio'] = float(s_fin / (s_init + EPS))

    return deltas