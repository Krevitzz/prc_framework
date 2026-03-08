"""
prc.featuring.registries.timeline_lite

Responsabilité : Layer timeline — extraction features temporelles sur history complet

Remplace : universal_lite.py (projections 3 points → signal T points)

Architecture :
    Fonctions de mesure  : state → float (building blocks, pas des features)
    _MEASURE_FNS         : mapping nom → fonction, dispatch dans extract()
    extract()            : history → dict features
                           1. Calcul signaux [fn(history[t]) for t in T]
                           2. catch22+24 sur chaque signal
                           3. Dérivées simples (alimentent regimes_lite)
                           4. Métadonnées santé (has_nan_inf, is_collapsed)

Nommage features :
    catch22  : {fn_name}__{catch22_feature_name}
               ex: euclidean_norm__CO_f1ecac
    dérivées : norm_ratio, entropy_delta, effective_rank_delta,
               log_condition_delta, spread_ratio
    absolues : norm_final, condition_number_svd_final
               (valeurs finales absolues pour regimes_lite)
    santé    : has_nan_inf, is_collapsed (filtrés par préfixe has_/is_)

Notes :
    - NaN si fonction de mesure échoue sur history[t] (signal physique préservé)
    - catch22 reçoit le signal tel quel — NaN gérés feature par feature
    - Pas d'imputation, pas de seuil global sur longueur signal
    - Nouveau signal : ajouter une fonction de mesure + l'enregistrer dans _MEASURE_FNS
"""

import numpy as np
import pycatch22
from collections import defaultdict
from typing import Dict, Callable, List

_EPSILON = 1e-10


# =============================================================================
# FONCTIONS DE MESURE (state → float)
# Identiques à universal_lite — rôle : générer les signaux, pas des features
# =============================================================================

def euclidean_norm(state: np.ndarray) -> float:
    """Norme euclidienne L2 — flatten automatique."""
    return float(np.linalg.norm(state))


def entropy(state: np.ndarray, bins: int = 50) -> float:
    """
    Entropie Shannon sur distribution des valeurs.

    Adaptatif : n_bins = min(bins, max(2, n_unique)).
    Évite crash sur états quasi-constants.
    """
    flat = state.flatten()
    if len(flat) == 0:
        return 0.0
    if np.max(flat) == np.min(flat):
        return 0.0
    n_unique = len(np.unique(np.round(flat, decimals=10)))
    n_bins = max(2, min(bins, n_unique))
    hist, _ = np.histogram(flat, bins=n_bins)
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return 0.0
    hist = hist / hist_sum
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))


def _to_matrix(state: np.ndarray) -> np.ndarray:
    """Reshape tenseur → matrice 2D pour SVD (unfolding mode-0)."""
    if state.ndim == 1:
        return state.reshape(1, -1)
    if state.ndim == 2:
        return state
    return state.reshape(state.shape[0], -1)


def _svd_values(state: np.ndarray) -> np.ndarray:
    """Valeurs singulières σ, triées décroissant. Array vide si échec."""
    matrix = _to_matrix(state)
    try:
        return np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.array([])


def effective_rank(state: np.ndarray) -> float:
    """
    Rang effectif : exp(H(p)) où p = σᵢ / Σσᵢ.

    Référence : Roy & Vetterli (2007).
    """
    sigmas = _svd_values(state)
    if len(sigmas) == 0:
        return np.nan
    sigma_sum = np.sum(sigmas)
    if sigma_sum < 1e-15:
        return np.nan
    p = sigmas / sigma_sum
    p = p[p > 1e-15]
    if len(p) == 0:
        return 1.0
    return float(np.exp(-np.sum(p * np.log(p))))


def condition_number_svd(state: np.ndarray) -> float:
    """Conditionnement : σ₁ / σₙ. np.inf si σₙ ≈ 0."""
    sigmas = _svd_values(state)
    if len(sigmas) == 0:
        return np.nan
    if len(sigmas) == 1:
        return float(sigmas[0] / (sigmas[0] + _EPSILON))
    if sigmas[-1] < 1e-15:
        return np.inf
    return float(sigmas[0] / sigmas[-1])


def singular_value_spread(state: np.ndarray) -> float:
    """Étendue spectrale : σ₁ - σₙ."""
    sigmas = _svd_values(state)
    if len(sigmas) < 2:
        return np.nan
    return float(sigmas[0] - sigmas[-1])


# Mapping nom → fonction — seul endroit à toucher pour ajouter un signal
_MEASURE_FNS: Dict[str, Callable] = {
    'euclidean_norm'      : euclidean_norm,
    'entropy'             : entropy,
    'effective_rank'      : effective_rank,
    'condition_number_svd': condition_number_svd,
    'singular_value_spread': singular_value_spread,
}


# =============================================================================
# NOMS CATCH22 — référence pour initialisation features NaN si catch22 échoue
# =============================================================================

def _get_catch22_names(catch24: bool = True) -> List[str]:
    """Retourne la liste des noms catch22(+24) via appel dummy."""
    try:
        result = pycatch22.catch22_all([0.0, 1.0, 0.5, 0.2, 0.8, 0.3,
                                        0.9, 0.1, 0.6, 0.4], catch24=catch24)
        return result['names']
    except Exception:
        return []

_CATCH22_NAMES_24 = _get_catch22_names(catch24=True)
_CATCH22_NAMES_22 = _get_catch22_names(catch24=False)

# =============================================================================
# MEASURE STATE — appelé à chaque itération kernel (streaming)
# =============================================================================

def measure_state(state: np.ndarray, func_configs: List) -> Dict[str, float]:
    """
    Applique _MEASURE_FNS sur état courant.

    Args:
        state        : np.ndarray — état courant (un pas de temps)
        func_configs : Liste noms fonctions (depuis config['functions'])

    Returns:
        Dict[str, float] — {fn_name: scalaire} — NaN si fonction échoue

    Notes:
        - Appelé à chaque itération kernel — doit être rapide
        - NaN si fonction échoue — signal physique préservé
        - Pas de catch22 ici — uniquement les scalaires bruts
    """
    measures: Dict[str, float] = {}

    for func_config in func_configs:
        fn_name = func_config if isinstance(func_config, str) else func_config.get('name')
        func = _MEASURE_FNS.get(fn_name)

        if func is None:
            print(f"[WARNING] timeline_lite: fonction de mesure '{fn_name}' inconnue")
            continue

        try:
            val = func(state)
            measures[fn_name] = float(val) if np.isfinite(val) else np.nan
        except Exception:
            measures[fn_name] = np.nan

    return measures


# =============================================================================
# EXTRACT FROM SIGNALS — appelé une fois après la boucle kernel
# =============================================================================

def extract_from_signals(
    signals    : Dict[str, List[float]],
    has_nan_inf: bool,
    last_state : np.ndarray,
    config     : Dict,
) -> Dict[str, float]:
    """
    Normalisation + catch22 + dérivées + santé depuis signaux accumulés.

    Args:
        signals     : {fn_name: [val_t0, val_t1, ...]} — accumulés pendant streaming
        has_nan_inf : bool accumulé pendant streaming (nan/inf détectés dans états)
        last_state  : dernier état capturé — pour is_collapsed
        config      : Dict depuis timeline.yaml

    Returns:
        Dict[str, float] — toutes features timeline + dérivées + santé

    Notes:
        - Même logique que extract() steps 2-5
        - signals sont des listes Python → convertis en np.ndarray ici
        - NaN si signal entièrement inf/nan — signal physique préservé
    """
    catch24      = config.get('catch24', True)
    catch22_names = _CATCH22_NAMES_24 if catch24 else _CATCH22_NAMES_22

    features: Dict[str, float] = {}

    # Convertir listes → np.ndarray
    signals_np: Dict[str, np.ndarray] = {
        fn: np.array(vals, dtype=float) for fn, vals in signals.items()
    }

    T = max((len(v) for v in signals_np.values()), default=0)

    # ─── 1. Normalisation signaux pour catch22 + tagging qualité ─────────────
    #
    # Deux modes de normalisation :
    #   Relatif  (standard) : sign(r) * log1p(|r|)  où r = signal / ref[0]
    #                         Compresse explosions, centre sur évolution depuis t=0
    #   Absolu   (fallback) : sign(x) * log1p(|x|)  sur signal brut
    #                         Utilisé si ref[0] ≈ 0 (ex: entropie d'un état uniforme)
    #
    # Tags toujours finis, toujours présents dans le parquet :
    #   {fn}__signal_finite_ratio  : fraction valeurs finies dans signal brut [0.0, 1.0]
    #   {fn}__signal_norm_absolute : 0.0 = normalisation relative standard
    #                                1.0 = fallback absolu (ref ≈ 0)

    signals_norm: Dict[str, np.ndarray] = {}

    for fn_name, signal in signals_np.items():
        finite_mask = np.isfinite(signal)
        finite_ratio = float(np.mean(finite_mask)) if len(finite_mask) > 0 else 0.0

        features[f'{fn_name}__signal_finite_ratio'] = finite_ratio

        if not finite_mask.any():
            signals_norm[fn_name] = None
            features[f'{fn_name}__signal_norm_absolute'] = 0.0
            continue

        ref = signal[finite_mask][0]

        if abs(ref) < _EPSILON:
            signals_norm[fn_name] = np.sign(signal) * np.log1p(np.abs(signal))
            features[f'{fn_name}__signal_norm_absolute'] = 1.0
        else:
            ratio = signal / ref
            signals_norm[fn_name] = np.sign(ratio) * np.log1p(np.abs(ratio))
            features[f'{fn_name}__signal_norm_absolute'] = 0.0

    # ─── 2. catch22+24 sur signaux normalisés ─────────────────────────────────

    for fn_name, signal_norm in signals_norm.items():
        if signal_norm is None:
            for name in catch22_names:
                features[f'{fn_name}__{name}'] = np.nan
            continue
        signal_list = signal_norm.tolist()
        try:
            result = pycatch22.catch22_all(signal_list, catch24=catch24)
            for name, val in zip(result['names'], result['values']):
                features[f'{fn_name}__{name}'] = float(val) if np.isfinite(val) else np.nan
        except Exception as e:
            print(f"[WARNING] timeline_lite: catch22 échoué sur '{fn_name}': {e}")
            for name in catch22_names:
                features[f'{fn_name}__{name}'] = np.nan

    # ─── 3. Dérivées simples ──────────────────────────────────────────────────

    def _val(signal: np.ndarray, idx: int) -> float:
        if signal is None or len(signal) == 0:
            return np.nan
        v = signal[idx]
        return float(v) if np.isfinite(v) else np.nan

    def _safe(v) -> bool:
        return v is not None and np.isfinite(v)

    def _sig(name: str) -> np.ndarray:
        return signals_np.get(name, np.full(T, np.nan))

    n_sig = _sig('euclidean_norm')
    e_sig = _sig('entropy')
    r_sig = _sig('effective_rank')
    c_sig = _sig('condition_number_svd')
    s_sig = _sig('singular_value_spread')

    n0, nT = _val(n_sig, 0),  _val(n_sig, -1)
    e0, eT = _val(e_sig, 0),  _val(e_sig, -1)
    r0, rT = _val(r_sig, 0),  _val(r_sig, -1)
    c0, cT = _val(c_sig, 0),  _val(c_sig, -1)
    s0, sT = _val(s_sig, 0),  _val(s_sig, -1)

    features['norm_ratio']            = float(nT / (n0 + _EPSILON)) if (_safe(n0) and _safe(nT)) else np.nan
    features['entropy_delta']         = float(eT - e0)               if (_safe(e0) and _safe(eT)) else np.nan
    features['effective_rank_delta']  = float(rT - r0)               if (_safe(r0) and _safe(rT)) else np.nan
    features['spread_ratio']          = float(sT / (s0 + _EPSILON))  if (_safe(s0) and _safe(sT)) else np.nan

    if _safe(c0) and _safe(cT):
        features['log_condition_delta'] = float(np.log(cT + _EPSILON) - np.log(c0 + _EPSILON))
    else:
        features['log_condition_delta'] = np.nan

    features['norm_final']                 = float(nT) if _safe(nT) else np.nan
    features['condition_number_svd_final'] = float(cT) if _safe(cT) else np.nan

    # ─── 4. Métadonnées santé ────────────────────────────────────────────────

    features['has_nan_inf']  = bool(has_nan_inf)
    features['is_collapsed'] = bool(np.std(last_state) < _EPSILON) if last_state is not None else True

    return features


# =============================================================================
# EXTRACT — wrapper rétrocompat (history complet → streaming interne)
# =============================================================================

def extract(history: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Wrapper rétrocompat — reconstruit signals depuis history complet.

    Args:
        history : np.ndarray (T, *dims)
        config  : Dict depuis timeline.yaml

    Notes:
        - Délègue à measure_state() + extract_from_signals()
        - Conserver tant que hub_featuring n'a pas migré
    """
    func_configs = config.get('functions', list(_MEASURE_FNS.keys()))

    signals: Dict[str, List[float]] = {
        fn: [] for fn in [
            f if isinstance(f, str) else f.get('name')
            for f in func_configs
            if _MEASURE_FNS.get(f if isinstance(f, str) else f.get('name')) is not None
        ]
    }

    has_nan_inf = False
    last_state  = None

    for state in history:
        if not np.all(np.isfinite(state)):
            has_nan_inf = True
        measures = measure_state(state, func_configs)
        for fn_name, val in measures.items():
            signals[fn_name].append(val)
        last_state = state

    return extract_from_signals(signals, has_nan_inf, last_state, config)