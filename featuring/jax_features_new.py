"""
featuring/jax_features_new.py

Features informationnelles et géométriques PRC v7 — F1→F7 complets.

Trois fonctions publiques :
  measure_state(...)   → (dict scalaires, A_k_new, P_k_new, sigmas)  [dans lax.scan]
  _post_scan_jax(...)  → dict scalaires JAX shape ()                  [dans _run_fn, vmappable]
  post_scan(...)       → dict floats Python                           [wrapper compat, run_one_jax]

Différences vs jax_features.py :
  1. _post_scan_jax ajoutée — version pure JAX de post_scan, intégrable dans _run_fn
  2. F7 : try/except → lax.cond                (vmappable)
  3. Santé : isinstance(v, float) → jnp.stack  (vmappable)
  4. F2 entropy_production_rate : len() → .shape[0]
  5. FEATURE_NAMES_ORDERED_FOR_HEALTH ajouté
  6. post_scan devient wrapper de _post_scan_jax

Architecture carry étendu :
  carry = (state, key, state_prev, sigmas_prev, A_k, P_k)
  sigmas_prev : carry O1 — évite 1 SVD par pas

Contrainte JAX critique :
  measure_state doit retourner une structure FIXE à chaque appel.
  lax.scan trace la structure au premier appel et l'impose à tous les pas.
  → Toutes les clés présentes à chaque appel.
  → NaN si non applicable (F4 non-différentiable, F3 mode inexistant).

Convention :
  Mode-0 unfolding : M = state.reshape(n0, -1)
  sigmas = SVD(M, compute_uv=False) — réutilisés F1/F2/F3 + DMD
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Callable, List, Tuple

EPS = 1e-8

# =============================================================================
# FEATURE_NAMES — source de vérité colonnes parquet
# =============================================================================

FEATURE_NAMES: List[str] = [

    # F1 — Spectrale (dans-scan → mean/delta/final post-scan)
    'f1_effective_rank_mean',
    'f1_effective_rank_delta',
    'f1_effective_rank_final',
    'f1_spectral_gap_mean',
    'f1_spectral_gap_final',
    'f1_nuclear_frobenius_ratio_mean',
    'f1_nuclear_frobenius_ratio_final',
    'f1_sv_decay_rate_mean',
    'f1_sv_decay_rate_final',
    'f1_rank1_residual_mean',
    'f1_rank1_residual_final',
    'f1_condition_number_mean',
    'f1_condition_number_delta',
    'f1_frob_norm_mean',
    'f1_frob_norm_final',

    # F2 — Informationnelle
    'f2_von_neumann_entropy_mean',
    'f2_von_neumann_entropy_delta',
    'f2_von_neumann_entropy_final',
    'f2_renyi2_entropy_mean',
    'f2_renyi2_entropy_final',
    'f2_shannon_entropy_mean',
    'f2_shannon_entropy_delta',
    'f2_entropy_production_rate',

    # F3 — Enchevêtrement inter-modes
    'f3_entanglement_entropy_mode0_mean',
    'f3_entanglement_entropy_mode0_final',
    'f3_entanglement_entropy_mode1_mean',   # NaN structurel si rank=2
    'f3_entanglement_entropy_mode1_final',  # NaN structurel si rank=2
    'f3_mode_asymmetry_mean',
    'f3_mode_asymmetry_delta',
    'f3_mode_asymmetry_final',
    'f3_inter_mode_sv_var_mean',
    'f3_inter_mode_sv_var_final',

    # F4 — Dynamique locale
    'f4_trace_J_mean',
    'f4_trace_J_std',
    'f4_trace_J_final',
    'f4_jvp_norm_mean',
    'f4_jvp_norm_final',
    'f4_jacobian_asymmetry_mean',
    'f4_jacobian_asymmetry_final',
    'f4_local_lyapunov_mean',
    'f4_local_lyapunov_std',
    'f4_lyapunov_empirical_mean',   # empirique — tous gammas
    'f4_lyapunov_empirical_std',

    # F5 — Transport
    'f5_delta_D_mean',
    'f5_delta_D_total',
    'f5_frob_gradient_mean',
    'f5_frob_gradient_final',
    'f5_bregman_cost_mean',
    'f5_bregman_cost_total',

    # F6 — Causale (post-scan)
    'f6_te_norm_to_entropy',
    'f6_te_entropy_to_asymmetry',
    'f6_te_lyapunov_to_rank',
    'f6_causal_asymmetry_index',

    # F7 — DMD spectral (post-scan)
    'f7_dmd_spectral_radius',
    'f7_dmd_n_complex_pairs',
    'f7_dmd_spectral_entropy',
    'f7_dmd_decay_rate',

    # Post-scan dérivées et autocorr
    'ps_norm_ratio',
    'ps_condition_ratio',
    'ps_rank_delta',
    'ps_frob_delta',
    'ps_first_min_ac_von_neumann',
    'ps_first_min_ac_mode_asymmetry',
    'ps_first_min_ac_frob',
    'ps_pnn40_von_neumann',
    'ps_pnn40_mode_asymmetry',

    # Santé
    'health_has_inf',
    'health_is_collapsed',
]


# =============================================================================
# FEATURES_STRUCTURAL_NAN — source de vérité NaN applicabilité
# =============================================================================
# Ces features retournent NaN pour des raisons d'applicabilité, pas d'erreur.
# L'analyseur lit cette constante pour distinguer NaN structurel de NaN pathologique.
FEATURES_STRUCTURAL_NAN: dict = {
    'f3_entanglement_entropy_mode1_mean' : 'rank_eff == 2',
    'f3_entanglement_entropy_mode1_final': 'rank_eff == 2',
    'f3_inter_mode_sv_var_mean'          : 'rank_eff == 2',
    'f3_inter_mode_sv_var_final'         : 'rank_eff == 2',
    'f5_frob_gradient_mean'              : 'rank_eff == 2',
    'f5_frob_gradient_final'             : 'rank_eff == 2',
    'f4_trace_J_mean'            : 'differentiable == False',
    'f4_trace_J_std'             : 'differentiable == False',
    'f4_trace_J_final'           : 'differentiable == False',
    'f4_jvp_norm_mean'           : 'differentiable == False',
    'f4_jvp_norm_final'          : 'differentiable == False',
    'f4_jacobian_asymmetry_mean' : 'differentiable == False',
    'f4_jacobian_asymmetry_final': 'differentiable == False',
    'f4_local_lyapunov_mean'     : 'differentiable == False',
    'f4_local_lyapunov_std'      : 'differentiable == False',
}


# =============================================================================
# FEATURE_NAMES_ORDERED_FOR_HEALTH
# =============================================================================
# Liste stable des features non-health pour le stack santé dans _post_scan_jax.
# Dérivée de FEATURE_NAMES — doit rester synchronisée avec l'ordre de
# construction du dict dans _post_scan_jax.
FEATURE_NAMES_ORDERED_FOR_HEALTH: List[str] = [
    f for f in FEATURE_NAMES if not f.startswith('health_')
]


# =============================================================================
# HELPERS INTERNES — compilés dans lax.scan
# =============================================================================

def _mode0_unfolding(state: jnp.ndarray) -> jnp.ndarray:
    """Mode-0 unfolding : (n0, prod_autres_dims)."""
    n0 = state.shape[0]
    return state.reshape(n0, -1)


def _svd_no_vectors(M: jnp.ndarray) -> jnp.ndarray:
    """SVD sans vecteurs — pour F1.1→F1.4, F1.6, F2, F3."""
    return jnp.linalg.svd(M, compute_uv=False)


def _svd_with_vectors(M: jnp.ndarray):
    """SVD complète — pour F1.5 uniquement."""
    return jnp.linalg.svd(M, full_matrices=False)


def _entropy_from_probs(p: jnp.ndarray) -> jnp.ndarray:
    """Entropie de Shannon sur distribution de probabilités."""
    return -jnp.sum(p * jnp.log(p + EPS))


def _approx_transfer_entropy(
    signal_A: jnp.ndarray,
    signal_B: jnp.ndarray,
) -> jnp.ndarray:
    """
    Approximation Granger de la transfer entropy A→B.

    TE(A→B) ≈ |corr(A_t, B_{t+1})| - |corr(A_{t+1}, B_t)|
    >0 : A cause B. <0 : B cause A.
    """
    A_lag  = signal_A[:-1]
    B_next = signal_B[1:]
    B_lag  = signal_B[:-1]
    A_next = signal_A[1:]

    def _safe_corr(x, y):
        x_c   = x - jnp.mean(x)
        y_c   = y - jnp.mean(y)
        denom = jnp.sqrt(jnp.sum(x_c**2) * jnp.sum(y_c**2)) + EPS
        return jnp.sum(x_c * y_c) / denom

    te_AB = jnp.abs(_safe_corr(A_lag, B_next))
    te_BA = jnp.abs(_safe_corr(B_lag, A_next))
    return te_AB - te_BA


def _first_min_autocorr(signal: jnp.ndarray) -> jnp.ndarray:
    """
    Premier minimum de l'autocorrélation normalisée.
    Approximation JAX : premier changement de signe de la dérivée.

    Note : lax.dynamic_slice obligatoire — lag est un indice tracé dans vmap.
    """
    s       = signal - jnp.mean(signal)
    T       = s.shape[0]
    max_lag = T // 2
    lags    = jnp.arange(1, max_lag)
    s2      = jnp.sum(s**2) + EPS

    def _ac_at_lag(lag):
        a    = lax.dynamic_slice(s, (0,),   (max_lag,))
        b    = lax.dynamic_slice(s, (lag,), (max_lag,))
        mask = jnp.arange(max_lag) < (T - lag)
        return jnp.sum(a * b * mask) / s2

    ac        = jax.vmap(_ac_at_lag)(lags)
    diffs     = jnp.diff(ac)
    sign_neg  = jnp.argmax(ac < 0)
    sign_flip = jnp.argmax(diffs > 0) + 1
    return jnp.minimum(sign_neg, sign_flip).astype(jnp.float32)


def _dmd_streaming_update(
    sigmas_prev: jnp.ndarray,   # (n_dof,)
    sigmas_curr: jnp.ndarray,   # (n_dof,)
    A_k        : jnp.ndarray,   # (n_dof, n_dof)
    P_k        : jnp.ndarray,   # (n_dof, n_dof)
    forget     : float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mise à jour rank-1 de l'opérateur DMD dans l'espace spectral.
    Online DMD (Hemati et al. 2014) adapté à l'espace des sigmas.
    """
    x     = sigmas_prev
    y     = sigmas_curr
    Px    = P_k @ x
    denom = forget + jnp.dot(x, Px) + EPS
    k     = Px / denom
    e     = y - A_k @ x
    A_k_new = A_k + jnp.outer(e, k)
    P_k_new = (P_k - jnp.outer(Px, Px) / denom) / forget
    return A_k_new, P_k_new


# =============================================================================
# F1 — SPECTRALE (dans-scan)
# =============================================================================

def _f1_spectral(
    state : jnp.ndarray,
    sigmas: jnp.ndarray,
    M     : jnp.ndarray,
) -> dict:
    """F1.1 → F1.6 depuis sigmas + M."""

    frob_norm = jnp.linalg.norm(state)
    n0        = M.shape[0]

    # F1.1 — Rang effectif entropique (Roy & Vetterli 2007)
    p_s           = sigmas / (jnp.sum(sigmas) + EPS)
    effective_rank = jnp.exp(_entropy_from_probs(p_s))

    # F1.2 — Spectral gap
    spectral_gap = (sigmas[0] - sigmas[1]) / (sigmas[0] + EPS)

    # F1.3 — Nuclear / Frobenius ratio
    nuclear_norm            = jnp.sum(sigmas)
    frob_sigmas             = jnp.linalg.norm(sigmas)
    nuclear_frobenius_ratio = nuclear_norm / (jnp.sqrt(n0) * frob_sigmas + EPS)

    # F1.4 — Taux de décroissance spectrale
    log_sigmas    = jnp.log(sigmas + EPS)
    i_vals        = jnp.arange(len(sigmas), dtype=jnp.float32)
    i_mean        = jnp.mean(i_vals)
    s_mean        = jnp.mean(log_sigmas)
    cov_is        = jnp.mean((i_vals - i_mean) * (log_sigmas - s_mean))
    var_i         = jnp.mean((i_vals - i_mean)**2) + EPS
    sv_decay_rate = -cov_is / var_i

    # F1.5 — Rang-1 résiduel
    rank1_residual = jnp.sqrt(jnp.sum(sigmas[1:] ** 2)) / (frob_norm + EPS)

    # F1.6 — Condition number
    condition_number = sigmas[0] / (sigmas[-1] + EPS)

    return {
        'effective_rank'         : effective_rank,
        'spectral_gap'           : spectral_gap,
        'nuclear_frobenius_ratio': nuclear_frobenius_ratio,
        'sv_decay_rate'          : sv_decay_rate,
        'rank1_residual'         : rank1_residual,
        'condition_number'       : condition_number,
        'frob_norm'              : frob_norm,
    }


# =============================================================================
# F2 — INFORMATIONNELLE (dans-scan)
# =============================================================================

def _f2_informational(sigmas: jnp.ndarray, state: jnp.ndarray) -> dict:
    """F2.1 → F2.3 depuis sigmas et état."""

    p_sq        = sigmas**2 / (jnp.sum(sigmas**2) + EPS)
    von_neumann = _entropy_from_probs(p_sq)
    renyi2      = -jnp.log(jnp.sum(p_sq**2) + EPS)
    flat        = jnp.abs(state.reshape(-1))
    flat_norm   = flat / (jnp.sum(flat) + EPS)
    shannon     = _entropy_from_probs(flat_norm)

    return {
        'von_neumann_entropy': von_neumann,
        'renyi2_entropy'     : renyi2,
        'shannon_entropy'    : shannon,
    }


# =============================================================================
# F3 — ENCHEVÊTREMENT INTER-MODES (dans-scan)
# =============================================================================

def _f3_entanglement(state: jnp.ndarray, sigmas_mode0: jnp.ndarray) -> dict:
    """
    F3.1 → F3.3.
    entanglement_entropy_mode1 = NaN structurel si rank=2.
    """
    rank = len(state.shape)   # Python int statique à la compilation

    # F3.1 mode-0
    p_mode0  = sigmas_mode0**2 / (jnp.sum(sigmas_mode0**2) + EPS)
    ee_mode0 = _entropy_from_probs(p_mode0)

    # F3.1 mode-1 — uniquement si rank >= 3
    def _ee_mode1_rank3():
        M1 = jnp.moveaxis(state, 1, 0).reshape(state.shape[1], -1)
        s1 = jnp.linalg.svd(M1, compute_uv=False)
        p1 = s1**2 / (jnp.sum(s1**2) + EPS)
        return _entropy_from_probs(p1)

    ee_mode1 = lax.cond(
        rank >= 3,
        _ee_mode1_rank3,
        lambda: jnp.array(jnp.nan),
    )

    # F3.2 — Mode asymmetry
    n0         = state.shape[0]
    M          = state.reshape(n0, -1)
    min_d      = min(M.shape[0], M.shape[1])   # statique
    M_sq       = M[:min_d, :min_d]
    mode_asymmetry = jnp.linalg.norm(M_sq - M_sq.T) / (jnp.linalg.norm(M) + EPS)

    # F3.3 — Variance inter-mode des σ₁
    sv1_mode0 = sigmas_mode0[0]

    def _sv1_mode1():
        M1 = jnp.moveaxis(state, 1, 0).reshape(state.shape[1], -1)
        return jnp.linalg.svd(M1, compute_uv=False)[0]

    sv1_mode1         = lax.cond(rank >= 3, _sv1_mode1, lambda: sv1_mode0)
    inter_mode_sv_var = jnp.var(jnp.array([sv1_mode0, sv1_mode1]))

    return {
        'entanglement_entropy_mode0': ee_mode0,
        'entanglement_entropy_mode1': ee_mode1,
        'mode_asymmetry'            : mode_asymmetry,
        'inter_mode_sv_var'         : inter_mode_sv_var,
    }


# =============================================================================
# F4 — DYNAMIQUE LOCALE (dans-scan)
# =============================================================================

def _f4_lyapunov_empirical(
    state     : jnp.ndarray,
    state_prev: jnp.ndarray,
    state_next: jnp.ndarray,
) -> jnp.ndarray:
    """Lyapunov empirique — tous gammas, différentiables ou non."""
    delta_curr = jnp.linalg.norm(state_next - state)
    delta_prev = jnp.linalg.norm(state - state_prev) + EPS
    return jnp.log(delta_curr / delta_prev + EPS)


def _f4_hutchinson(
    state   : jnp.ndarray,
    gamma_fn: Callable,
    params  : dict,
    key     : jax.Array,
) -> dict:
    """F4.1 → F4.4 Hutchinson — gammas différentiables uniquement."""
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, state.shape)

    _, jvp_v = jax.jvp(
        lambda s: gamma_fn(s, params, key),
        (state,),
        (v,),
    )

    trace_J        = jnp.sum(v * jvp_v)
    jvp_norm       = jnp.linalg.norm(jvp_v)
    local_lyapunov = jnp.log(jvp_norm + EPS)

    _, jvp_vT = jax.jvp(
        lambda s: gamma_fn(s, params, key),
        (state,),
        (jvp_v,),
    )
    jacobian_asymmetry = jnp.linalg.norm(jvp_v - jvp_vT) / (jvp_norm + EPS)

    return {
        'trace_J'           : trace_J,
        'jvp_norm'          : jvp_norm,
        'jacobian_asymmetry': jacobian_asymmetry,
        'local_lyapunov'    : local_lyapunov,
    }


def _f4_dynamics(
    state            : jnp.ndarray,
    state_prev       : jnp.ndarray,
    state_next       : jnp.ndarray,
    gamma_fn         : Callable,
    params           : dict,
    key              : jax.Array,
    is_differentiable: bool,
) -> dict:
    """
    Dispatch F4 selon is_differentiable (Python bool statique — static_argnums).
    Deux graphes XLA distincts compilés, zéro branchement dynamique.
    """
    lyapunov_empirical = _f4_lyapunov_empirical(state, state_prev, state_next)

    if is_differentiable:
        hutch = _f4_hutchinson(state, gamma_fn, params, key)
    else:
        nan   = jnp.array(jnp.nan)
        hutch = {
            'trace_J'           : nan,
            'jvp_norm'          : nan,
            'jacobian_asymmetry': nan,
            'local_lyapunov'    : nan,
        }

    return {**hutch, 'lyapunov_empirical': lyapunov_empirical}


# =============================================================================
# F5 — TRANSPORT (dans-scan)
# =============================================================================

def _f5_transport(
    state     : jnp.ndarray,
    state_prev: jnp.ndarray,
) -> dict:
    """F5.1 → F5.3 — nécessite state_prev du carry."""

    frob_state = jnp.linalg.norm(state) + EPS
    delta_D    = jnp.linalg.norm(state - state_prev) / frob_state

    n0 = state.shape[0]
    M  = state.reshape(n0, -1)

    def _row_var_ratio():
        row_norms_m0 = jnp.linalg.norm(M, axis=1)
        M1           = jnp.moveaxis(state, 1, 0).reshape(state.shape[1], -1)
        row_norms_m1 = jnp.linalg.norm(M1, axis=1)
        return jnp.std(row_norms_m0) / (jnp.std(row_norms_m1) + EPS)

    # NaN structurel si rank=2 (cf. FEATURES_STRUCTURAL_NAN)
    frob_gradient = lax.cond(
        len(state.shape) >= 3,
        _row_var_ratio,
        lambda: jnp.array(jnp.nan),
    )

    bregman_cost = jnp.sum((state - state_prev)**2) / (frob_state**2 + EPS)

    return {
        'delta_D'      : delta_D,
        'frob_gradient': frob_gradient,
        'bregman_cost' : bregman_cost,
    }


# =============================================================================
# measure_state — point d'entrée dans lax.scan
# =============================================================================

def measure_state(
    state            : jnp.ndarray,
    state_next       : jnp.ndarray,
    state_prev       : jnp.ndarray,
    gamma_fn         : Callable,
    params           : dict,
    key              : jax.Array,
    A_k              : jnp.ndarray,
    P_k              : jnp.ndarray,
    is_differentiable: bool = True,
    sigmas_prev      : jnp.ndarray = None,
) -> Tuple[dict, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calcule toutes les mesures du pas courant + met à jour le carry DMD.

    Appelée à l'intérieur de lax.scan — structure de retour FIXE.
    Tous les scalaires shape () présents à chaque appel.

    Args:
        state             : Tenseur au pas t
        state_next        : Tenseur au pas t+1
        state_prev        : Tenseur au pas t-1 (carry)
        gamma_fn          : Fonction gamma (statique)
        params            : Params gamma (dynamique)
        key               : Subkey PRNG du pas courant
        A_k               : Carry DMD estimateur (n_dof, n_dof)
        P_k               : Carry DMD covariance inverse (n_dof, n_dof)
        is_differentiable : Python bool statique (static_argnums dans _run_jit)
        sigmas_prev       : sigmas au pas t-1 (carry O1) — évite 1 SVD/pas

    Returns:
        (measures, A_k_new, P_k_new, sigmas)
        measures : dict 22 scalaires shape () — structure fixe
        sigmas   : valeurs singulières du pas courant — à passer en carry
    """
    M      = _mode0_unfolding(state)
    sigmas = _svd_no_vectors(M)

    f1 = _f1_spectral(state, sigmas, M)
    f2 = _f2_informational(sigmas, state)
    f3 = _f3_entanglement(state, sigmas)
    f4 = _f4_dynamics(state, state_prev, state_next, gamma_fn, params, key, is_differentiable)
    f5 = _f5_transport(state, state_prev)

    A_k_new, P_k_new = _dmd_streaming_update(sigmas_prev, sigmas, A_k, P_k)

    measures = {
        'f1_effective_rank'            : f1['effective_rank'],
        'f1_spectral_gap'              : f1['spectral_gap'],
        'f1_nuclear_frobenius_ratio'   : f1['nuclear_frobenius_ratio'],
        'f1_sv_decay_rate'             : f1['sv_decay_rate'],
        'f1_rank1_residual'            : f1['rank1_residual'],
        'f1_condition_number'          : f1['condition_number'],
        'f1_frob_norm'                 : f1['frob_norm'],
        'f2_von_neumann_entropy'       : f2['von_neumann_entropy'],
        'f2_renyi2_entropy'            : f2['renyi2_entropy'],
        'f2_shannon_entropy'           : f2['shannon_entropy'],
        'f3_entanglement_entropy_mode0': f3['entanglement_entropy_mode0'],
        'f3_entanglement_entropy_mode1': f3['entanglement_entropy_mode1'],
        'f3_mode_asymmetry'            : f3['mode_asymmetry'],
        'f3_inter_mode_sv_var'         : f3['inter_mode_sv_var'],
        'f4_trace_J'                   : f4['trace_J'],
        'f4_jvp_norm'                  : f4['jvp_norm'],
        'f4_jacobian_asymmetry'        : f4['jacobian_asymmetry'],
        'f4_local_lyapunov'            : f4['local_lyapunov'],
        'f4_lyapunov_empirical'        : f4['lyapunov_empirical'],
        'f5_delta_D'                   : f5['delta_D'],
        'f5_frob_gradient'             : f5['frob_gradient'],
        'f5_bregman_cost'              : f5['bregman_cost'],
    }

    return measures, A_k_new, P_k_new, sigmas


# =============================================================================
# _f7_jax — F7 DMD spectral, pure JAX, vmappable
# =============================================================================

def _f7_safe(A_k: jnp.ndarray) -> Tuple:
    """F7 sur A_k finie — branche lax.cond.

    Sanitization obligatoire avant eigvals : sous vmap, JAX évalue les deux
    branches de lax.cond pour tous les éléments du batch avant de sélectionner.
    cuSolver plante sur inf/nan même si la branche _f7_nan sera retournée.
    jnp.where remplace inf/nan par 0 — le résultat est masqué par lax.cond
    pour les éléments où A_k n'est pas finie.
    """
    A_k_clean       = jnp.where(jnp.isfinite(A_k), A_k, jnp.zeros_like(A_k))
    eigenvalues     = jnp.linalg.eigvals(A_k_clean)
    abs_eigs        = jnp.abs(eigenvalues)
    spectral_radius = jnp.max(abs_eigs)
    n_complex       = jnp.sum(jnp.abs(jnp.imag(eigenvalues)) > 1e-4).astype(jnp.float32)
    p_dmd           = abs_eigs / (jnp.sum(abs_eigs) + EPS)
    dmd_entropy     = _entropy_from_probs(p_dmd)
    idx_dom         = jnp.argmax(abs_eigs)
    decay           = jnp.real(jnp.log(eigenvalues[idx_dom] + EPS))
    return spectral_radius, n_complex, dmd_entropy, decay


def _f7_nan(_: jnp.ndarray) -> Tuple:
    """F7 fallback NaN — branche lax.cond si A_k non finie."""
    nan = jnp.array(jnp.nan)
    return nan, jnp.array(0.0), nan, nan


# =============================================================================
# _post_scan_jax — version pure JAX, vmappable
# =============================================================================

def _post_scan_jax(
    signals   : dict,
    last_state: jnp.ndarray,
    A_k       : jnp.ndarray,
) -> dict:
    """
    Agrège les signaux dans-scan + calcule F6, F7, dérivées, santé.

    Version pure JAX — tracée et vmappable.
    Intégrable directement dans _run_fn pour que scan + post_scan = 1 kernel XLA.
    Signals shape (T,) consommés ici — jamais matérialisés hors du kernel en mode batch.

    Args:
        signals    : {feature_name: jnp.array(T,)} — signaux depuis lax.scan
        last_state : Dernier état — shape statique par kernel_group
        A_k        : Matrice DMD finale (n_dof, n_dof)

    Returns:
        dict {feature_name: jnp.array shape ()} — scalaires JAX, 68 clés
        Convertir en floats Python via {k: float(v) for k, v in out.items()}
        uniquement au moment de matérialiser (sync GPU→CPU).
    """
    def _s(key):    return signals[key]
    def _mean(key): return jnp.mean(_s(key))
    def _std(key):  return jnp.std(_s(key))
    def _fin(key):  return _s(key)[-1]
    def _dlt(key):  return _s(key)[-1] - _s(key)[0]

    # -------------------------------------------------------------------------
    # F1
    # -------------------------------------------------------------------------
    out = {
        'f1_effective_rank_mean'          : _mean('f1_effective_rank'),
        'f1_effective_rank_delta'         : _dlt('f1_effective_rank'),
        'f1_effective_rank_final'         : _fin('f1_effective_rank'),
        'f1_spectral_gap_mean'            : _mean('f1_spectral_gap'),
        'f1_spectral_gap_final'           : _fin('f1_spectral_gap'),
        'f1_nuclear_frobenius_ratio_mean' : _mean('f1_nuclear_frobenius_ratio'),
        'f1_nuclear_frobenius_ratio_final': _fin('f1_nuclear_frobenius_ratio'),
        'f1_sv_decay_rate_mean'           : _mean('f1_sv_decay_rate'),
        'f1_sv_decay_rate_final'          : _fin('f1_sv_decay_rate'),
        'f1_rank1_residual_mean'          : _mean('f1_rank1_residual'),
        'f1_rank1_residual_final'         : _fin('f1_rank1_residual'),
        'f1_condition_number_mean'        : _mean('f1_condition_number'),
        'f1_condition_number_delta'       : _dlt('f1_condition_number'),
        'f1_frob_norm_mean'               : _mean('f1_frob_norm'),
        'f1_frob_norm_final'              : _fin('f1_frob_norm'),
    }

    # -------------------------------------------------------------------------
    # F2
    # T = signal.shape[0] est statique par kernel_group (même max_it pour tous les
    # samples du groupe) — .shape[0] est correct sous tracé, len() ne l'est pas.
    # -------------------------------------------------------------------------
    vn_sig = _s('f2_von_neumann_entropy')
    T      = vn_sig.shape[0]
    out.update({
        'f2_von_neumann_entropy_mean' : _mean('f2_von_neumann_entropy'),
        'f2_von_neumann_entropy_delta': _dlt('f2_von_neumann_entropy'),
        'f2_von_neumann_entropy_final': _fin('f2_von_neumann_entropy'),
        'f2_renyi2_entropy_mean'      : _mean('f2_renyi2_entropy'),
        'f2_renyi2_entropy_final'     : _fin('f2_renyi2_entropy'),
        'f2_shannon_entropy_mean'     : _mean('f2_shannon_entropy'),
        'f2_shannon_entropy_delta'    : _dlt('f2_shannon_entropy'),
        'f2_entropy_production_rate'  : (vn_sig[-1] - vn_sig[0]) / (T + EPS),
    })

    # -------------------------------------------------------------------------
    # F3
    # -------------------------------------------------------------------------
    out.update({
        'f3_entanglement_entropy_mode0_mean' : _mean('f3_entanglement_entropy_mode0'),
        'f3_entanglement_entropy_mode0_final': _fin('f3_entanglement_entropy_mode0'),
        'f3_entanglement_entropy_mode1_mean' : _mean('f3_entanglement_entropy_mode1'),
        'f3_entanglement_entropy_mode1_final': _fin('f3_entanglement_entropy_mode1'),
        'f3_mode_asymmetry_mean'             : _mean('f3_mode_asymmetry'),
        'f3_mode_asymmetry_delta'            : _dlt('f3_mode_asymmetry'),
        'f3_mode_asymmetry_final'            : _fin('f3_mode_asymmetry'),
        'f3_inter_mode_sv_var_mean'          : _mean('f3_inter_mode_sv_var'),
        'f3_inter_mode_sv_var_final'         : _fin('f3_inter_mode_sv_var'),
    })

    # -------------------------------------------------------------------------
    # F4
    # -------------------------------------------------------------------------
    out.update({
        'f4_trace_J_mean'            : _mean('f4_trace_J'),
        'f4_trace_J_std'             : _std('f4_trace_J'),
        'f4_trace_J_final'           : _fin('f4_trace_J'),
        'f4_jvp_norm_mean'           : _mean('f4_jvp_norm'),
        'f4_jvp_norm_final'          : _fin('f4_jvp_norm'),
        'f4_jacobian_asymmetry_mean' : _mean('f4_jacobian_asymmetry'),
        'f4_jacobian_asymmetry_final': _fin('f4_jacobian_asymmetry'),
        'f4_local_lyapunov_mean'     : _mean('f4_local_lyapunov'),
        'f4_local_lyapunov_std'      : _std('f4_local_lyapunov'),
        'f4_lyapunov_empirical_mean' : _mean('f4_lyapunov_empirical'),
        'f4_lyapunov_empirical_std'  : _std('f4_lyapunov_empirical'),
    })

    # -------------------------------------------------------------------------
    # F5
    # -------------------------------------------------------------------------
    out.update({
        'f5_delta_D_mean'       : _mean('f5_delta_D'),
        'f5_delta_D_total'      : jnp.sum(_s('f5_delta_D')),
        'f5_frob_gradient_mean' : _mean('f5_frob_gradient'),
        'f5_frob_gradient_final': _fin('f5_frob_gradient'),
        'f5_bregman_cost_mean'  : _mean('f5_bregman_cost'),
        'f5_bregman_cost_total' : jnp.sum(_s('f5_bregman_cost')),
    })

    # -------------------------------------------------------------------------
    # F6 — Transfer entropy (pure JAX, vmappable)
    # -------------------------------------------------------------------------
    frob_sig = _s('f1_frob_norm')
    asym_sig = _s('f3_mode_asymmetry')
    lyap_sig = _s('f4_lyapunov_empirical')
    rank_sig = _s('f1_effective_rank')

    te_ne = _approx_transfer_entropy(frob_sig, vn_sig)
    te_ea = _approx_transfer_entropy(vn_sig, asym_sig)
    te_lr = _approx_transfer_entropy(lyap_sig, rank_sig)

    out.update({
        'f6_te_norm_to_entropy'      : te_ne,
        'f6_te_entropy_to_asymmetry' : te_ea,
        'f6_te_lyapunov_to_rank'     : te_lr,
        'f6_causal_asymmetry_index'  : jnp.mean(jnp.abs(jnp.array([te_ne, te_ea, te_lr]))),
    })

    # -------------------------------------------------------------------------
    # F7 — DMD spectral via lax.cond (vmappable, remplace try/except)
    # Les deux branches doivent retourner des arrays de même dtype et shape.
    # _f7_safe et _f7_nan sont définies au niveau module (pas de closure).
    # -------------------------------------------------------------------------
    spectral_radius, n_complex, dmd_entropy, decay = lax.cond(
        jnp.all(jnp.isfinite(A_k)),
        _f7_safe,
        _f7_nan,
        A_k,
    )

    out.update({
        'f7_dmd_spectral_radius' : spectral_radius,
        'f7_dmd_n_complex_pairs' : n_complex,
        'f7_dmd_spectral_entropy': dmd_entropy,
        'f7_dmd_decay_rate'      : decay,
    })

    # -------------------------------------------------------------------------
    # Post-scan dérivées et autocorr
    # -------------------------------------------------------------------------
    out.update({
        'ps_norm_ratio'      : _s('f1_frob_norm')[-1] / (_s('f1_frob_norm')[0] + EPS),
        'ps_condition_ratio' : _s('f1_condition_number')[-1] / (_s('f1_condition_number')[0] + EPS),
        'ps_rank_delta'      : _dlt('f1_effective_rank'),
        'ps_frob_delta'      : _dlt('f1_frob_norm'),
        'ps_first_min_ac_von_neumann'   : _first_min_autocorr(vn_sig),
        'ps_first_min_ac_mode_asymmetry': _first_min_autocorr(asym_sig),
        'ps_first_min_ac_frob'          : _first_min_autocorr(frob_sig),
        'ps_pnn40_von_neumann'    : jnp.mean(
            jnp.abs(jnp.diff(vn_sig)) > 0.4 * jnp.std(vn_sig)
        ),
        'ps_pnn40_mode_asymmetry' : jnp.mean(
            jnp.abs(jnp.diff(asym_sig)) > 0.4 * jnp.std(asym_sig)
        ),
    })

    # -------------------------------------------------------------------------
    # Santé
    # health_has_inf : détectée sur les 66 features non-health via jnp.stack.
    # Remplace isinstance(v, float) non traçable de l'ancienne version.
    # NaN structurels (FEATURES_STRUCTURAL_NAN) ignorés volontairement —
    # ils signalent l'applicabilité, pas une erreur de calcul.
    # -------------------------------------------------------------------------
    health_stack = jnp.stack([out[k] for k in FEATURE_NAMES_ORDERED_FOR_HEALTH])
    has_inf      = jnp.any(jnp.isinf(health_stack))
    is_collapsed = jnp.std(last_state) < EPS

    out['health_has_inf']      = has_inf.astype(jnp.float32)
    out['health_is_collapsed'] = is_collapsed.astype(jnp.float32)

    return out


# =============================================================================
# post_scan — wrapper de compatibilité (run_one_jax, usage scalaire)
# =============================================================================

def post_scan(
    signals   : dict,
    last_state: jnp.ndarray,
    A_k       : jnp.ndarray,
    P_k       : jnp.ndarray,   # conservé pour compatibilité — non utilisé en post-scan
) -> dict:
    """
    Wrapper de _post_scan_jax pour usage scalaire (run_one_jax).

    Retourne un dict de floats Python purs pour le parquet.
    float() force la matérialisation GPU→CPU — block_until_ready implicite.

    En mode batch (hub_running), utiliser directement _post_scan_jax intégré
    dans _run_fn — sync via np.array() dans _sync_and_flush.
    """
    features_jax = _post_scan_jax(signals, last_state, A_k)
    return {k: float(v) for k, v in features_jax.items()}