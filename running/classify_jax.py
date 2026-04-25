"""
Classification, masque adaptatif et features P1 — tout JAX/GPU.

Ce module opère entre P1 et P2, sur les timelines qui sont encore sur GPU.
Aucun transfert CPU. Les résultats (statuts, masque) sont consommés
directement par P2 sur GPU.

Architecture compilation:
    Toute la classification (classify + characterize + mask + features)
    est regroupée dans UNE SEULE fonction JIT (classify_and_mask).
    Raison: chaque SubBatchProcess est un processus séparé sans cache JIT
    partagé. Une compilation unique au lieu de 5 réduit l'overhead.

@ROLE    Classification et masque adaptatif — GPU, entre P1 et P2
@LAYER   running

@EXPORTS
  STATUS_OK, STATUS_TRUNCATED, STATUS_EXPLOSION, STATUS_COLLAPSED → int
  REGIME_*, STATUS_NAMES, REGIME_NAMES → constantes
  classify_and_mask(frob, is_finite, last_states, cos_dissim, delta_D, max_it)
      → (statuses, t_effectives, regimes, periods, mask, p1_features, mask_features)

@CONFORMITY
  OK    Zéro transfert CPU entre P1 et P2 (D3)
  OK    Constantes depuis features_registry (P4)
  OK    Une seule compilation JIT par processus
"""

import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

from running.features_registry import (
    EPS,
    EXPLOSION_FACTOR,
    EXPLOSION_MONOTONE_WINDOW,
    COLLAPSE_RELATIVE_THRESHOLD,
    FLAT_CV_THRESHOLD,
    TUKEY_FENCE_MULT,
    MONOTONE_TOL,
    AUTOCORR_MAX_LAG_DIVISOR,
    MIN_SVD_POINTS,
    MIN_SVD_FRACTION,
    ZONE_MARGIN_MIN,
    ZONE_MARGIN_FRACTION,
    MIN_ABS_VARIATION,
)


# =========================================================================
# CONSTANTES D'ENCODAGE
# =========================================================================

STATUS_OK = 0
STATUS_TRUNCATED = 1
STATUS_EXPLOSION = 2
STATUS_COLLAPSED = 3

STATUS_NAMES = {
    STATUS_OK: 'OK',
    STATUS_TRUNCATED: 'OK_TRUNCATED',
    STATUS_EXPLOSION: 'EXPLOSION',
    STATUS_COLLAPSED: 'COLLAPSED',
}

REGIME_FLAT = 0
REGIME_OSCILLATING = 1
REGIME_TRANSITIONAL = 2
REGIME_EXPLOSIVE = 3
REGIME_MIXED = 4

REGIME_NAMES = {
    REGIME_FLAT: 'FLAT',
    REGIME_OSCILLATING: 'OSCILLATING',
    REGIME_TRANSITIONAL: 'TRANSITIONAL',
    REGIME_EXPLOSIVE: 'EXPLOSIVE',
    REGIME_MIXED: 'MIXED',
}


# =========================================================================
# HELPERS
# =========================================================================

def _nan_safe_mean(values, valid_mask):
    """Moyenne sur les éléments valides. (B, T) → (B,), (B,)."""
    safe = jnp.where(valid_mask, values, 0.0)
    count = jnp.sum(valid_mask.astype(jnp.float32), axis=1)
    return safe.sum(axis=1) / (count + EPS), count


def _nan_safe_std(values, valid_mask, mean_b):
    """Écart-type sur les éléments valides."""
    safe = jnp.where(valid_mask, (values - mean_b[:, None])**2, 0.0)
    count = jnp.sum(valid_mask.astype(jnp.float32), axis=1)
    return jnp.sqrt(safe.sum(axis=1) / (count + EPS))


def _percentile_via_sort(values, valid_mask, q):
    """Percentile en JAX via sort + indexation."""
    sortable = jnp.where(valid_mask, values, jnp.inf)
    sorted_v = jnp.sort(sortable, axis=1)
    n_valid = jnp.sum(valid_mask.astype(jnp.int32), axis=1)
    idx = jnp.clip((n_valid * q).astype(jnp.int32), 0, sorted_v.shape[1] - 1)
    return sorted_v[jnp.arange(values.shape[0]), idx]


# =========================================================================
# POINT D'ENTRÉE UNIQUE — UNE SEULE COMPILATION JIT
#
# max_it est static car utilisé dans des opérations Python-level
# (max(), int(), range(), linspace).
# =========================================================================

@partial(jit, static_argnums=(5,))
def classify_and_mask(frob, is_finite, last_states, cos_dissim, delta_D, max_it):
    """
    Classification complète: statuts + régimes + masque + features.
    UNE SEULE compilation JIT, UNE SEULE trace XLA.

    Args:
        frob:         (B, T) float32
        is_finite:    (B, T) float32 (1.0/0.0)
        last_states:  (B, *state_shape)
        cos_dissim:   (B, T) float32
        delta_D:      (B, T) float32
        max_it:       int (static)

    Returns: tuple de 7 éléments:
        statuses, t_effectives, regimes, periods, mask, p1_features, mask_features
    """
    B, T = frob.shape
    time_grid = jnp.arange(T)[None, :]

    # ─────────────────────────────────────────────────────────────
    # 4a — CLASSIFICATION DES RUNS
    # ─────────────────────────────────────────────────────────────

    # EXPLOSION: premier step non-fini
    is_finite_bool = is_finite > 0.5
    all_finite = jnp.all(is_finite_bool, axis=1)
    has_nonfinite = ~all_finite
    t_first_nonfinite = jnp.argmin(is_finite_bool.astype(jnp.int32), axis=1)
    t_eff_explosion = jnp.where(has_nonfinite, t_first_nonfinite, T)

    # COLLAPSED: état final quasi-constant
    flat = last_states.reshape(B, -1)
    abs_mean = jnp.mean(jnp.abs(flat), axis=1)
    std_val = jnp.std(flat, axis=1)
    ratio = std_val / (abs_mean + EPS)
    is_collapsed = (ratio < COLLAPSE_RELATIVE_THRESHOLD) & all_finite

    # OK_TRUNCATED: explosion économique
    half = T // 2
    frob_baseline = jnp.median(frob[:, :half], axis=1)
    threshold = frob_baseline * EXPLOSION_FACTOR
    exceeds = frob > threshold[:, None]
    any_exceeds = jnp.any(exceeds, axis=1)
    t_first_exceed = jnp.argmax(exceeds.astype(jnp.int32), axis=1)

    # Fenêtre monotone
    M = min(EXPLOSION_MONOTONE_WINDOW, T - 1)
    frob_diff = jnp.diff(frob, axis=1)
    is_growing = (frob_diff > 0).astype(jnp.float32)
    cum_growing = jnp.cumsum(is_growing, axis=1)
    shifted = jnp.concatenate([jnp.zeros((B, M), dtype=jnp.float32),
                               cum_growing[:, :-M]], axis=1)
    window_sums = cum_growing - shifted
    is_monotone_at = window_sums >= M
    safe_idx = jnp.clip(t_first_exceed - 1, 0, T - 2)
    monotone_at_exceed = is_monotone_at[jnp.arange(B), safe_idx]
    is_truncated = any_exceeds & monotone_at_exceed & all_finite & ~is_collapsed
    t_eff_truncated = t_first_exceed

    # Cascade de priorité
    statuses = jnp.full(B, STATUS_OK, dtype=jnp.int32)
    t_effectives = jnp.full(B, T, dtype=jnp.int32)
    statuses = jnp.where(is_truncated, STATUS_TRUNCATED, statuses)
    t_effectives = jnp.where(is_truncated, t_eff_truncated, t_effectives)
    statuses = jnp.where(is_collapsed, STATUS_COLLAPSED, statuses)
    statuses = jnp.where(has_nonfinite, STATUS_EXPLOSION, statuses)
    t_effectives = jnp.where(has_nonfinite, t_eff_explosion, t_effectives)

    valid = time_grid < t_effectives[:, None]

    # ─────────────────────────────────────────────────────────────
    # 4b — CARACTÉRISATION + DÉTECTION DE PÉRIODE (FFT)
    #
    # L'autocorrélation via FFT remplace la boucle Python sur les lags.
    # Coût: O(T log T) en une seule opération XLA vs O(max_lag × T)
    # en max_lag opérations séquentielles.
    # ─────────────────────────────────────────────────────────────

    mean_cd, count_cd = _nan_safe_mean(cos_dissim, valid)
    std_cd = _nan_safe_std(cos_dissim, valid, mean_cd)
    cv_cd = std_cd / (mean_cd + EPS)

    # Autocorrélation via FFT
    max_lag = max_it // AUTOCORR_MAX_LAG_DIVISOR

    centered = jnp.where(valid, cos_dissim - mean_cd[:, None], 0.0)
    n_fft = 2 * T
    F = jnp.fft.rfft(centered, n=n_fft, axis=1)
    power = F * jnp.conj(F)
    ac_full = jnp.fft.irfft(power, n=n_fft, axis=1)[:, :T]

    energy = ac_full[:, 0:1] + EPS
    ac = ac_full / energy
    ac = ac[:, :max_lag]

    # Masquer les lags invalides par run
    max_lag_per_run = t_effectives // AUTOCORR_MAX_LAG_DIVISOR
    lag_grid = jnp.arange(max_lag)[None, :]
    ac = jnp.where(lag_grid < max_lag_per_run[:, None], ac, -1.0)

    # Détection de maxima locaux: pic au-dessus de 0.3
    left = ac[:, :-2]
    center = ac[:, 1:-1]
    right = ac[:, 2:]
    is_peak = (center > left) & (center > right) & (center > 0.3)
    has_peak = jnp.any(is_peak, axis=1)
    first_peak = jnp.argmax(is_peak.astype(jnp.int32), axis=1) + 1
    periods = jnp.where(has_peak, first_peak.astype(jnp.float32), jnp.nan)

    # Tukey fence pour TRANSITIONAL
    cd_valid = jnp.where(valid, cos_dissim, 0.0)
    diff_cd = jnp.abs(jnp.diff(cd_valid, axis=1))
    valid_diff = valid[:, :-1] & valid[:, 1:]
    diff_cd = jnp.where(valid_diff, diff_cd, 0.0)

    q1_cd = _percentile_via_sort(diff_cd, valid_diff, 0.25)
    q3_cd = _percentile_via_sort(diff_cd, valid_diff, 0.75)
    iqr_cd = q3_cd - q1_cd
    fence_cd = q3_cd + TUKEY_FENCE_MULT * iqr_cd
    outliers_cd = valid_diff & (diff_cd > fence_cd[:, None])
    has_outliers = jnp.any(outliers_cd, axis=1)

    has_period = ~jnp.isnan(periods)

    # Cascade des régimes
    regimes = jnp.full(B, REGIME_MIXED, dtype=jnp.int32)
    regimes = jnp.where(has_outliers, REGIME_TRANSITIONAL, regimes)
    regimes = jnp.where(has_period, REGIME_OSCILLATING, regimes)
    regimes = jnp.where(
        (cv_cd < FLAT_CV_THRESHOLD) & ~has_period, REGIME_FLAT, regimes)
    regimes = jnp.where(
        statuses == STATUS_TRUNCATED, REGIME_EXPLOSIVE, regimes)

    # ─────────────────────────────────────────────────────────────
    # 4c — MASQUE ADAPTATIF
    # ─────────────────────────────────────────────────────────────

    # 1. Plancher uniforme
    K_floor = max(MIN_SVD_POINTS, int(max_it * MIN_SVD_FRACTION))
    floor_indices = jnp.linspace(0, max_it - 1, K_floor).astype(jnp.int32)
    floor_mask_1d = jnp.zeros(T, dtype=jnp.bool_)
    floor_mask_1d = floor_mask_1d.at[floor_indices].set(True)
    mask = jnp.broadcast_to(floor_mask_1d[None, :], (B, T))
    mask = mask | False  # copie mutable

    # 2. Tukey fence sur |diff(delta_D)| et |diff(cos_dissim)|
    dd_clean = jnp.where(valid & jnp.isfinite(delta_D), delta_D, 0.0)
    cd_clean = jnp.where(valid & jnp.isfinite(cos_dissim), cos_dissim, 0.0)

    abs_diff_dd = jnp.abs(jnp.diff(dd_clean, axis=1))
    abs_diff_cd = jnp.abs(jnp.diff(cd_clean, axis=1))
    abs_diff_dd = jnp.where(abs_diff_dd < MIN_ABS_VARIATION, 0.0, abs_diff_dd)
    abs_diff_cd = jnp.where(abs_diff_cd < MIN_ABS_VARIATION, 0.0, abs_diff_cd)

    q1_dd = _percentile_via_sort(abs_diff_dd, valid_diff, 0.25)
    q3_dd = _percentile_via_sort(abs_diff_dd, valid_diff, 0.75)
    iqr_dd = q3_dd - q1_dd
    fence_dd = q3_dd + TUKEY_FENCE_MULT * iqr_dd

    outliers_dd = valid_diff & (abs_diff_dd > fence_dd[:, None])
    outliers = outliers_dd | outliers_cd

    outlier_points = jnp.zeros((B, T), dtype=jnp.bool_)
    outlier_points = outlier_points.at[:, :-1].set(
        outlier_points[:, :-1] | outliers)
    outlier_points = outlier_points.at[:, 1:].set(
        outlier_points[:, 1:] | outliers)

    mask = mask | outlier_points

    # 3. Dilatation avec marge
    margin = max(ZONE_MARGIN_MIN, int(max_it * ZONE_MARGIN_FRACTION))
    dilated = mask | False
    for shift in range(1, margin + 1):
        dilated = dilated.at[:, shift:].set(dilated[:, shift:] | mask[:, :-shift])
        dilated = dilated.at[:, :-shift].set(dilated[:, :-shift] | mask[:, shift:])
    mask = dilated

    # 4. Runs monotones: 2 points seulement
    #    Un run est monotone seulement si FROB ET COS_DISSIM sont tous deux
    #    quasi-monotones. Un run avec frob monotone mais cos_dissim actif
    #    a une structure interne qui évolue → on garde le masque complet.

    frob_valid = jnp.where(valid, frob, 0.0)
    frob_diffs = jnp.diff(frob_valid, axis=1)
    signs_frob = jnp.sign(frob_diffs)
    signs_frob = jnp.where(jnp.abs(frob_diffs) < EPS, 0.0, signs_frob)
    changes_frob = jnp.sum(
        jnp.abs(jnp.diff(signs_frob, axis=1)) > 0, axis=1)

    cd_diffs = jnp.diff(cd_clean, axis=1)
    signs_cd = jnp.sign(cd_diffs)
    signs_cd = jnp.where(jnp.abs(cd_diffs) < EPS, 0.0, signs_cd)
    changes_cd = jnp.sum(
        jnp.abs(jnp.diff(signs_cd, axis=1)) > 0, axis=1)

    is_monotone = (changes_frob < MONOTONE_TOL) & (changes_cd < MONOTONE_TOL)

    mono_mask = jnp.zeros((B, T), dtype=jnp.bool_)
    mono_mask = mono_mask.at[:, 0].set(True)
    last_valid_idx = jnp.clip(t_effectives - 1, 0, T - 1)
    mono_mask = mono_mask.at[jnp.arange(B), last_valid_idx].set(True)

    mask = jnp.where(is_monotone[:, None], mono_mask, mask)

    # 5. Points oscillatoires — seulement pour les runs OK avec période
    has_period_valid = has_period & (statuses == STATUS_OK)
    period_safe = jnp.where(has_period_valid, periods, 1.0)
    remainder = time_grid % period_safe[:, None]
    near_multiple = (remainder < 1.5) | (remainder > period_safe[:, None] - 1.5)
    osc_points = has_period_valid[:, None] & near_multiple & valid
    mask = mask | osc_points

    # 6. Protections finales
    mask = mask & valid
    is_excluded = (
        (statuses == STATUS_EXPLOSION) | (statuses == STATUS_COLLAPSED))
    mask = mask & ~is_excluded[:, None]

    # ─────────────────────────────────────────────────────────────
    # 4d — FEATURES P1
    # ─────────────────────────────────────────────────────────────

    d_cd = jnp.diff(cd_valid, axis=1)
    signs_d = jnp.sign(d_cd)
    signs_d = jnp.where((jnp.abs(d_cd) < EPS) | ~valid_diff, 0.0, signs_d)
    sign_diff_d = jnp.abs(jnp.diff(signs_d, axis=1))
    nzc = jnp.sum(sign_diff_d > 0, axis=1).astype(jnp.float32) / 2.0

    p1_features = {
        'p1_cos_dissim_mean': mean_cd,
        'p1_cos_dissim_std': std_cd,
        'p1_cos_dissim_cv': cv_cd,
        'p1_estimated_period': periods,
        'p1_n_zero_crossings': nzc,
    }

    # ─────────────────────────────────────────────────────────────
    # 4e — FEATURES MASQUE
    # ─────────────────────────────────────────────────────────────

    n_active = jnp.sum(mask.astype(jnp.int32), axis=1)
    coverage = n_active.astype(jnp.float32) / max_it

    diff_mask = jnp.diff(mask.astype(jnp.int32), axis=1)
    n_starts = jnp.sum(diff_mask == 1, axis=1) + mask[:, 0].astype(jnp.int32)
    n_transitions = n_starts.astype(jnp.float32)

    indices_for_last = jnp.arange(T)[None, :] * mask.astype(jnp.int32)
    t_last = jnp.max(indices_for_last, axis=1).astype(jnp.float32)
    indices_for_first = jnp.where(mask, jnp.arange(T)[None, :], max_it)
    t_first = jnp.min(indices_for_first, axis=1).astype(jnp.float32)

    has_any = n_active > 0
    t_first_norm = jnp.where(has_any, t_first / max_it, jnp.nan)
    t_last_norm = jnp.where(has_any, t_last / max_it, jnp.nan)

    cd_for_mean_mask = jnp.where(mask, cos_dissim, 0.0)
    mean_amp = jnp.where(
        has_any,
        jnp.sum(cd_for_mean_mask, axis=1) / (n_active.astype(jnp.float32) + EPS),
        jnp.nan)
    cd_for_max_mask = jnp.where(mask, cos_dissim, -jnp.inf)
    max_amp = jnp.where(has_any, jnp.max(cd_for_max_mask, axis=1), jnp.nan)

    has_multiple = n_starts >= 2
    spacing = jnp.where(
        has_multiple,
        (t_last - t_first) / (n_starts.astype(jnp.float32) - 1.0) / max_it,
        jnp.nan)

    mask_features = {
        'mask_n_transitions': n_transitions,
        'mask_t_first_norm': t_first_norm,
        'mask_t_last_norm': t_last_norm,
        'mask_mean_amplitude': mean_amp,
        'mask_max_amplitude': max_amp,
        'mask_mean_spacing_norm': spacing,
        'mask_coverage_frac': coverage,
    }

    return statuses, t_effectives, regimes, periods, mask, p1_features, mask_features
