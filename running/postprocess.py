"""
Post-traitement CPU et assemblage final — numpy uniquement.

Ce module est le SEUL consommateur du transfert GPU→CPU.
À partir d'ici, plus aucune donnée JAX.

Architecture SC6 (D1, D2, D9, D10, D14) :

    Le transfert GPU→CPU se fait UNE SEULE FOIS (P11 vases communicants).
    Après le transfert, le GPU est libéré pour le groupe suivant.

    Trois catégories de features post-scan (DOC1 §5) :

        C1 — Agrégats scalaires  (69 scalaires)
             Une fonction unifiée parcourt AGG_MAP et produit les agrégats
             sur couche A ET couche B (SC1-D2 : fusion des anciens
             DYNAMIC_AGG_MAP et SPECTRAL_AGG_MAP).

        C2 — Post-process grammaire signal  (34 scalaires)
             DMD, F6, stationnarité, entropy production, autocorr, PNN,
             temporal. Alimentés EXCLUSIVEMENT par timeline_tables_A.
             Jamais sur feature_tables_B (charter §1.5, P10).

        C3 — Post-process grammaire virages  (9 scalaires)
             Phasic features. Deux sources natives couche B
             (f2_von_neumann_entropy, f1_effective_rank) et une source
             projetée A→virages (mode_asymmetry_o2, projection licite §1.7
             Option 2 DOC1 §5.4).

Convention NaN-éparse des buffers SC5-D3 :

    timeline_tables_A[k][b, t]  = NaN si t >= t_effectives[b]
    feature_tables_B[k][b, t]   = NaN si mask[b, t] == False

    Conséquence : np.nanmean / np.nanstd / np.nanpercentile donnent les
    bons résultats SANS filtre post-scan (piège P20).

Pour l'analysing — interprétation :

    Les features sont des OBSERVATIONS numériques (charter §1.8).
    Aucune n'est un verdict. NaN signifie "non applicable",
    pas une erreur.

@ROLE    Post-traitement CPU — C1 agrégats, C2 signal, C3 virages, parquet
@LAYER   running

@EXPORTS
  transfer_to_cpu(p1_out, p2_out, classify_out) → Dict[str, Any]
  aggregate_by_agg_map(tables) → Dict[str, (B,) float32]
  compute_dmd_on_observables(tables_A, t_eff) → Dict[str, (B,) float32]
  compute_f6_on_observables(tables_A, t_eff) → Dict[str, (B,) float32]
  compute_stationarity(tables_A, t_eff) → Dict[str, (B,) float32]
  compute_entropy_production(tables_A, t_eff) → (B,) float32
  compute_autocorrelations(tables_A) → Dict[str, (B,) float32]
  compute_pnn_features(tables_A) → Dict[str, (B,) float32]
  compute_temporal_features(tables_A, t_eff) → Dict[str, (B,) float32]
  compute_phasic_features(tables_B, tables_A, mask) → Dict[str, (B,) float32]
  build_col_data(buffers, c1_agg, c2_signal, c3_virages,
                 metadata, group_meta, sub_batch, is_diff, rank_eff,
                 B, max_it) → Dict[str, Any]
  apply_structural_nan(col_data, is_diff, rank_eff) → None
  apply_runtime_nan(col_data, count_active, mask_features) → None

@LIFECYCLE
  CREATES  cpu_data       dict numpy — images host des buffers GPU
  CREATES  c1_agg         dict 69 scalaires — agrégats C1
  CREATES  c2_signal      dict 34 scalaires — C2 signal
  CREATES  c3_virages     dict 9 scalaires — C3 virages
  CREATES  col_data       dict parquet — consommé par col_queue (étape 10)
  DELETES  rien — tous les dicts transfèrent la propriété au col_queue

@CONFORMITY
  OK    Charter §1.5 / P10   grammaire signal → timelines A, virages → mask
  OK    Charter §1.7         projection signal→virages licite (phasic asym)
  OK    Charter §1.8         features = observations, pas verdicts
  OK    Charter P4           zéro paramètre hardcodé (tout via registry)
  OK    Charter P6           persistance avant analyse (col_queue)
  OK    SC1-D2               fusion AGG_MAP — aggregate_by_agg_map unifiée
  OK    SC1-D9               health_has_inf / health_is_collapsed retirés
  OK    SC5-D3               pas de filtre post-scan (NaN déjà posés)
  OK    D9                   DMD sur 9 observables couche A (exclusions
                             is_finite + frob_gradient SD-C)
  OK    D10                  stationnarité par observable couche A (11)
  OK    D14                  parquet 3 niveaux (11 A continues + 15 B
                             virages + 127 scalaires)
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple

from running.features_registry import (
    EPS,
    FEATURE_NAMES,
    LAYER_A_KEYS,
    LAYER_B_KEYS,
    AGG_MAP,
    PASS1_KEYS,
    TIMELINE_COLUMNS_A,
    TIMELINE_COLUMNS_B,
    MASK_INDICES_COLUMN,
    FEATURES_STRUCTURAL_NAN,
    FEATURES_RUNTIME_NAN,
    IQR_Q_LOW,
    IQR_Q_HIGH,
    PLATEAU_THRESHOLD_MULT,
    PLATEAU_THRESHOLD_FLOOR,
    PNN_THRESHOLD,
    DMD_FORGET_FACTOR,
    DMD_PK_INIT_SCALE,
    DMD_COMPLEX_THRESHOLD,
    STATIONARITY_TAIL_FRAC,
    TURBULENCE_CONSTANT,
)
from running.classify_jax import (
    STATUS_NAMES,
    REGIME_NAMES,
)


# =========================================================================
# CONSTANTES DE MODULE — choix SC6 SD-STRUCT-2 (dans postprocess, pas registre)
# =========================================================================

# Observables injectés dans le DMD (D9 + SD-C) :
#   - is_finite exclu : binaire 0/1, aucun sens spectral
#   - frob_gradient exclu : NaN structurel rank_eff == 2, contaminerait
#     la matrice cross-rangs
# Reste 9 observables utiles pour caractériser la dynamique de Γ.
OBS_DMD_KEYS: List[str] = [
    k for k in LAYER_A_KEYS
    if k not in ('is_finite', 'frob_gradient')
]  # 9 entrées

# Paires F6 Transfer Entropy (D8 + registre L702-707) :
#   source_key → cible_key : clés de lookup dans timeline_tables_A.
#   feature_name : nom abrégé aligné sur F6_FEATURE_NAMES du registre
#                  (mode_asymmetry_o2 → mode_asym_o2 pour lisibilité).
# Ordre cohérent avec F6_FEATURE_NAMES.
F6_PAIRS: List[Tuple[str, str, str]] = [
    ('frob',           'shannon_comp',       'f6_te_frob_to_shannon_comp'),
    ('lyap_empirical', 'frob',                'f6_te_lyap_empirical_to_frob'),
    ('cos_dissim',     'mode_asymmetry_o2',   'f6_te_cos_dissim_to_mode_asym_o2'),
    ('delta_D',        'bregman_cost',        'f6_te_delta_D_to_bregman_cost'),
]


# =========================================================================
# SECTION 0 — TRANSFERT GPU → CPU
# =========================================================================

def transfer_to_cpu(p1_out, p2_out, classify_out) -> Dict[str, Any]:
    """
    Point unique de transfert GPU→CPU (P11 vases communicants).

    Après cet appel, plus aucune donnée JAX dans le flux. Le budget
    VRAM peut être libéré (fait par l'appelant juste après).

    Args:
        p1_out       : sortie run_pass1 (JAX) — 4 timelines PASS1_KEYS + last_states
        p2_out       : sortie run_pass2 (JAX) — timeline_tables_A (11),
                       feature_tables_B (15), last_states, count_active
        classify_out : dict JAX — 7 clés classify

    Returns:
        dict numpy pur, plat, consommable par les compute_* :
          'timelines_p1'       : Dict[str, (B, max_it) float32]  — 4 clés
          'timeline_tables_A'  : Dict[str, (B, max_it) float32]  — 11 clés
          'feature_tables_B'   : Dict[str, (B, max_it) float32]  — 15 clés
          'last_states'        : (B, *state_shape) float32       — état P2 final
          'statuses'           : (B,) int32
          't_effectives'       : (B,) int32
          'regimes'             : (B,) int32
          'periods'            : (B,) float32
          'mask'                : (B, max_it) bool
          'count_active'       : (B,) int32
          'p1_features'        : Dict[str, (B,) float32]         — 5 clés
          'mask_features'      : Dict[str, (B,) float32]         — 7 clés
    """
    def _to_np(x):
        return np.array(x) if hasattr(x, 'shape') else x

    result: Dict[str, Any] = {}

    # P1 — 4 timelines screening (PASS1_KEYS)
    result['timelines_p1'] = {
        k: _to_np(p1_out[k]) for k in PASS1_KEYS
    }

    # P2 — deux tables NaN-éparses + état final + compteur actif
    result['timeline_tables_A'] = {
        k: _to_np(p2_out['timeline_tables_A'][k]) for k in LAYER_A_KEYS
    }
    result['feature_tables_B'] = {
        k: _to_np(p2_out['feature_tables_B'][k]) for k in LAYER_B_KEYS
    }
    result['last_states']  = _to_np(p2_out['last_states'])
    result['count_active'] = _to_np(p2_out['count_active'])

    # Classify — 7 clés à plat
    for k in ('statuses', 't_effectives', 'regimes', 'periods', 'mask'):
        result[k] = _to_np(classify_out[k])
    result['p1_features'] = {
        kk: _to_np(vv) for kk, vv in classify_out['p1_features'].items()
    }
    result['mask_features'] = {
        kk: _to_np(vv) for kk, vv in classify_out['mask_features'].items()
    }

    return result


# =========================================================================
# SECTION 1 — UTILITAIRES INTERNES
# =========================================================================

def _first_last_nan_safe(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Première et dernière valeur finie par run.

    Exploite la convention NaN-éparse SC5-D3 : les valeurs non applicables
    sont déjà NaN, pas besoin de masque externe.

    Args:
        values : (B, T) float32

    Returns:
        first   : (B,) float32 — première valeur finie, NaN si aucune
        last    : (B,) float32 — dernière valeur finie, NaN si aucune
        has_any : (B,) bool
    """
    B, T = values.shape
    is_valid = np.isfinite(values)
    has_any = np.any(is_valid, axis=1)

    first_idx = np.argmax(is_valid, axis=1)
    first_vals = values[np.arange(B), first_idx]
    first_vals = np.where(has_any, first_vals, np.nan)

    flipped = is_valid[:, ::-1]
    last_idx_rev = np.argmax(flipped, axis=1)
    last_idx = (T - 1) - last_idx_rev
    last_vals = values[np.arange(B), last_idx]
    last_vals = np.where(has_any, last_vals, np.nan)

    return first_vals.astype(np.float32), last_vals.astype(np.float32), has_any


def _extract_at_mask(values: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """
    Extrait les valeurs aux points masqués, run par run.

    SD-A + SD-STRUCT-5 : liste Python de B arrays 1D de longueurs K_i
    variables. Aucun padding.

    Utilisé en interne par compute_phasic_features et par build_col_data
    pour les timelines parquet niveau 2.

    Args:
        values : (B, T) float32
        mask   : (B, T) bool

    Returns:
        List[np.ndarray] — B entrées, longueurs K_i variables.
        Les NaN sont conservés tels quels dans la sortie (le consommateur
        filtre selon son besoin).
    """
    B = values.shape[0]
    return [values[i, mask[i]] for i in range(B)]


# =========================================================================
# SECTION 2 — AGRÉGATS C1 (SC1-D2 — fusion dynamic + spectral)
# =========================================================================

def aggregate_by_agg_map(tables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Agrégats scalaires unifiés — une seule routine pour A et B.

    Parcourt AGG_MAP et produit {f'{key}_{agg}': (B,) float32} pour chaque
    couple (key, agg). Les NaN-éparses encodent déjà l'inapplicabilité
    (SC5-D3) — np.nanmean / np.nanstd donnent les bons résultats
    directement sans filtre externe.

    Sémantique des agrégats :
      mean  : np.nanmean sur l'axe temporel
      std   : np.nanstd  sur l'axe temporel
      final : dernière valeur finie par run
      delta : last - first (valeurs finies)
      total : np.nansum (quantités cumulables : delta_D, bregman_cost)

    Args:
        tables : Dict[str, (B, T) float32] — doit contenir toutes les clés
                 de AGG_MAP (i.e. LAYER_A_KEYS + LAYER_B_KEYS)

    Returns:
        dict de 69 scalaires (B,) float32.
    """
    col_data: Dict[str, np.ndarray] = {}

    # Cache first/last par clé pour éviter le recalcul
    first_last_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for key, aggs in AGG_MAP.items():
        values = tables[key]  # (B, T) float32

        # Suppress warnings pour les runs entièrement NaN (ex. rank==2 +
        # frob_gradient) : c'est une observation structurelle, pas un bug.
        with np.errstate(invalid='ignore'):
            mean_vals = np.nanmean(values, axis=1).astype(np.float32) \
                if any(a in aggs for a in ('mean',)) else None
            std_vals = np.nanstd(values, axis=1).astype(np.float32) \
                if 'std' in aggs else None
            total_vals = np.nansum(values, axis=1).astype(np.float32) \
                if 'total' in aggs else None

        if any(a in aggs for a in ('final', 'delta')):
            first_v, last_v, has_any = _first_last_nan_safe(values)
            first_last_cache[key] = (first_v, last_v, has_any)

        for agg in aggs:
            col_name = f'{key}_{agg}'
            if agg == 'mean':
                col_data[col_name] = mean_vals
            elif agg == 'std':
                col_data[col_name] = std_vals
            elif agg == 'total':
                col_data[col_name] = total_vals
            elif agg == 'final':
                _, last_v, _ = first_last_cache[key]
                col_data[col_name] = last_v
            elif agg == 'delta':
                first_v, last_v, has_any = first_last_cache[key]
                delta = last_v - first_v
                delta = np.where(has_any, delta, np.nan)
                col_data[col_name] = delta.astype(np.float32)

    return col_data


# =========================================================================
# SECTION 3 — POST-PROCESS C2 (GRAMMAIRE SIGNAL)
# =========================================================================
# Alimentés EXCLUSIVEMENT par timeline_tables_A (uniformes).
# Jamais sur feature_tables_B — règle grammaire charter §1.5, P10.


# ── 3.1 DMD sur 9 observables (D9 + SD-C) ─────────────────────────────────

def compute_dmd_on_observables(
    timeline_tables_A: Dict[str, np.ndarray],
    t_effectives: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    DMD RLS streaming sur les 9 observables couche A (OBS_DMD_KEYS).

    Entrée : matrice (T_eff_b, 9) par run. Méthode : RLS sur la séquence
    de vecteurs d'observables jusqu'à t_effective par run.

    Test d'universalité (charter §1.4) : si deux (enc, Γ) donnent le même
    spectre DMD sur ces observables canoniques → Γ universel.

    Args:
        timeline_tables_A : Dict[str, (B, max_it) float32]
        t_effectives      : (B,) int32

    Returns:
        4 scalaires (B,) float32 : f7_dmd_spectral_radius,
        f7_dmd_n_complex_pairs, f7_dmd_spectral_entropy,
        f7_dmd_decay_rate.

    NaN runtime : t_effective < 3 (au moins deux transitions nécessaires).
    """
    B = t_effectives.shape[0]
    D = len(OBS_DMD_KEYS)  # 9

    # Construire la matrice (B, max_it, 9) des 9 observables empilés
    # Les NaN SC5-D3 restent tels quels : on les gère dans la boucle RLS
    stacked = np.stack(
        [timeline_tables_A[k] for k in OBS_DMD_KEYS],
        axis=-1,
    )  # (B, max_it, 9)

    T = stacked.shape[1]

    # RLS streaming — matrice d'évolution D×D (pas n_dof×n_dof)
    forget = DMD_FORGET_FACTOR
    A_k = np.zeros((B, D, D), dtype=np.float64)
    P_k = np.tile(
        np.eye(D, dtype=np.float64) * DMD_PK_INIT_SCALE,
        (B, 1, 1),
    )

    # On itère de 1 à T-1 ; pour chaque run, les updates s'arrêtent
    # à t_effectives[b] - 1 (après ça, on écrase par 0 pour ne pas
    # corrompre A_k, P_k).
    T_max_eff = int(np.max(t_effectives)) if B > 0 else 0

    for step in range(1, T_max_eff):
        update_mask = step < t_effectives  # (B,) bool

        x = stacked[:, step - 1, :].astype(np.float64)
        y = stacked[:, step, :].astype(np.float64)

        # NaN (structurels ou post-t_eff) → 0 pour stabilité numérique
        x = np.where(np.isfinite(x), x, 0.0)
        y = np.where(np.isfinite(y), y, 0.0)

        Px = np.einsum('bij,bj->bi', P_k, x)
        denom = forget + np.einsum('bi,bi->b', x, Px)
        denom = denom[:, None] + EPS
        k_vec = Px / denom
        e = y - np.einsum('bij,bj->bi', A_k, x)

        mask_3d = update_mask[:, None, None]

        delta_A = np.einsum('bi,bj->bij', e, k_vec)
        A_k = A_k + np.where(mask_3d, delta_A, 0.0)

        delta_P = np.einsum('bi,bj->bij', Px, Px) / denom[:, :, None]
        P_k = np.where(mask_3d, (P_k - delta_P) / forget, P_k)

    # Eigenvalues
    A_clean = np.where(np.isfinite(A_k), A_k, 0.0)
    eigenvalues = np.linalg.eigvals(A_clean)  # (B, D) complex
    abs_eigs = np.abs(eigenvalues)

    spectral_radius = np.max(abs_eigs, axis=1).astype(np.float32)
    n_complex = np.sum(
        np.abs(np.imag(eigenvalues)) > DMD_COMPLEX_THRESHOLD,
        axis=1,
    ).astype(np.float32)

    p_dmd = abs_eigs / (np.sum(abs_eigs, axis=1, keepdims=True) + EPS)
    spectral_entropy = -np.sum(
        p_dmd * np.log(p_dmd + EPS),
        axis=1,
    ).astype(np.float32)

    idx_dom = np.argmax(abs_eigs, axis=1)
    dom_eigs = eigenvalues[np.arange(B), idx_dom]
    decay_rate = np.real(np.log(np.abs(dom_eigs) + EPS)).astype(np.float32)

    # NaN runtime : t_effective < 3 → pas assez de données pour RLS
    insufficient = t_effectives < 3
    spectral_radius[insufficient] = np.nan
    spectral_entropy[insufficient] = np.nan
    decay_rate[insufficient] = np.nan
    # n_complex reste à 0 (sens défini : aucun mode détecté)

    return {
        'f7_dmd_spectral_radius':   spectral_radius,
        'f7_dmd_n_complex_pairs':    n_complex,
        'f7_dmd_spectral_entropy':   spectral_entropy,
        'f7_dmd_decay_rate':         decay_rate,
    }


# ── 3.2 F6 Transfer Entropy sur 4 paires (D8 + registre L702-707) ──────────

def _corr_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Corrélation Pearson vectorisée, NaN-safe. x, y: (B, T) → (B,)."""
    xc = x - np.nanmean(x, axis=1, keepdims=True)
    yc = y - np.nanmean(y, axis=1, keepdims=True)
    num = np.nansum(xc * yc, axis=1)
    den = np.sqrt(np.nansum(xc ** 2, axis=1) * np.nansum(yc ** 2, axis=1)) + EPS
    return num / den


def _approx_transfer_entropy_batch(sig_A: np.ndarray, sig_B: np.ndarray) -> np.ndarray:
    """
    TE approximée : différence des corrélations asymétriques entre lag(A)
    et lead(B) vs lag(B) et lead(A). (B, T) → (B,) float.
    """
    A_lag = sig_A[:, :-1]
    B_next = sig_B[:, 1:]
    B_lag = sig_B[:, :-1]
    A_next = sig_A[:, 1:]
    return np.abs(_corr_batch(A_lag, B_next)) - np.abs(_corr_batch(B_lag, A_next))


def compute_f6_on_observables(
    timeline_tables_A: Dict[str, np.ndarray],
    t_effectives: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Transfer Entropy sur 4 paires d'observables couche A (F6_PAIRS).

    Grammaire signal stricte : timelines A continues NaN-éparses, TE calculée
    directement. Les NaN SC5-D3 pour t >= t_eff sont gérés par _corr_batch
    via nanmean/nansum.

    Args:
        timeline_tables_A : Dict[str, (B, max_it) float32]
        t_effectives       : (B,) int32

    Returns:
        5 scalaires (B,) float32 : 4 f6_te_* + f6_causal_asymmetry_index.

    NaN runtime : t_effective < 3.
    """
    B = t_effectives.shape[0]
    out: Dict[str, np.ndarray] = {}

    te_values: List[np.ndarray] = []
    for src, tgt, feat_name in F6_PAIRS:
        sig_A = timeline_tables_A[src]
        sig_B = timeline_tables_A[tgt]
        te = _approx_transfer_entropy_batch(sig_A, sig_B)
        te_values.append(te)
        out[feat_name] = te.astype(np.float32)

    # Agrégat global — moyenne des valeurs absolues
    cai = np.mean(np.abs(np.stack(te_values, axis=1)), axis=1)
    out['f6_causal_asymmetry_index'] = cai.astype(np.float32)

    # NaN runtime — pas assez de données pour TE
    insufficient = t_effectives < 3
    for key in out:
        out[key][insufficient] = np.nan

    return out


# ── 3.3 Stationnarité — 11 scalaires (D10, DOC1 §5.3 C2.6) ────────────────

def compute_stationarity(
    timeline_tables_A: Dict[str, np.ndarray],
    t_effectives: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Stationnarité par observable couche A :
        stat_delta_{k} = |mean(last frac%) - mean(first frac%)|
                         / (std_all + EPS)

    Bornes de tranches calculées PAR RUN depuis t_effectives, pas depuis
    max_it — sinon biais systémique pour OK_TRUNCATED.

    Args:
        timeline_tables_A : Dict[str, (B, max_it) float32]
        t_effectives       : (B,) int32

    Returns:
        11 scalaires (B,) float32 : stat_delta_{k} pour k ∈ LAYER_A_KEYS.

    NaN runtime : t_effective < 10 (au moins 2 points par tranche de 20%).
    """
    B = t_effectives.shape[0]
    frac = STATIONARITY_TAIL_FRAC
    out: Dict[str, np.ndarray] = {}

    # Calcul des bornes par run
    tail_len = np.maximum((t_effectives.astype(np.float32) * frac).astype(np.int32), 1)  # (B,)
    # first slice : [0, tail_len)
    # last slice  : [t_eff - tail_len, t_eff)
    insufficient = t_effectives < 10

    for k in LAYER_A_KEYS:
        values = timeline_tables_A[k]  # (B, max_it)
        stat_delta = np.full(B, np.nan, dtype=np.float32)

        # Loop par run — inévitable puisque chaque run a ses propres bornes
        with np.errstate(invalid='ignore'):
            for b in range(B):
                if insufficient[b]:
                    continue
                t_eff = int(t_effectives[b])
                tl = int(tail_len[b])
                first_slice = values[b, :tl]
                last_slice = values[b, t_eff - tl:t_eff]
                all_slice = values[b, :t_eff]

                first_mean = np.nanmean(first_slice)
                last_mean = np.nanmean(last_slice)
                all_std = np.nanstd(all_slice)

                if np.isfinite(first_mean) and np.isfinite(last_mean) and np.isfinite(all_std):
                    stat_delta[b] = np.abs(last_mean - first_mean) / (all_std + EPS)

        out[f'stat_delta_{k}'] = stat_delta

    return out


# ── 3.4 Entropy production rate — 1 scalaire ──────────────────────────────

def compute_entropy_production(
    timeline_tables_A: Dict[str, np.ndarray],
    t_effectives: np.ndarray,
) -> np.ndarray:
    """
    Taux de production d'entropie — pente de régression linéaire sur
    shannon_comp (timeline couche A, grammaire signal).

    Ancrage : A3 — production d'entropie irréductible.

    ATTENTION : nouvelle sémantique par rapport à l'ancien pipeline.
    L'ancien opérait sur f2_von_neumann_entropy (spectral) aux virages.
    Le nouveau opère sur shannon_comp (proxy O(n²)) en timeline complète.
    Ce n'est PAS la même quantité physique — c'est un proxy.

    Args:
        timeline_tables_A : doit contenir 'shannon_comp'
        t_effectives       : (B,) int32

    Returns:
        (B,) float32 — f2_entropy_production_rate.

    NaN runtime : t_effective < 3 (pente indéfinie).
    """
    B = t_effectives.shape[0]
    signal = timeline_tables_A['shannon_comp']  # (B, max_it)
    result = np.full(B, np.nan, dtype=np.float32)

    for b in range(B):
        t_eff = int(t_effectives[b])
        if t_eff < 3:
            continue
        y = signal[b, :t_eff]
        valid = np.isfinite(y)
        n_valid = int(np.sum(valid))
        if n_valid < 3:
            continue
        x = np.arange(t_eff, dtype=np.float32)[valid]
        y_clean = y[valid]
        # Régression linéaire simple via moments
        x_mean = np.mean(x)
        y_mean = np.mean(y_clean)
        xy = np.mean((x - x_mean) * (y_clean - y_mean))
        xx = np.mean((x - x_mean) ** 2)
        if xx > EPS:
            result[b] = float(xy / xx)

    return result


# ── 3.5 Autocorrélations — 3 scalaires ────────────────────────────────────

def _compute_first_min_autocorr_one(signal: np.ndarray) -> np.ndarray:
    """
    Premier minimum de l'autocorrélation normalisée, NaN-safe.

    Args:
        signal : (B, T) float32

    Returns:
        (B,) float32 — position du 1er minimum ou NaN si signal invalide.
    """
    B, T = signal.shape
    result = np.full(B, np.nan, dtype=np.float32)

    for i in range(B):
        sig = signal[i, :]
        valid = np.isfinite(sig)
        n_valid = int(np.sum(valid))
        if n_valid < 4:
            continue

        sig_clean = sig[valid]
        centered = sig_clean - np.mean(sig_clean)
        energy = np.sum(centered ** 2)
        if energy < EPS:
            continue

        max_lag = n_valid // 2
        if max_lag < 3:
            continue

        ac = np.zeros(max_lag)
        for lag in range(1, max_lag):
            ac[lag] = np.sum(centered[:n_valid - lag] * centered[lag:]) / energy

        # Premier négatif
        neg_mask = ac[1:] < 0
        first_neg = int(np.argmax(neg_mask) + 1) if bool(np.any(neg_mask)) else max_lag

        # Premier upturn
        dac = np.diff(ac[1:])
        up_mask = dac > 0
        first_up = int(np.argmax(up_mask) + 1) if bool(np.any(up_mask)) else max_lag

        result[i] = float(min(first_neg, first_up))

    return result


def compute_autocorrelations(
    timeline_tables_A: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Premier minimum d'autocorrélation sur 3 observables couche A.

    Sources : frob, shannon_comp, mode_asymmetry_o2 (registre L713-717).

    Args:
        timeline_tables_A : Dict[str, (B, max_it) float32]

    Returns:
        3 scalaires (B,) float32 : ps_first_min_ac_{frob, shannon_comp,
        mode_asym_o2}.
    """
    return {
        'ps_first_min_ac_frob':           _compute_first_min_autocorr_one(
            timeline_tables_A['frob']),
        'ps_first_min_ac_shannon_comp':   _compute_first_min_autocorr_one(
            timeline_tables_A['shannon_comp']),
        'ps_first_min_ac_mode_asym_o2':   _compute_first_min_autocorr_one(
            timeline_tables_A['mode_asymmetry_o2']),
    }


# ── 3.6 PNN40 — 2 scalaires ───────────────────────────────────────────────

def _compute_pnn_one(signal: np.ndarray) -> np.ndarray:
    """
    Fraction des différences successives dépassant PNN_THRESHOLD × std.

    Args:
        signal : (B, T) float32

    Returns:
        (B,) float32.
    """
    B, T = signal.shape
    if T < 2:
        return np.full(B, np.nan, dtype=np.float32)
    d = np.abs(np.diff(signal, axis=1))
    s = np.nanstd(signal, axis=1, keepdims=True)
    return np.nanmean(d > PNN_THRESHOLD * s, axis=1).astype(np.float32)


def compute_pnn_features(
    timeline_tables_A: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    PNN40 sur 2 observables couche A.

    Sources : shannon_comp, mode_asymmetry_o2 (registre L722-725).

    Returns:
        2 scalaires (B,) float32 : ps_pnn40_{shannon_comp, mode_asym_o2}.
    """
    return {
        'ps_pnn40_shannon_comp':  _compute_pnn_one(timeline_tables_A['shannon_comp']),
        'ps_pnn40_mode_asym_o2':  _compute_pnn_one(timeline_tables_A['mode_asymmetry_o2']),
    }


# ── 3.7 Temporal features — 8 scalaires ───────────────────────────────────

def _compute_temporal_one(
    signal: np.ndarray,
    t_effectives: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    4 features temporelles sur un signal couche A.

    V3 : changepoint_t_norm normalisé par t_effectives (pas max_it) —
    corrige le biais OK_TRUNCATED de l'ancien pipeline.

    Args:
        signal       : (B, T) float32, NaN-safe
        t_effectives : (B,) int32 — utilisé uniquement pour cp_norm

    Returns:
        (iqr, plateau_frac, cusum_delta, changepoint_t_norm) chaque (B,).
    """
    signal = signal.astype(np.float32, copy=False)
    B, T = signal.shape
    if T < 2:
        nan_b = np.full(B, np.nan, dtype=np.float32)
        return (nan_b.copy(), nan_b.copy(), nan_b.copy(), nan_b.copy())

    # IQR
    with np.errstate(invalid='ignore'):
        iqr = (np.nanpercentile(signal, IQR_Q_HIGH * 100, axis=1) -
               np.nanpercentile(signal, IQR_Q_LOW * 100, axis=1))

    # Plateau fraction
    deltas = np.abs(np.diff(signal, axis=1))
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(deltas, axis=1, keepdims=True)
    thr = np.maximum(med * PLATEAU_THRESHOLD_MULT, PLATEAU_THRESHOLD_FLOOR)
    is_plateau = np.isfinite(deltas) & (deltas < thr)
    n_finite = np.sum(np.isfinite(deltas), axis=1)
    pfrac = np.sum(is_plateau, axis=1) / np.maximum(n_finite, 1)

    # CUSUM delta — NaN-safe via cumsum
    is_fin = np.isfinite(signal)
    cumcount = np.cumsum(is_fin.astype(np.float32), axis=1)
    safe = np.where(is_fin, signal, 0.0)
    cumval = np.cumsum(safe, axis=1)

    count_total = cumcount[:, -1]
    half_target = (count_total // 2).astype(np.int32)

    reached = cumcount >= half_target[:, None]
    T_half_idx = np.argmax(reached, axis=1)

    sum_first = cumval[np.arange(B), T_half_idx]
    count_first = cumcount[np.arange(B), T_half_idx]
    mean_first = np.where(count_first > 0, sum_first / np.maximum(count_first, 1), 0.0)

    sum_total = cumval[:, -1]
    sum_second = sum_total - sum_first
    count_second = count_total - count_first
    mean_second = np.where(count_second > 0, sum_second / np.maximum(count_second, 1), 0.0)

    cusum_delta = mean_second - mean_first

    # Changepoint — V3 : normaliser par t_effectives, pas T
    deltas_for_cp = np.abs(np.diff(signal, axis=1))
    deltas_for_cp = np.where(np.isfinite(deltas_for_cp), deltas_for_cp, 0.0)
    cp_idx = np.argmax(deltas_for_cp, axis=1).astype(np.float32)
    t_eff_safe = np.maximum(t_effectives.astype(np.float32), 1.0)
    cp_norm = cp_idx / t_eff_safe
    cp_norm = np.where(count_total < 3, np.nan, cp_norm)

    return (iqr.astype(np.float32),
            pfrac.astype(np.float32),
            cusum_delta.astype(np.float32),
            cp_norm.astype(np.float32))


def compute_temporal_features(
    timeline_tables_A: Dict[str, np.ndarray],
    t_effectives: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Features temporelles sur 2 observables couche A.

    Sources : frob, shannon_comp (registre L731-740).

    Args:
        timeline_tables_A : Dict[str, (B, max_it) float32]
        t_effectives      : (B,) int32

    Returns:
        8 scalaires (B,) float32 — temporal_{frob,shannon_comp}_{iqr,
        plateau_frac, cusum_delta, changepoint_t_norm}.
    """
    frob_iqr, frob_pfrac, frob_cusum, frob_cp = _compute_temporal_one(
        timeline_tables_A['frob'], t_effectives)
    shan_iqr, shan_pfrac, shan_cusum, shan_cp = _compute_temporal_one(
        timeline_tables_A['shannon_comp'], t_effectives)

    return {
        'temporal_frob_iqr':                         frob_iqr,
        'temporal_frob_plateau_frac':                frob_pfrac,
        'temporal_frob_cusum_delta':                 frob_cusum,
        'temporal_frob_changepoint_t_norm':          frob_cp,
        'temporal_shannon_comp_iqr':                 shan_iqr,
        'temporal_shannon_comp_plateau_frac':        shan_pfrac,
        'temporal_shannon_comp_cusum_delta':         shan_cusum,
        'temporal_shannon_comp_changepoint_t_norm':  shan_cp,
    }


# =========================================================================
# SECTION 4 — POST-PROCESS C3 (GRAMMAIRE VIRAGES)
# =========================================================================

def _compute_phasic_one_signal(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    3 features phasic sur une séquence ordonnée 1D de valeurs aux virages.

    Ne suppose PAS d'uniformité temporelle (§1.5, grammaire virages).
    Mesures ordinales uniquement.

    Args:
        sig    : 1D array de longueur K (variable), valeurs finies uniquement
        prefix : e.g. 'phasic_svn', 'phasic_rank', 'phasic_mode_asym_o2'

    Returns:
        dict de 3 entrées {prefix}_{n_reversals, max_monotone_frac, range_ratio}.
    """
    K = len(sig)
    nan3 = {
        f'{prefix}_n_reversals':        np.nan,
        f'{prefix}_max_monotone_frac':  np.nan,
        f'{prefix}_range_ratio':        np.nan,
    }
    if K < 2:
        return nan3

    d = np.diff(sig)
    signs = np.sign(d)
    signs[np.abs(d) < EPS] = 0

    nr = float(np.sum(np.abs(np.diff(signs)) > 0))

    if K < 3:
        mm = 1.0
    else:
        mr, cr = 1, 1
        for j in range(1, len(signs)):
            if signs[j] == signs[j - 1] and signs[j] != 0:
                cr += 1
                mr = max(mr, cr)
            else:
                cr = 1
        mm = float(mr + 1) / K

    rng = float(np.nanmax(sig) - np.nanmin(sig))
    mabs = float(np.nanmean(np.abs(sig)))

    return {
        f'{prefix}_n_reversals':        nr,
        f'{prefix}_max_monotone_frac':  mm,
        f'{prefix}_range_ratio':        rng / (mabs + EPS),
    }


def compute_phasic_features(
    feature_tables_B: Dict[str, np.ndarray],
    timeline_tables_A: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Phasic features — 9 scalaires (3 sources × 3 mesures).

    Trois sources :
      - svn            ← feature_tables_B['f2_von_neumann_entropy']  (natif couche B)
      - rank           ← feature_tables_B['f1_effective_rank']         (natif couche B)
      - mode_asym_o2   ← timeline_tables_A['mode_asymmetry_o2'][mask]  (projection
                         signal → virages, §1.7 Option 2 licite, SC1-D6)

    Pour svn et rank : les valeurs aux virages sont déjà encodées par
    les NaN-éparses de feature_tables_B. L'extraction via mask filtre
    directement les valeurs finies.

    Pour mode_asym_o2 : sous-échantillonnage explicite au masque du signal
    couche A continue.

    Args:
        feature_tables_B  : Dict[str, (B, max_it) float32]
        timeline_tables_A : Dict[str, (B, max_it) float32]
        mask              : (B, max_it) bool

    Returns:
        9 scalaires (B,) float32 — PHASIC_FEATURE_NAMES.

    NaN runtime : K_i < 2 → 3 features de la source concernée NaN.
    """
    B = mask.shape[0]

    # Pré-allocation NaN
    keys_out = []
    for pfx in ('phasic_svn', 'phasic_rank', 'phasic_mode_asym_o2'):
        for suf in ('n_reversals', 'max_monotone_frac', 'range_ratio'):
            keys_out.append(f'{pfx}_{suf}')
    col_data = {k: np.full(B, np.nan, dtype=np.float32) for k in keys_out}

    # Extraction run-par-run aux virages
    # Pour svn et rank : filtre isfinite en plus du mask (sécurité)
    svn_list = _extract_at_mask(feature_tables_B['f2_von_neumann_entropy'], mask)
    rank_list = _extract_at_mask(feature_tables_B['f1_effective_rank'], mask)
    asym_list = _extract_at_mask(timeline_tables_A['mode_asymmetry_o2'], mask)

    for i in range(B):
        svn_i = svn_list[i][np.isfinite(svn_list[i])]
        rank_i = rank_list[i][np.isfinite(rank_list[i])]
        asym_i = asym_list[i][np.isfinite(asym_list[i])]

        for sig, pfx in (
            (svn_i, 'phasic_svn'),
            (rank_i, 'phasic_rank'),
            (asym_i, 'phasic_mode_asym_o2'),
        ):
            phasic = _compute_phasic_one_signal(sig, pfx)
            for k, v in phasic.items():
                col_data[k][i] = np.float32(v)

    return col_data


# =========================================================================
# SECTION 5 — ASSEMBLAGE PARQUET (D14 — 3 niveaux)
# =========================================================================

def build_col_data(
    buffers: Dict[str, Any],
    c1_agg: Dict[str, np.ndarray],
    c2_signal: Dict[str, np.ndarray],
    c3_virages: Dict[str, np.ndarray],
    metadata: Dict[str, np.ndarray],
    group_meta: Dict[str, Any],
    sub_batch: List[Dict[str, Any]],
    is_diff: bool,
    rank_eff: int,
    B: int,
    max_it: int,
) -> Dict[str, Any]:
    """
    Assemble le dict col_data prêt pour write_col_data_to_parquet.

    Parquet 3 niveaux (D14) :
      Niveau 1 — 11 timelines couche A continues
                 (longueur t_effective, uniforme)
      Niveau 2 — 15 timelines couche B aux virages
                 + 1 colonne mask_t_indices partagée par run
                 (longueur K_i, variable)
      Niveau 3 — 127 scalaires (69 C1 + 34 C2 + 9 C3 + 15 metadata)
                 + 14 colonnes METADATA_COLUMNS

    Args:
        buffers    : sortie de transfer_to_cpu
        c1_agg     : 69 scalaires couche A+B
        c2_signal  : 34 scalaires C2
        c3_virages : 9 scalaires C3
        metadata   : 15 scalaires (p1_features 5 + mask_features 7 + meta 3)
        group_meta : dict group-level (phase, n_dof, rank_eff, max_it)
        sub_batch  : list[dict] de longueur B — seeds, ids, params
        is_diff    : bool — is_differentiable
        rank_eff   : int
        B, max_it  : int

    Returns:
        col_data : Dict complet prêt pour parquet.
    """
    col_data: Dict[str, Any] = {}

    # ── Phase 1 : pré-allocation scalaires NaN ──
    for name in FEATURE_NAMES:
        col_data[name] = np.full(B, np.nan, dtype=np.float32)

    # ── Phase 2 : metadata parquet (METADATA_COLUMNS, 14 entrées) ──
    col_data['phase']           = np.full(B, group_meta['phase'],           dtype=object)
    col_data['gamma_id']        = np.full(B, sub_batch[0]['gamma_id'],      dtype=object)
    col_data['encoding_id']     = np.array(
        [d['encoding_id']        for d in sub_batch], dtype=object)
    col_data['modifier_id']     = np.array(
        [d['modifier_id']        for d in sub_batch], dtype=object)
    col_data['gamma_params']    = np.array(
        [json.dumps(d.get('gamma_params',    {})) for d in sub_batch], dtype=object)
    col_data['encoding_params'] = np.array(
        [json.dumps(d.get('encoding_params', {})) for d in sub_batch], dtype=object)
    col_data['modifier_params'] = np.array(
        [json.dumps(d.get('modifier_params', {})) for d in sub_batch], dtype=object)
    col_data['n_dof']           = np.full(B, group_meta['n_dof'],    dtype=np.int32)
    col_data['rank_eff']        = np.full(B, group_meta['rank_eff'], dtype=np.int32)
    col_data['max_it']          = np.full(B, group_meta['max_it'],   dtype=np.int32)
    col_data['seed_CI']         = np.array([d['seed_CI']  for d in sub_batch], dtype=np.int64)
    col_data['seed_run']        = np.array([d['seed_run'] for d in sub_batch], dtype=np.int64)
    col_data['run_status']      = np.array(
        [STATUS_NAMES[int(s)] for s in buffers['statuses']], dtype=object)
    col_data['p1_regime_class'] = np.array(
        [REGIME_NAMES[int(r)] for r in buffers['regimes']], dtype=object)

    # ── Phase 3 : timelines 3 niveaux ──
    t_effectives = buffers['t_effectives']
    mask = buffers['mask']
    tables_A = buffers['timeline_tables_A']
    tables_B = buffers['feature_tables_B']

    # Niveau 1 : 11 timelines couche A continues, longueur t_effective par run
    for k, col_name in zip(LAYER_A_KEYS, TIMELINE_COLUMNS_A):
        arr = tables_A[k]  # (B, max_it)
        col_data[col_name] = [
            arr[i, :int(t_effectives[i])].tolist()
            for i in range(B)
        ]

    # Niveau 2 : 15 timelines couche B aux virages + mask_t_indices
    for k, col_name in zip(LAYER_B_KEYS, TIMELINE_COLUMNS_B):
        arr = tables_B[k]  # (B, max_it)
        col_data[col_name] = [
            arr[i, mask[i]].tolist()
            for i in range(B)
        ]

    # mask_t_indices : indices temporels du masque, 1 liste par run,
    # partagée entre les 15 colonnes B
    col_data[MASK_INDICES_COLUMN] = [
        np.where(mask[i])[0].astype(np.int32).tolist()
        for i in range(B)
    ]

    # ── Phase 4 : scalaires — remplissage des 3 buckets + metadata ──
    for k, v in c1_agg.items():
        col_data[k] = v.astype(np.float32)
    for k, v in c2_signal.items():
        col_data[k] = v.astype(np.float32)
    for k, v in c3_virages.items():
        col_data[k] = v.astype(np.float32)
    for k, v in metadata.items():
        col_data[k] = v.astype(np.float32)

    # ── Phase 5 : NaN rules ──
    apply_structural_nan(col_data, is_diff, rank_eff)
    apply_runtime_nan(col_data, buffers['count_active'], buffers['mask_features'])

    return col_data


def apply_structural_nan(col_data: Dict[str, Any], is_diff: bool, rank_eff: int) -> None:
    """
    NaN structurels — connus à la compilation depuis is_diff et rank_eff.
    Écrasent toute valeur calculée (charter P7).
    """
    for col, reason in FEATURES_STRUCTURAL_NAN.items():
        if col not in col_data:
            continue
        # Pour les timelines (list), on produit une liste de NaN par run
        val = col_data[col]
        is_list = isinstance(val, list)
        should_nan = False
        if 'differentiable' in reason and not is_diff:
            should_nan = True
        elif 'rank_eff == 2' in reason and rank_eff == 2:
            should_nan = True
        if should_nan:
            if is_list:
                col_data[col] = [[np.nan] * len(l) for l in val]
            else:
                col_data[col] = np.full_like(val, np.nan, dtype=np.float32)


def apply_runtime_nan(
    col_data: Dict[str, Any],
    count_active: np.ndarray,
    mask_features: Dict[str, np.ndarray],
) -> None:
    """
    NaN runtime — dépendants des données du run.

    Ces NaN SONT des observations scientifiques (charter P7 + §1.8) :
    "cette mesure n'est pas applicable pour ce run" — signal, pas erreur.
    """
    n_trans = mask_features.get(
        'mask_n_transitions',
        np.zeros_like(count_active, dtype=np.float32),
    )
    for col, reason in FEATURES_RUNTIME_NAN.items():
        if col not in col_data:
            continue
        if 'n_transitions == 0' in reason:
            col_data[col][n_trans == 0] = np.nan
        elif 'n_transitions < 2' in reason:
            col_data[col][n_trans < 2] = np.nan
        elif 'K_i < 2' in reason or 'K < 2' in reason:
            col_data[col][count_active < 2] = np.nan
        elif 'non_periodic' in reason:
            period_col = col_data.get('p1_estimated_period')
            if period_col is not None:
                col_data[col][np.isnan(period_col)] = np.nan
