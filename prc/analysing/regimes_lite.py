"""
prc.analysing.regimes_lite

Responsabilité : Classification régimes basée features scalaires

RECÂBLAGE timeline : Les features norm_ratio, entropy_delta, effective_rank_delta,
    log_condition_delta, spread_ratio, norm_final, condition_number_svd_final,
    has_nan_inf, is_collapsed sont maintenant calculées dans timeline_lite.extract()
    et présentes dans le dict features en entrée.
    classify_regime() ne recalcule plus rien depuis _initial/_final.

VERSION ENRICHIE : Régimes CROISSANCE_FAIBLE/FORTE conservés
FIX : refine_conserves_norm_with_cv — annotation pure, pas reclassement
"""

import numpy as np
from pathlib import Path
from typing import Dict, List

from utils.data_loading_lite import load_yaml


# =============================================================================
# CHARGEMENT THRESHOLDS
# =============================================================================

def load_regime_thresholds(profile: str = 'default') -> Dict:
    """Charge seuils régimes depuis YAML local."""
    module_dir = Path(__file__).parent
    yaml_path = module_dir / f'configs/regimes/{profile}.yaml'

    if not yaml_path.exists():
        yaml_path = module_dir / 'configs/regimes/default.yaml'

    if not yaml_path.exists():
        return {
            'CONSERVES_NORM'     : {'ratio_threshold': 1.3, 'cv_threshold': 0.10},
            'CROISSANCE_FAIBLE'  : {'ratio_min': 1.3, 'ratio_max': 3.0},
            'CROISSANCE_FORTE'   : {'ratio_min': 3.0, 'ratio_max': 10.0},
            'NUMERIC_INSTABILITY': {'condition_threshold': 1e6, 'norm_threshold': 1e10},
            'EFFONDREMENT'       : {'ratio_threshold': 0.1},
            'SATURATION'         : {'ratio_threshold': 10, 'condition_threshold': 1e3},
            'ASYMMETRY_BREAKING' : {'asymmetry_ratio_threshold': 0.5},
            'TRIVIAL'            : {'cv_threshold': 0.01},
        }

    return load_yaml(yaml_path)


# =============================================================================
# CLASSIFICATION RÉGIME INDIVIDUEL
# =============================================================================

def classify_regime(features: Dict, thresholds: Dict) -> str:
    """
    Classifie régime depuis features scalaires.

    Source features (calculées dans timeline_lite.extract) :
        has_nan_inf              : bool — NaN/Inf détectés dans history
        is_collapsed             : bool — std(history[-1]) < ε
        norm_ratio               : float — norm_final / norm_initial
        norm_final               : float — valeur absolue norme finale
        condition_number_svd_final : float — conditionnement final
        entropy_delta            : float — entropy_final - entropy_initial
    """
    # Récupération features depuis dict
    has_nan       = features.get('has_nan_inf', False)
    norm_ratio    = features.get('norm_ratio', 1.0)
    norm_final    = features.get('norm_final', 1.0)
    condition_num = features.get('condition_number_svd_final')
    is_collapsed  = features.get('is_collapsed', False)

    # Guard norm_final None / NaN
    if norm_final is None or not np.isfinite(norm_final):
        norm_final = np.nan
    if norm_ratio is None or not np.isfinite(norm_ratio):
        norm_ratio = np.nan

    # 1. PATHOLOGIES (prioritaires)
    if has_nan or (not np.isnan(norm_final) and np.isinf(norm_final)):
        return "NUMERIC_INSTABILITY"

    if condition_num is not None and np.isfinite(condition_num):
        th_cond = thresholds.get('NUMERIC_INSTABILITY', {}).get('condition_threshold', 1e6)
        if condition_num > th_cond:
            return "NUMERIC_INSTABILITY"

    th_norm_instab = thresholds.get('NUMERIC_INSTABILITY', {}).get('norm_threshold', 1e10)
    if not np.isnan(norm_final) and norm_final > th_norm_instab:
        return "NUMERIC_INSTABILITY"

    if np.isnan(norm_ratio):
        return "UNCATEGORIZED"

    # EFFONDREMENT
    th_effondrement = thresholds.get('EFFONDREMENT', {}).get('ratio_threshold', 0.1)
    if norm_ratio < th_effondrement:
        return "EFFONDREMENT"

    # 2. CONSERVATION
    th_conserve = thresholds.get('CONSERVES_NORM', {}).get('ratio_threshold', 1.3)
    if norm_ratio < th_conserve:
        return "CONSERVES_NORM"

    # 3. CROISSANCE FAIBLE
    th_cf = thresholds.get('CROISSANCE_FAIBLE', {})
    if th_cf:
        if th_cf.get('ratio_min', 1.3) <= norm_ratio < th_cf.get('ratio_max', 3.0):
            return "CROISSANCE_FAIBLE"

    # 4. CROISSANCE FORTE
    th_cfo = thresholds.get('CROISSANCE_FORTE', {})
    if th_cfo:
        if th_cfo.get('ratio_min', 3.0) <= norm_ratio < th_cfo.get('ratio_max', 10.0):
            return "CROISSANCE_FORTE"

    # 5. SATURATION (norm_ratio >= 10)
    th_sat_ratio = thresholds.get('SATURATION', {}).get('ratio_threshold', 10)
    th_sat_cond  = thresholds.get('SATURATION', {}).get('condition_threshold', 1e3)

    if norm_ratio >= th_sat_ratio:
        if condition_num is None or not np.isfinite(condition_num) or condition_num < th_sat_cond:
            return "SATURATION"

    return "UNCATEGORIZED"


# =============================================================================
# CLASSIFICATION BATCH + RÉCURRENCE
# =============================================================================

def classify_regimes_batch(
    rows: List[Dict],
    indices: List[int],
    thresholds: Dict
) -> Dict:
    """Classifie régimes batch + récurrence atomics par régime."""
    subset_rows = [rows[i] for i in indices]
    n_runs = len(subset_rows)

    regime_assignments = {}
    for i, row in zip(indices, subset_rows):
        regime = classify_regime(row['features'], thresholds)

        if regime not in regime_assignments:
            regime_assignments[regime] = []
        regime_assignments[regime].append(i)

    regimes_data = {}

    for regime, run_indices in regime_assignments.items():
        count = len(run_indices)
        fraction = count / n_runs if n_runs > 0 else 0.0

        gamma_rec    = _compute_atomic_recurrence_full(rows, run_indices, 'gamma_id')
        encoding_rec = _compute_atomic_recurrence_full(rows, run_indices, 'encoding_id')

        regimes_data[regime] = {
            'count'             : count,
            'fraction'          : fraction,
            'recurrence_gamma'  : gamma_rec,
            'recurrence_encoding': encoding_rec,
            'run_indices'       : run_indices,
        }

    regimes_data = dict(
        sorted(regimes_data.items(), key=lambda x: x[1]['count'], reverse=True)
    )

    return {
        'regimes'      : regimes_data,
        'regime_counts': {k: v['count'] for k, v in regimes_data.items()},
        'n_runs'       : n_runs,
    }


def _compute_atomic_recurrence_full(
    rows: List[Dict],
    indices: List[int],
    atomic_key: str
) -> Dict[str, Dict]:
    """Calcule récurrence atomics (format COMPLET)."""
    subset_rows = [rows[i] for i in indices]
    n_total = len(subset_rows)

    if n_total == 0:
        return {}

    counts = {}
    for row in subset_rows:
        atomic_id = row['composition'][atomic_key]
        counts[atomic_id] = counts.get(atomic_id, 0) + 1

    recurrence = {
        atomic_id: {
            'count'       : count,
            'fraction'    : count / n_total,
            'total_subset': n_total,
        }
        for atomic_id, count in counts.items()
    }

    return dict(
        sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True)[:5]
    )


# =============================================================================
# ANNOTATION CV CROSS-RUNS (pas de reclassement)
# =============================================================================

def refine_conserves_norm_with_cv(
    rows: List[Dict],
    regime_data: Dict,
    thresholds: Dict
) -> Dict:
    """
    Annote CONSERVES_NORM avec CV cross-runs.

    Annotation pure — pas de reclassement.
    Source : norm_final (fourni par timeline_lite).

    Ajoute dans regimes_data['CONSERVES_NORM'] :
        'cv_across_runs'  : float
        'high_dispersion' : bool
        'cv_threshold'    : float
    """
    if 'CONSERVES_NORM' not in regime_data['regimes']:
        return regime_data

    conserves_indices = regime_data['regimes']['CONSERVES_NORM']['run_indices']

    norm_finals = []
    for idx in conserves_indices:
        nf = rows[idx]['features'].get('norm_final')
        if nf is not None and np.isfinite(nf):
            norm_finals.append(nf)

    if len(norm_finals) < 2:
        regime_data['regimes']['CONSERVES_NORM']['cv_across_runs']  = None
        regime_data['regimes']['CONSERVES_NORM']['high_dispersion'] = False
        return regime_data

    cv = np.std(norm_finals) / (np.abs(np.mean(norm_finals)) + 1e-10)
    th_cv = thresholds.get('CONSERVES_NORM', {}).get('cv_threshold', 0.10)

    regime_data['regimes']['CONSERVES_NORM']['cv_across_runs']  = float(cv)
    regime_data['regimes']['CONSERVES_NORM']['high_dispersion'] = bool(cv > th_cv)
    regime_data['regimes']['CONSERVES_NORM']['cv_threshold']    = float(th_cv)

    return regime_data
