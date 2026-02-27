"""
prc.analysing.regimes_lite

Responsabilité : Classification régimes basée features scalaires

VERSION ENRICHIE : Régimes CROISSANCE_FAIBLE/FORTE ajoutés
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
            'CONSERVES_NORM': {'ratio_threshold': 1.3, 'cv_threshold': 0.10},
            'CROISSANCE_FAIBLE': {'ratio_min': 1.3, 'ratio_max': 3.0, 'cv_threshold': 0.20},
            'CROISSANCE_FORTE': {'ratio_min': 3.0, 'ratio_max': 10.0, 'cv_threshold': 0.30},
            'NUMERIC_INSTABILITY': {'condition_threshold': 1e6, 'norm_threshold': 1e10},
            'EFFONDREMENT': {'ratio_threshold': 0.1},
            'SATURATION': {'ratio_threshold': 10, 'condition_threshold': 1e3},
            'ASYMMETRY_BREAKING': {'asymmetry_ratio_threshold': 0.5},
            'TRIVIAL': {'cv_threshold': 0.01}
        }

    return load_yaml(yaml_path)


# =============================================================================
# CLASSIFICATION RÉGIME INDIVIDUEL
# =============================================================================

def classify_regime(features: Dict, thresholds: Dict) -> str:
    """
    Classifie régime depuis features scalaires.

    VERSION ENRICHIE : Détection CROISSANCE_FAIBLE/FORTE
    """
    has_nan = features.get('has_nan_inf', False)
    norm_initial = features.get('euclidean_norm_initial', 1.0)
    norm_final = features.get('euclidean_norm_final', 1.0)

    norm_ratio = norm_final / max(norm_initial, 1e-10)

    condition_number = features.get('condition_number_svd_final')
    asymmetry_norm = features.get('asymmetry_norm_final')

    # 1. PATHOLOGIES (prioritaires)
    if has_nan or np.isinf(norm_final):
        return "NUMERIC_INSTABILITY"

    if condition_number is not None and np.isfinite(condition_number):
        th_cond = thresholds.get('NUMERIC_INSTABILITY', {}).get('condition_threshold', 1e6)
        if condition_number > th_cond:
            return "NUMERIC_INSTABILITY"

    th_norm_instab = thresholds.get('NUMERIC_INSTABILITY', {}).get('norm_threshold', 1e10)
    if norm_final > th_norm_instab:
        return "NUMERIC_INSTABILITY"

    # EFFONDREMENT
    th_effondrement = thresholds.get('EFFONDREMENT', {}).get('ratio_threshold', 0.1)
    if norm_ratio < th_effondrement:
        return "EFFONDREMENT"

    # 2. CONSERVATION
    th_conserve_ratio = thresholds.get('CONSERVES_NORM', {}).get('ratio_threshold', 1.3)
    if norm_ratio < th_conserve_ratio:
        return "CONSERVES_NORM"

    # 3. CROISSANCE
    th_croiss_faible = thresholds.get('CROISSANCE_FAIBLE', {})
    if th_croiss_faible:
        ratio_min = th_croiss_faible.get('ratio_min', 1.3)
        ratio_max = th_croiss_faible.get('ratio_max', 3.0)
        if ratio_min <= norm_ratio < ratio_max:
            return "CROISSANCE_FAIBLE"

    th_croiss_forte = thresholds.get('CROISSANCE_FORTE', {})
    if th_croiss_forte:
        ratio_min = th_croiss_forte.get('ratio_min', 3.0)
        ratio_max = th_croiss_forte.get('ratio_max', 10.0)
        if ratio_min <= norm_ratio < ratio_max:
            return "CROISSANCE_FORTE"

    # 4. SATURATION (ratio >= 10)
    th_sat_ratio = thresholds.get('SATURATION', {}).get('ratio_threshold', 10)
    th_sat_cond = thresholds.get('SATURATION', {}).get('condition_threshold', 1e3)

    if norm_ratio >= th_sat_ratio:
        if condition_number is None or not np.isfinite(condition_number) or condition_number < th_sat_cond:
            return "SATURATION"

    # 5. ASYMMETRY_BREAKING
    if asymmetry_norm is not None and np.isfinite(asymmetry_norm) and norm_final > 1e-10:
        th_asym = thresholds.get('ASYMMETRY_BREAKING', {}).get('asymmetry_ratio_threshold', 0.5)
        asym_ratio = asymmetry_norm / norm_final
        if asym_ratio > th_asym:
            return "ASYMMETRY_BREAKING"

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

        gamma_rec = _compute_atomic_recurrence_full(rows, run_indices, 'gamma_id')
        encoding_rec = _compute_atomic_recurrence_full(rows, run_indices, 'encoding_id')

        regimes_data[regime] = {
            'count': count,
            'fraction': fraction,
            'recurrence_gamma': gamma_rec,
            'recurrence_encoding': encoding_rec,
            'run_indices': run_indices
        }

    regimes_data = dict(
        sorted(regimes_data.items(), key=lambda x: x[1]['count'], reverse=True)
    )

    return {
        'regimes': regimes_data,
        'regime_counts': {k: v['count'] for k, v in regimes_data.items()},
        'n_runs': n_runs
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

    recurrence = {}
    for atomic_id, count in counts.items():
        recurrence[atomic_id] = {
            'count': count,
            'fraction': count / n_total,
            'total_subset': n_total
        }

    recurrence = dict(
        sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True)[:5]
    )

    return recurrence


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

    FIX : Ne reclasse PLUS vers UNCATEGORIZED.
    CONSERVES_NORM avec CV élevé reste CONSERVES_NORM.
    Le CV est une information sur la dispersion inter-encodings,
    pas une raison de déclasser le régime.

    Ajoute dans regimes_data['CONSERVES_NORM'] :
        'cv_across_runs'  : float — dispersion relative des normes finales
        'high_dispersion' : bool  — True si cv > cv_threshold
        'cv_threshold'    : float — seuil utilisé (depuis YAML)

    Args:
        rows        : Liste complète rows
        regime_data : Dict depuis classify_regimes_batch()
        thresholds  : Dict seuils depuis load_regime_thresholds()

    Returns:
        regime_data enrichi (annotations uniquement, classification inchangée)
    """
    if 'CONSERVES_NORM' not in regime_data['regimes']:
        return regime_data

    conserves_indices = regime_data['regimes']['CONSERVES_NORM']['run_indices']

    norm_finals = []
    for idx in conserves_indices:
        norm_final = rows[idx]['features'].get('euclidean_norm_final')
        if norm_final is not None and np.isfinite(norm_final):
            norm_finals.append(norm_final)

    if len(norm_finals) < 2:
        regime_data['regimes']['CONSERVES_NORM']['cv_across_runs'] = None
        regime_data['regimes']['CONSERVES_NORM']['high_dispersion'] = False
        return regime_data

    cv = np.std(norm_finals) / (np.abs(np.mean(norm_finals)) + 1e-10)
    th_cv = thresholds.get('CONSERVES_NORM', {}).get('cv_threshold', 0.10)

    # Annotation pure — pas de reclassement
    regime_data['regimes']['CONSERVES_NORM']['cv_across_runs'] = float(cv)
    regime_data['regimes']['CONSERVES_NORM']['high_dispersion'] = bool(cv > th_cv)
    regime_data['regimes']['CONSERVES_NORM']['cv_threshold'] = float(th_cv)

    return regime_data
