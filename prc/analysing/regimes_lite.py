"""
prc.analysing.regimes_lite

Responsabilité : Classification régimes basée features scalaires

FIX : Format récurrence uniforme (complet, pas simplifié)
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
    """Classifie régime depuis features scalaires."""
    has_nan = features.get('has_nan_inf', False)
    norm_initial = features.get('euclidean_norm_initial', 1.0)
    norm_final = features.get('euclidean_norm_final', 1.0)
    
    norm_ratio = norm_final / max(norm_initial, 1e-10)
    
    condition_number = features.get('condition_number_final')
    asymmetry_norm = features.get('asymmetry_norm_final')
    
    # 1. PATHOLOGIES
    if has_nan:
        return "NUMERIC_INSTABILITY"
    
    if condition_number is not None:
        th_cond = thresholds.get('NUMERIC_INSTABILITY', {}).get('condition_threshold', 1e6)
        if condition_number > th_cond:
            return "NUMERIC_INSTABILITY"
    
    th_norm_instab = thresholds.get('NUMERIC_INSTABILITY', {}).get('norm_threshold', 1e10)
    if norm_final > th_norm_instab:
        return "NUMERIC_INSTABILITY"
    
    th_effondrement = thresholds.get('EFFONDREMENT', {}).get('ratio_threshold', 0.1)
    if norm_ratio < th_effondrement:
        return "EFFONDREMENT"
    
    # 2. CONSERVATION
    th_conserve_ratio = thresholds.get('CONSERVES_NORM', {}).get('ratio_threshold', 1.3)
    if norm_ratio < th_conserve_ratio:
        return "CONSERVES_NORM"
    
    # 3. SATURATION
    th_sat_ratio = thresholds.get('SATURATION', {}).get('ratio_threshold', 10)
    th_sat_cond = thresholds.get('SATURATION', {}).get('condition_threshold', 1e3)
    
    if norm_ratio > th_sat_ratio:
        if condition_number is None or condition_number < th_sat_cond:
            return "SATURATION"
    
    # 4. ASYMMETRY_BREAKING
    if asymmetry_norm is not None:
        th_asym = thresholds.get('ASYMMETRY_BREAKING', {}).get('asymmetry_ratio_threshold', 0.5)
        asym_ratio = asymmetry_norm / max(norm_final, 1e-10)
        
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
        
        # FIX : Récurrence format COMPLET (cohérent avec outliers_lite)
        gamma_rec = _compute_atomic_recurrence_full(rows, run_indices, 'gamma_id')
        encoding_rec = _compute_atomic_recurrence_full(rows, run_indices, 'encoding_id')
        
        regimes_data[regime] = {
            'count': count,
            'fraction': fraction,
            'recurrence_gamma': gamma_rec,
            'recurrence_encoding': encoding_rec,
            'run_indices': run_indices
        }
    
    regimes_data = dict(sorted(regimes_data.items(), key=lambda x: x[1]['count'], reverse=True))
    
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
    """
    Calcule récurrence atomics (format COMPLET).
    
    Returns:
        {
            'GAM-001': {
                'count': 8,
                'fraction': 0.80,
                'total_subset': 10
            },
            ...
        }
    
    Notes:
        - Format cohérent avec outliers_lite.compute_atomic_recurrence()
        - Top 5 atomics par fraction décroissante
    """
    subset_rows = [rows[i] for i in indices]
    n_total = len(subset_rows)
    
    if n_total == 0:
        return {}
    
    counts = {}
    for row in subset_rows:
        atomic_id = row['composition'][atomic_key]
        counts[atomic_id] = counts.get(atomic_id, 0) + 1
    
    # Format complet
    recurrence = {}
    for atomic_id, count in counts.items():
        recurrence[atomic_id] = {
            'count': count,
            'fraction': count / n_total,
            'total_subset': n_total
        }
    
    # Trier par fraction décroissante, top 5
    recurrence = dict(sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True)[:5])
    
    return recurrence


# =============================================================================
# VÉRIFICATION CV CROSS-RUNS (CONSERVES_NORM)
# =============================================================================

def refine_conserves_norm_with_cv(
    rows: List[Dict],
    regime_data: Dict,
    thresholds: Dict
) -> Dict:
    """Vérifie CV cross-runs pour régime CONSERVES_NORM."""
    if 'CONSERVES_NORM' not in regime_data['regimes']:
        return regime_data
    
    conserves_indices = regime_data['regimes']['CONSERVES_NORM']['run_indices']
    
    norm_finals = []
    for idx in conserves_indices:
        norm_final = rows[idx]['features'].get('euclidean_norm_final')
        if norm_final is not None and np.isfinite(norm_final):
            norm_finals.append(norm_final)
    
    if len(norm_finals) < 2:
        return regime_data
    
    cv = np.std(norm_finals) / (np.abs(np.mean(norm_finals)) + 1e-10)
    
    th_cv = thresholds.get('CONSERVES_NORM', {}).get('cv_threshold', 0.10)
    
    if cv > th_cv:
        regime_data['regimes']['CONSERVES_NORM']['count'] = 0
        regime_data['regimes']['CONSERVES_NORM']['fraction'] = 0.0
        regime_data['regimes']['CONSERVES_NORM']['note'] = f'CV={cv:.3f} > {th_cv} (trop dispersé)'
        
        if 'UNCATEGORIZED' not in regime_data['regimes']:
            regime_data['regimes']['UNCATEGORIZED'] = {
                'count': 0,
                'fraction': 0.0,
                'recurrence_gamma': {},
                'recurrence_encoding': {},
                'run_indices': []
            }
        
        regime_data['regimes']['UNCATEGORIZED']['count'] += len(conserves_indices)
        regime_data['regimes']['UNCATEGORIZED']['run_indices'].extend(conserves_indices)
        regime_data['regimes']['UNCATEGORIZED']['fraction'] = (
            regime_data['regimes']['UNCATEGORIZED']['count'] / regime_data['n_runs']
        )
    
    return regime_data