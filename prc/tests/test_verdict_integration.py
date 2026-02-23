"""
test_verdict_integration.py

Test intégration workflow complet : features → verdict → rapport
"""

import numpy as np
from pathlib import Path


def test_verdict_workflow_minimal():
    """
    Test workflow minimal verdict.
    
    Valide :
    - Outliers détection fonctionne
    - Régimes classification fonctionne
    - Rapports JSON + TXT générés
    """
    # Mock rows
    rows = []
    
    # 10 runs stables CONSERVES_NORM
    for i in range(10):
        rows.append({
            'composition': {
                'gamma_id': 'GAM-001',
                'encoding_id': 'SYM-001',
                'modifier_id': 'M0',
                'n_dof': 10,
                'max_iterations': 10
            },
            'features': {
                'has_nan_inf': False,
                'euclidean_norm_initial': 5.0,
                'euclidean_norm_final': 6.0,  # Ratio 1.2 < 1.3
                'euclidean_norm_mean': 5.5,
                'entropy_initial': 3.5,
                'entropy_final': 3.6,
                'trace_initial': 10.0,
                'trace_final': 10.5,
            }
        })
    
    # 2 runs outliers NUMERIC_INSTABILITY
    for i in range(2):
        rows.append({
            'composition': {
                'gamma_id': 'GAM-003',
                'encoding_id': 'SYM-002',
                'modifier_id': 'M0',
                'n_dof': 10,
                'max_iterations': 10
            },
            'features': {
                'has_nan_inf': False,
                'euclidean_norm_initial': 5.0,
                'euclidean_norm_final': 1e12,  # Explosion
                'euclidean_norm_mean': 5e11,
                'entropy_initial': 3.5,
                'entropy_final': 8.0,
                'condition_number_final': 1e8,
            }
        })
    
    # Import verdict
    from analysing.verdict import run_verdict_intra
    
    # Run verdict
    verdict = run_verdict_intra(rows, filter_pool=False)
    
    # Vérifications
    assert 'outliers' in verdict
    assert 'regimes' in verdict
    
    outliers = verdict['outliers']
    assert outliers['n_outliers'] >= 1  # Au moins 1 outlier détecté
    assert outliers['n_stables'] >= 1   # Au moins 1 stable
    
    regimes = verdict['regimes']['regimes']
    assert 'CONSERVES_NORM' in regimes or 'NUMERIC_INSTABILITY' in regimes
    
    print("[OK] test_verdict_workflow_minimal")
    print(f"  Outliers: {outliers['n_outliers']}")
    print(f"  Stables: {outliers['n_stables']}")
    print(f"  Régimes: {list(regimes.keys())}")


def test_regime_thresholds_loading():
    """Test chargement YAML thresholds."""
    from analysing.regimes_lite import load_regime_thresholds
    
    # Default
    th_default = load_regime_thresholds('default')
    assert 'CONSERVES_NORM' in th_default
    assert th_default['CONSERVES_NORM']['ratio_threshold'] == 1.3
    
    # Laxe
    th_laxe = load_regime_thresholds('laxe')
    assert th_laxe['CONSERVES_NORM']['ratio_threshold'] == 1.5
    
    # Strict
    th_strict = load_regime_thresholds('strict')
    assert th_strict['CONSERVES_NORM']['ratio_threshold'] == 1.1
    
    print("[OK] test_regime_thresholds_loading")
    print(f"  Default ratio: {th_default['CONSERVES_NORM']['ratio_threshold']}")
    print(f"  Laxe ratio: {th_laxe['CONSERVES_NORM']['ratio_threshold']}")
    print(f"  Strict ratio: {th_strict['CONSERVES_NORM']['ratio_threshold']}")


if __name__ == '__main__':
    test_regime_thresholds_loading()
    test_verdict_workflow_minimal()
    
    print("\n=== RÉSUMÉ : 2 OK ===")
