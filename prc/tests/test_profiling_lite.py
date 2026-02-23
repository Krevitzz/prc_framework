"""
test_profiling_lite.py

Tests : prc.profiling (aggregation cross-runs)

Méthodologie : Observations pures, validation aggregation
"""

from pathlib import Path
import sys
import numpy as np

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.aggregation_lite import (
    aggregate_feature_by_entity,
    aggregate_all_features_by_entity,
)
from profiling.hub_profiling import run_profiling


# =============================================================================
# TESTS AGGREGATION
# =============================================================================

def test_aggregate_feature_by_entity():
    """Test aggregation feature unique."""
    # Mock rows
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 10.0}
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 12.0}
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 15.0}
        },
    ]
    
    agg = aggregate_feature_by_entity(rows, 'gamma_id', 'euclidean_norm_final')
    
    return {
        'test': 'aggregate_feature_by_entity',
        'n_gammas': len(agg),
        'has_GAM_001': 'GAM-001' in agg,
        'has_GAM_002': 'GAM-002' in agg,
        'GAM_001_median': agg['GAM-001']['median'],
        'GAM_001_n_runs': agg['GAM-001']['n_runs'],
        'GAM_002_median': agg['GAM-002']['median'],
        'correct_GAM_001_median': abs(agg['GAM-001']['median'] - 11.0) < 0.01,
        'correct_GAM_002_median': abs(agg['GAM-002']['median'] - 15.0) < 0.01,
    }


def test_aggregate_all_features():
    """Test aggregation toutes features."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_final': 10.0,
                'entropy_initial': 2.5,
                'has_nan_inf': False,
            }
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_final': 12.0,
                'entropy_initial': 2.7,
                'has_nan_inf': False,
            }
        },
    ]
    
    agg = aggregate_all_features_by_entity(rows, 'gamma_id')
    
    return {
        'test': 'aggregate_all_features',
        'n_gammas': len(agg),
        'n_features_GAM_001': len(agg['GAM-001']),
        'has_norm': 'euclidean_norm_final' in agg['GAM-001'],
        'has_entropy': 'entropy_initial' in agg['GAM-001'],
        'has_flag': 'has_nan_inf' in agg['GAM-001'],  # Should be False (filtered)
        'flags_filtered': 'has_nan_inf' not in agg['GAM-001'],
    }


def test_aggregate_skip_nan():
    """Test aggregation skip NaN."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 10.0}
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': np.nan}
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-003', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 12.0}
        },
    ]
    
    agg = aggregate_feature_by_entity(rows, 'gamma_id', 'euclidean_norm_final')
    
    return {
        'test': 'aggregate_skip_nan',
        'n_runs_GAM_001': agg['GAM-001']['n_runs'],
        'correct_n_runs': agg['GAM-001']['n_runs'] == 2,  # 3 rows, 1 NaN skipped
        'median': agg['GAM-001']['median'],
        'correct_median': abs(agg['GAM-001']['median'] - 11.0) < 0.01,
    }


# =============================================================================
# TESTS HUB PROFILING
# =============================================================================

def test_run_profiling():
    """Test run_profiling complet."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 10.0, 'entropy_initial': 2.5}
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M1', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 12.0, 'entropy_initial': 2.7}
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 15.0, 'entropy_initial': 3.0}
        },
    ]
    
    profiling = run_profiling(rows)
    
    return {
        'test': 'run_profiling',
        'n_observations': profiling['n_observations'],
        'n_gammas': len(profiling['gamma']),
        'n_encodings': len(profiling['encoding']),
        'n_modifiers': len(profiling['modifier']),
        'has_gamma': 'gamma' in profiling,
        'has_encoding': 'encoding' in profiling,
        'has_modifier': 'modifier' in profiling,
        'gamma_GAM_001_median': profiling['gamma']['GAM-001']['euclidean_norm_final']['median'],
    }


def test_profiling_multiple_features():
    """Test profiling avec plusieurs features."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_initial': 5.0,
                'euclidean_norm_final': 10.0,
                'euclidean_norm_mean': 7.5,
                'entropy_initial': 2.5,
                'entropy_final': 3.0,
            }
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_initial': 6.0,
                'euclidean_norm_final': 12.0,
                'euclidean_norm_mean': 9.0,
                'entropy_initial': 2.7,
                'entropy_final': 3.2,
            }
        },
    ]
    
    profiling = run_profiling(rows)
    
    n_features_gamma = len(profiling['gamma']['GAM-001'])
    
    return {
        'test': 'profiling_multiple_features',
        'n_features_profiled': n_features_gamma,
        'expected_n_features': 5,
        'correct_n_features': n_features_gamma == 5,
        'has_all_norms': all(
            f in profiling['gamma']['GAM-001']
            for f in ['euclidean_norm_initial', 'euclidean_norm_final', 'euclidean_norm_mean']
        ),
        'has_all_entropies': all(
            f in profiling['gamma']['GAM-001']
            for f in ['entropy_initial', 'entropy_final']
        ),
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_aggregate_feature_by_entity,
        test_aggregate_all_features,
        test_aggregate_skip_nan,
        test_run_profiling,
        test_profiling_multiple_features,
    ]
    
    results = []
    
    for test_fn in tests:
        try:
            result = test_fn()
            result['status'] = 'OK'
        except Exception as e:
            result = {
                'test': test_fn.__name__,
                'status': 'ERROR',
                'error': str(e),
            }
        
        results.append(result)
    
    return results


if __name__ == '__main__':
    print("=== TESTS profiling_lite ===\n")
    
    results = run_all_tests()
    
    for result in results:
        test_name = result.get('test', 'unknown')
        status = result.get('status', 'UNKNOWN')
        
        print(f"[{status}] {test_name}")
        
        if status == 'ERROR':
            print(f"  ERROR: {result.get('error')}")
        else:
            for key, value in result.items():
                if key not in ['test', 'status']:
                    print(f"  {key}: {value}")
        
        print()
    
    # Résumé
    n_ok = sum(1 for r in results if r.get('status') == 'OK')
    n_error = sum(1 for r in results if r.get('status') == 'ERROR')
    
    print(f"=== RÉSUMÉ : {n_ok} OK, {n_error} ERROR ===")
