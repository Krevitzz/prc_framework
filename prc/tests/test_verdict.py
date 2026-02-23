"""
test_verdict.py

Tests : prc.analysing.verdict (profiling + analysing)

Méthodologie : Observations pures, tests verdict RAM et Parquet
"""

from pathlib import Path
import sys
import numpy as np

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysing.verdict import (
    load_pool_requirements,
    filter_rows_by_pool,
    run_verdict_intra,
)


# =============================================================================
# TESTS POOL REQUIREMENTS
# =============================================================================

def test_load_pool_requirements():
    """Test chargement pool_requirements.yaml."""
    requirements = load_pool_requirements()
    
    return {
        'test': 'load_pool_requirements',
        'has_n_dof': 'n_dof' in requirements,
        'has_deprecated': 'deprecated' in requirements,
        'n_dof_min_is_null': requirements['n_dof']['min'] is None,
        'deprecated_gammas_empty': len(requirements['deprecated']['gammas']) == 0,
    }


def test_filter_rows_by_pool_no_constraints():
    """Test filtrage sans contraintes (permissif)."""
    # Mock rows
    rows = [
        {
            'composition': {
                'gamma_id': 'GAM-001',
                'encoding_id': 'SYM-001',
                'modifier_id': 'M0',
                'n_dof': 50,
                'max_iterations': 1000,
            },
            'features': {'euclidean_norm_initial': 5.0}
        },
        {
            'composition': {
                'gamma_id': 'GAM-002',
                'encoding_id': 'ASY-001',
                'modifier_id': 'M1',
                'n_dof': 100,
                'max_iterations': 500,
            },
            'features': {'euclidean_norm_initial': 8.0}
        }
    ]
    
    # Requirements permissifs (tous null)
    requirements = {
        'n_dof': {'min': None, 'max': None},
        'max_iterations': {'min': None, 'max': None},
        'deprecated': {'gammas': [], 'encodings': [], 'modifiers': []},
    }
    
    filtered, n_skipped, skip_reasons = filter_rows_by_pool(
        rows, requirements, verbose=False
    )
    
    return {
        'test': 'filter_rows_by_pool_no_constraints',
        'n_filtered': len(filtered),
        'n_skipped': n_skipped,
        'all_passed': len(filtered) == 2,
    }


def test_filter_rows_by_pool_with_constraints():
    """Test filtrage avec contraintes."""
    # Mock rows
    rows = [
        {
            'composition': {
                'gamma_id': 'GAM-001',
                'encoding_id': 'SYM-001',
                'modifier_id': 'M0',
                'n_dof': 50,
                'max_iterations': 1000,
            },
            'features': {}
        },
        {
            'composition': {
                'gamma_id': 'GAM-002',
                'encoding_id': 'ASY-001',
                'modifier_id': 'M1',
                'n_dof': 100,
                'max_iterations': 500,
            },
            'features': {}
        },
        {
            'composition': {
                'gamma_id': 'GAM-003',  # Deprecated
                'encoding_id': 'R3-001',
                'modifier_id': 'M0',
                'n_dof': 200,
                'max_iterations': 200,
            },
            'features': {}
        }
    ]
    
    # Requirements avec contraintes
    requirements = {
        'n_dof': {'min': 100, 'max': None},
        'max_iterations': {'min': None, 'max': None},
        'deprecated': {'gammas': ['GAM-003'], 'encodings': [], 'modifiers': []},
    }
    
    filtered, n_skipped, skip_reasons = filter_rows_by_pool(
        rows, requirements, verbose=False
    )
    
    return {
        'test': 'filter_rows_by_pool_with_constraints',
        'n_filtered': len(filtered),
        'n_skipped': n_skipped,
        'expected_filtered': 1,  # Seul GAM-002 avec n_dof=100 passe
        'correct_filtering': len(filtered) == 1,
        'skip_reasons': skip_reasons,
        'has_n_dof_skip': 'n_dof_too_low' in skip_reasons,
        'has_deprecated_skip': 'gamma_deprecated' in skip_reasons,
    }


# =============================================================================
# TESTS VERDICT
# =============================================================================

def test_run_verdict_intra_no_filter():
    """Test verdict sans filtrage pool."""
    # Mock rows
    rows = [
        {
            'composition': {
                'gamma_id': 'GAM-001',
                'encoding_id': 'SYM-001',
                'modifier_id': 'M0',
                'n_dof': 50,
                'max_iterations': 1000,
            },
            'features': {'euclidean_norm_initial': 5.0}
        }
    ]
    
    result = run_verdict_intra(rows, filter_pool=False)
    
    return {
        'test': 'run_verdict_intra_no_filter',
        'has_profiling': 'profiling' in result,
        'has_analysing': 'analysing' in result,
        'has_metadata': 'metadata' in result,
        'n_filtered': result.get('n_filtered'),
        'n_skipped_pool': result.get('n_skipped_pool'),
        'profiling_not_none': result['profiling'] is not None,
        'analysing_is_dict': isinstance(result['analysing'], dict),
    }


def test_run_verdict_intra_with_filter():
    """Test verdict avec filtrage pool."""
    # Mock rows
    rows = [
        {
            'composition': {
                'gamma_id': 'GAM-001',
                'encoding_id': 'SYM-001',
                'modifier_id': 'M0',
                'n_dof': 50,
                'max_iterations': 1000,
            },
            'features': {'euclidean_norm_initial': 5.0}
        }
    ]
    
    # Avec filter_pool=True, charge requirements (tous null → passe)
    result = run_verdict_intra(rows, filter_pool=True)
    
    return {
        'test': 'run_verdict_intra_with_filter',
        'n_filtered': result.get('n_filtered'),
        'n_skipped_pool': result.get('n_skipped_pool'),
        'rows_passed': result['n_filtered'] == 1,
        'has_metadata': 'metadata' in result,
        'metadata_has_n_gammas': 'n_gammas' in result.get('metadata', {}),
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_load_pool_requirements,
        test_filter_rows_by_pool_no_constraints,
        test_filter_rows_by_pool_with_constraints,
        test_run_verdict_intra_no_filter,
        test_run_verdict_intra_with_filter,
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
    print("=== TESTS verdict.py ===\n")
    
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
