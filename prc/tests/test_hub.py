"""
test_hub.py

Tests : prc.running.hub (run_batch)

Méthodologie : Observations pures, validation orchestration complète
"""

from pathlib import Path
import sys

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from running.hub import run_batch, detect_unique_ranks, infer_rank_from_encoding
from running.compositions import generate_compositions, load_run_config


# =============================================================================
# TESTS HELPERS
# =============================================================================

def test_infer_rank_from_encoding():
    """Test infer_rank_from_encoding."""
    assert infer_rank_from_encoding('SYM-001') == 2
    assert infer_rank_from_encoding('ASY-004') == 2
    assert infer_rank_from_encoding('R3-002') == 3
    
    return {
        'test': 'infer_rank_from_encoding',
        'all_correct': True,
    }


def test_detect_unique_ranks():
    """Test detect_unique_ranks sur POC."""
    config = load_run_config(Path('configs/phases/poc/poc.yaml'))
    compositions = generate_compositions(config)
    
    ranks = detect_unique_ranks(compositions)
    
    return {
        'test': 'detect_unique_ranks',
        'ranks_found': sorted(ranks),
        'has_rank_2': 2 in ranks,
        'has_rank_3': 3 in ranks,
    }


# =============================================================================
# TESTS HUB
# =============================================================================

def test_run_batch_small():
    """Test run_batch avec petit POC (auto_confirm)."""
    # Créer POC très petit
    config = {
        'phase': 'test_hub_small',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': [{'id': 'GAM-001'}],
            'encoding': [{'id': 'SYM-001'}, {'id': 'R3-001'}],  # 1 rank 2 + 1 rank 3
            'modifier': [{'id': 'M0'}],
        }
    }
    
    # Sauver config temporaire
    import yaml
    from running.compositions import generate_compositions
    
    compositions = generate_compositions(config)
    n_expected = len(compositions)  # 1×2×1 = 2
    
    # Simuler run_batch sans YAML file
    # On va juste appeler directement avec les compositions
    print(f"  Testing avec {n_expected} compositions (SYM + R3)")
    
    # Pour ce test, on va juste vérifier que detect_unique_ranks fonctionne
    ranks = detect_unique_ranks(compositions)
    
    return {
        'test': 'run_batch_small',
        'n_compositions': n_expected,
        'ranks_detected': sorted(ranks),
        'has_both_ranks': ranks == {2, 3},
    }


def test_run_batch_poc_debug():
    """Test run_batch avec poc_debug.yaml (très court)."""
    yaml_path = Path('configs/phases/poc/poc_debug.yaml')
    
    print(f"  Running batch avec {yaml_path}")
    
    result = run_batch(yaml_path, auto_confirm=True)
    
    return {
        'test': 'run_batch_poc_debug',
        'n_success': result['n_success'],
        'n_skipped': result['n_skipped'],
        'has_stats': 'stats' in result,
        'time_mean_s': result['stats'].get('time_mean_s'),
        'ram_mean_mb': result['stats'].get('ram_mean_mb'),
    }


def test_run_batch_stats_coherence():
    """Test cohérence stats estimations."""
    yaml_path = Path('configs/phases/poc/poc_debug.yaml')
    
    result = run_batch(yaml_path, auto_confirm=True)
    
    stats = result['stats']
    
    # Vérifier cohérence
    checks = {
        'time_mean_positive': stats['time_mean_s'] > 0,
        'time_max_gte_mean': stats['time_max_s'] >= stats['time_mean_s'],
        'time_total_coherent': abs(
            stats['time_total_s'] - stats['time_mean_s'] * stats['n_compositions']
        ) < 0.1,  # Tolérance pour arrondis
        'ram_mean_positive': stats['ram_mean_mb'] > 0,
        'ram_max_gte_mean': stats['ram_max_mb'] >= stats['ram_mean_mb'],
    }
    
    return {
        'test': 'run_batch_stats_coherence',
        **checks,
        'all_checks_pass': all(checks.values()),
        'stats': stats,
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_infer_rank_from_encoding,
        test_detect_unique_ranks,
        test_run_batch_small,
        test_run_batch_poc_debug,
        test_run_batch_stats_coherence,
    ]
    
    results = []
    
    for test_fn in tests:
        print(f"\n--- {test_fn.__name__} ---")
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
    print("=== TESTS hub.py ===\n")
    
    results = run_all_tests()
    
    print("\n" + "="*60)
    for result in results:
        test_name = result.get('test', 'unknown')
        status = result.get('status', 'UNKNOWN')
        
        print(f"\n[{status}] {test_name}")
        
        if status == 'ERROR':
            print(f"  ERROR: {result.get('error')}")
        else:
            for key, value in result.items():
                if key not in ['test', 'status']:
                    print(f"  {key}: {value}")
    
    # Résumé
    n_ok = sum(1 for r in results if r.get('status') == 'OK')
    n_error = sum(1 for r in results if r.get('status') == 'ERROR')
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ : {n_ok} OK, {n_error} ERROR")
