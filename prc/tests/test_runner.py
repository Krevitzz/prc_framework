"""
test_runner.py

Tests : prc.running.runner (run_single)

Méthodologie : Observations pures, validation exhaustive cas nominaux + erreurs
"""

from pathlib import Path
import sys
import numpy as np
import warnings

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from running.compositions import load_run_config, generate_compositions
from running.runner import run_single, RunnerError


# =============================================================================
# TESTS RUNNER
# =============================================================================

def test_run_single_nominal():
    """Test run_single nominal avec composition POC."""
    import time
    
    # Charger une composition simple
    config = load_run_config(Path('configs/phases/poc/poc.yaml'))
    compositions = generate_compositions(config)
    
    # Prendre première composition
    composition = compositions[0]
    
    print(f"  Running: {composition['gamma_id']} × {composition['encoding_id']} × {composition['modifier_id']}")
    print(f"  max_iterations: {composition['max_iterations']}")
    
    # Run avec timing
    start = time.time()
    history = run_single(composition)
    elapsed = time.time() - start
    
    print(f"  Temps run: {elapsed:.3f}s")
    print(f"  Norm initial: {np.linalg.norm(history[0]):.3f}")
    print(f"  Norm final: {np.linalg.norm(history[-1]):.3f}")
    
    return {
        'test': 'run_single_nominal',
        'history_shape': history.shape,
        'history_dtype': str(history.dtype),
        'n_iterations': history.shape[0],
        'state_shape': history.shape[1:],
        'has_finite_values': np.all(np.isfinite(history)),
        'first_state_norm': float(np.linalg.norm(history[0])),
        'last_state_norm': float(np.linalg.norm(history[-1])),
        'elapsed_seconds': elapsed,
    }


def test_run_single_multicombo():
    """Test run_single sur plusieurs compositions."""
    import time
    
    config = load_run_config(Path('configs/phases/poc/poc.yaml'))
    compositions = generate_compositions(config)
    
    print(f"  Testing 5 compositions...")
    
    # Tester 5 premières compositions
    results = []
    total_time = 0
    
    for i, composition in enumerate(compositions[:5]):
        start = time.time()
        history = run_single(composition)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"    [{i}] {composition['gamma_id']} × {composition['encoding_id']}: {elapsed:.3f}s")
        
        results.append({
            'combo_idx': i,
            'gamma_id': composition['gamma_id'],
            'encoding_id': composition['encoding_id'],
            'history_shape': history.shape,
            'all_finite': np.all(np.isfinite(history)),
            'elapsed_seconds': elapsed,
        })
    
    print(f"  Total time: {total_time:.3f}s")
    
    return {
        'test': 'run_single_multicombo',
        'n_combos_tested': len(results),
        'all_succeeded': all(r['all_finite'] for r in results),
        'sample_results': results[:3],
        'total_elapsed_seconds': total_time,
    }


def test_run_single_explosion():
    """Test comportement avec gamma explosif (GAM-003)."""
    config = {
        'phase': 'test_explosion',
        'max_iterations': 50,  # Court pour accélérer
        'n_dof': 10,
        'axes': {
            'gamma': [{'id': 'GAM-003'}],  # Croissance exponentielle
            'encoding': [{'id': 'SYM-002'}],  # Random uniforme
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    composition = compositions[0]
    
    # Run avec warning attendu
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        history = run_single(composition)
        
        return {
            'test': 'run_single_explosion',
            'warning_raised': len(w) > 0,
            'history_shorter_than_max': history.shape[0] < 51,  # max_it + 1
            'history_not_empty': len(history) > 0,
            'last_iteration': history.shape[0] - 1,
        }


def test_run_single_shapes_consistency():
    """Test cohérence shapes sur tout le run."""
    config = {
        'phase': 'test_shapes',
        'max_iterations': 20,
        'n_dof': 15,
        'axes': {
            'gamma': [{'id': 'GAM-001'}],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M1'}],
        }
    }
    
    compositions = generate_compositions(config)
    composition = compositions[0]
    
    history = run_single(composition)
    
    # Vérifier toutes shapes identiques
    shapes_unique = set(tuple(state.shape) for state in history)
    
    return {
        'test': 'run_single_shapes_consistency',
        'history_shape': history.shape,
        'n_unique_shapes': len(shapes_unique),
        'all_same_shape': len(shapes_unique) == 1,
        'state_shape': history[0].shape,
    }


def test_run_single_composed_gamma():
    """Test run avec gamma composé (séquence)."""
    config = {
        'phase': 'test_composed',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': [
                [{'id': 'GAM-001'}],  # Ligne 1
                [{'id': 'GAM-002'}],  # Ligne 2
            ],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    composition = compositions[0]
    
    history = run_single(composition)
    
    return {
        'test': 'run_single_composed_gamma',
        'gamma_id': composition['gamma_id'],
        'is_composed': '-' in composition['gamma_id'],
        'history_shape': history.shape,
        'run_succeeded': np.all(np.isfinite(history)),
    }


def test_run_single_different_ndofs():
    """Test run avec différents n_dof."""
    results = []
    
    for n_dof in [5, 10, 20]:
        config = {
            'phase': 'test_ndof',
            'max_iterations': 10,
            'n_dof': n_dof,
            'axes': {
                'gamma': [{'id': 'GAM-001'}],
                'encoding': [{'id': 'SYM-001'}],
                'modifier': [{'id': 'M0'}],
            }
        }
        
        compositions = generate_compositions(config)
        history = run_single(compositions[0])
        
        results.append({
            'n_dof': n_dof,
            'history_shape': history.shape,
            'state_shape_correct': history.shape[1:] == (n_dof, n_dof),
        })
    
    return {
        'test': 'run_single_different_ndofs',
        'results': results,
        'all_correct': all(r['state_shape_correct'] for r in results),
    }


def test_run_single_short_multiline_debug():
    """Test gamma composé court avec traçage détaillé (poc_debug.yaml)."""
    import time
    
    config = load_run_config(Path('configs/phases/poc/poc_debug.yaml'))
    compositions = generate_compositions(config)
    
    composition = compositions[0]
    
    print(f"  Running: {composition['gamma_id']} × {composition['encoding_id']} × {composition['modifier_id']}")
    print(f"  max_iterations: {composition['max_iterations']}")
    print(f"  n_dof: {composition['n_dof']}")
    print(f"  Gamma type: {type(composition['gamma_callable'])}")
    
    # Run avec timing
    start = time.time()
    history = run_single(composition)
    elapsed = time.time() - start
    
    print(f"  Temps run: {elapsed:.3f}s")
    print(f"  History shape: {history.shape}")
    
    # Tracer normes à chaque étape
    print(f"  Normes par iteration:")
    for i in range(min(len(history), 10)):
        norm = np.linalg.norm(history[i])
        print(f"    Iter {i}: norm={norm:.3f}")
    
    return {
        'test': 'run_single_short_multiline_debug',
        'gamma_id': composition['gamma_id'],
        'is_composed': '-' in composition['gamma_id'],
        'history_shape': history.shape,
        'n_iterations_run': history.shape[0],
        'expected_iterations': composition['max_iterations'] + 1,
        'run_succeeded': np.all(np.isfinite(history)),
        'elapsed_seconds': elapsed,
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_run_single_nominal,
        test_run_single_multicombo,
        test_run_single_explosion,
        test_run_single_shapes_consistency,
        test_run_single_composed_gamma,
        test_run_single_different_ndofs,
        test_run_single_short_multiline_debug,
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
    print("=== TESTS runner.py ===\n")
    
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