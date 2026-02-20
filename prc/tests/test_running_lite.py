"""
test_running_lite.py

Tests : prc.running (data_loading + compositions)

Méthodologie : Observations pures, validation exhaustive options
"""

from pathlib import Path
import sys
import warnings

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loading_lite import (
    load_yaml, merge_configs, discover_gammas, 
    discover_encodings, discover_modifiers
)
from running.compositions import (
    load_run_config, generate_compositions
)


# =============================================================================
# SECTION 1 — TESTS DATA_LOADING
# =============================================================================

def test_load_yaml_path_absolu():
    """Test load_yaml avec Path absolu (YAML de run)."""
    poc_yaml = Path('configs/phases/poc/poc.yaml')
    config = load_yaml(poc_yaml)
    
    return {
        'test': 'load_yaml_path_absolu',
        'has_phase': 'phase' in config,
        'has_axes': 'axes' in config,
        'phase_value': config.get('phase'),
    }


def test_load_yaml_identifier_modes():
    """Test load_yaml avec identifier + modes."""
    results = {}
    
    # Mode default
    default_cfg = load_yaml('operators', mode='default')
    results['default_has_GAM001'] = 'GAM-001' in default_cfg
    results['default_GAM001_beta'] = default_cfg.get('GAM-001', {}).get('beta')
    
    # Mode laxe (doit merger avec default)
    laxe_cfg = load_yaml('operators', mode='laxe')
    results['laxe_has_GAM001'] = 'GAM-001' in laxe_cfg
    results['laxe_GAM001_beta'] = laxe_cfg.get('GAM-001', {}).get('beta')
    results['laxe_beta_changed'] = (
        laxe_cfg.get('GAM-001', {}).get('beta') != 
        default_cfg.get('GAM-001', {}).get('beta')
    )
    
    # Mode strict
    strict_cfg = load_yaml('operators', mode='strict')
    results['strict_GAM001_beta'] = strict_cfg.get('GAM-001', {}).get('beta')
    
    # Vérifier que les 3 modes ont des valeurs différentes
    betas = [
        default_cfg.get('GAM-001', {}).get('beta'),
        laxe_cfg.get('GAM-001', {}).get('beta'),
        strict_cfg.get('GAM-001', {}).get('beta'),
    ]
    results['three_different_betas'] = len(set(betas)) == 3
    
    return {'test': 'load_yaml_modes', **results}


def test_load_yaml_mode_inexistant():
    """Test load_yaml avec mode inexistant (doit warn et utiliser default)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_yaml('operators', mode='nonexistent')
        
        return {
            'test': 'load_yaml_mode_inexistant',
            'warning_raised': len(w) > 0,
            'config_loaded': 'GAM-001' in cfg,
        }


def test_merge_configs():
    """Test merge_configs direct."""
    base = {
        'a': 1,
        'b': {'x': 10, 'y': 20},
        'c': [1, 2, 3],
    }
    
    override = {
        'b': {'x': 99},  # Override partiel dict
        'c': [4, 5],     # Override complet liste
        'd': 4,          # Nouvelle clé
    }
    
    merged = merge_configs(base, override)
    
    return {
        'test': 'merge_configs',
        'a_unchanged': merged['a'] == 1,
        'b_x_overridden': merged['b']['x'] == 99,
        'b_y_preserved': merged['b']['y'] == 20,
        'c_replaced': merged['c'] == [4, 5],
        'd_added': merged['d'] == 4,
    }


def test_discovery():
    """Test discovery atomics."""
    gammas = discover_gammas()
    encodings = discover_encodings()
    modifiers = discover_modifiers()
    
    gamma_ids = [g['id'] for g in gammas]
    encoding_ids = [e['id'] for e in encodings]
    modifier_ids = [m['id'] for m in modifiers]
    
    return {
        'test': 'discovery',
        'n_gammas': len(gammas),
        'n_encodings': len(encodings),
        'n_modifiers': len(modifiers),
        'has_GAM001': 'GAM-001' in gamma_ids,
        'has_SYM001': 'SYM-001' in encoding_ids,
        'has_M0': 'M0' in modifier_ids,
        'all_have_callable': all('callable' in g for g in gammas),
    }


# =============================================================================
# SECTION 2 — TESTS COMPOSITIONS
# =============================================================================

def test_compositions_all():
    """Test generate_compositions avec 'all'."""
    config = {
        'phase': 'test_all',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': 'all',
            'encoding': 'all',
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    gamma_ids = set(c['gamma_id'] for c in compositions)
    encoding_ids = set(c['encoding_id'] for c in compositions)
    
    return {
        'test': 'compositions_all',
        'n_compositions': len(compositions),
        'n_unique_gammas': len(gamma_ids),
        'n_unique_encodings': len(encoding_ids),
        'first_has_callables': all(
            k in compositions[0] for k in 
            ['gamma_callable', 'encoding_callable', 'modifier_callable']
        ),
    }


def test_compositions_explicit_list():
    """Test avec liste explicite IDs."""
    config = {
        'phase': 'test_explicit',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': [{'id': 'GAM-001'}, {'id': 'GAM-009'}],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M1'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    gamma_ids = [c['gamma_id'] for c in compositions]
    
    return {
        'test': 'compositions_explicit_list',
        'n_compositions': len(compositions),  # 2×1×1 = 2
        'gamma_ids': gamma_ids,
        'expected_2_compositions': len(compositions) == 2,
    }


def test_compositions_random():
    """Test avec random: N."""
    config = {
        'phase': 'test_random',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': {'random': 3},
            'encoding': {'random': 5},
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    gamma_ids = set(c['gamma_id'] for c in compositions)
    encoding_ids = set(c['encoding_id'] for c in compositions)
    
    return {
        'test': 'compositions_random',
        'n_compositions': len(compositions),  # ~3×5×1 = 15
        'n_unique_gammas': len(gamma_ids),    # should be 3
        'n_unique_encodings': len(encoding_ids),  # should be 5
    }


def test_compositions_multiline_gamma():
    """Test multi-lignes gamma (séquence)."""
    config = {
        'phase': 'test_multiline',
        'max_iterations': 10,
        'n_dof': 10,
        'axes': {
            'gamma': [
                [{'id': 'GAM-001'}],  # ligne 1
                [{'id': 'GAM-009'}],  # ligne 2
            ],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    return {
        'test': 'compositions_multiline_gamma',
        'n_compositions': len(compositions),  # 1×1×1×1 = 1 (séquence composée)
        'first_gamma_id': compositions[0]['gamma_id'],
        'is_composed': '-' in compositions[0]['gamma_id'],  # GAM-001-GAM-009
    }


def test_compositions_n_dof_list():
    """Test n_dof comme liste (axe itération)."""
    config = {
        'phase': 'test_ndof_list',
        'max_iterations': 10,
        'n_dof': [5, 10, 20],
        'axes': {
            'gamma': [{'id': 'GAM-001'}],
            'encoding': [{'id': 'SYM-001'}],
            'modifier': [{'id': 'M0'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    n_dofs = [c['n_dof'] for c in compositions]
    
    return {
        'test': 'compositions_n_dof_list',
        'n_compositions': len(compositions),  # 1×1×1×3 = 3
        'n_dofs': n_dofs,
        'has_all_three': set(n_dofs) == {5, 10, 20},
    }


def test_compositions_config_mode():
    """Test config_mode laxe dans run."""
    config = {
        'phase': 'test_config_mode',
        'max_iterations': 10,
        'n_dof': 10,
        'config_mode': 'laxe',
        'axes': {
            'gamma': [{'id': 'GAM-001'}],
            'encoding': [{'id': 'SYM-003'}],
            'modifier': [{'id': 'M1'}],
        }
    }
    
    compositions = generate_compositions(config)
    
    # Vérifier que params sont bien chargés depuis laxe
    gamma_params = compositions[0]['gamma_params']
    encoding_params = compositions[0]['encoding_params']
    modifier_params = compositions[0]['modifier_params']
    
    return {
        'test': 'compositions_config_mode',
        'gamma_beta': gamma_params.get('beta'),  # doit être 1.5 (laxe)
        'encoding_sigma': encoding_params.get('sigma'),  # doit être 0.5 (laxe)
        'modifier_sigma': modifier_params.get('sigma'),  # doit être 0.1 (laxe)
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        # data_loading
        test_load_yaml_path_absolu,
        test_load_yaml_identifier_modes,
        test_load_yaml_mode_inexistant,
        test_merge_configs,
        test_discovery,
        
        # compositions
        test_compositions_all,
        test_compositions_explicit_list,
        test_compositions_random,
        test_compositions_multiline_gamma,
        test_compositions_n_dof_list,
        test_compositions_config_mode,
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
    print("=== TESTS EXHAUSTIFS data_loading + compositions ===\n")
    
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
