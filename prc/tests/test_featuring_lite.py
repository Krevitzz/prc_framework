"""
test_featuring_lite.py

Tests : prc.featuring (hub_lite, extractor_lite, layers_lite, universal_lite)

Méthodologie : Observations pures, validation extraction features
"""

from pathlib import Path
import sys
import numpy as np

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from featuring.hub_featuring import extract_features
from featuring.layers_lite import inspect_history
from featuring.extractor_lite import extract_universal_features, compute_projection
from featuring.registries.universal_lite import (
    euclidean_norm, entropy, mean_value, std_value
)
from utils.data_loading_lite import load_yaml


# =============================================================================
# TESTS REGISTRES
# =============================================================================

def test_universal_functions():
    """Test fonctions registre universal."""
    # Matrice 10×10
    state = np.random.rand(10, 10)
    
    norm = euclidean_norm(state)
    ent = entropy(state, bins=50)
    mean = mean_value(state)
    std = std_value(state)
    
    return {
        'test': 'universal_functions',
        'norm_positive': norm > 0,
        'entropy_positive': ent >= 0,
        'mean_in_range': 0 <= mean <= 1,  # rand uniform [0,1]
        'std_positive': std >= 0,
        'all_float': all(isinstance(x, float) for x in [norm, ent, mean, std]),
    }


def test_universal_functions_rank3():
    """Test fonctions universal sur rank 3."""
    # Tensor 5×5×5
    state = np.random.rand(5, 5, 5)
    
    norm = euclidean_norm(state)
    ent = entropy(state)
    
    return {
        'test': 'universal_functions_rank3',
        'norm_positive': norm > 0,
        'entropy_positive': ent >= 0,
        'functions_work_rank3': True,
    }


# =============================================================================
# TESTS LAYERS
# =============================================================================

def test_inspect_history_rank2():
    """Test inspect_history rank 2."""
    history = np.random.rand(201, 10, 10)
    info = inspect_history(history)
    
    return {
        'test': 'inspect_history_rank2',
        'rank': info['rank'],
        'shape': info['shape'],
        'is_square': info['is_square'],
        'rank_correct': info['rank'] == 2,
        'is_square_correct': info['is_square'] == True,
    }


def test_inspect_history_rank3():
    """Test inspect_history rank 3."""
    history = np.random.rand(101, 5, 5, 5)
    info = inspect_history(history)
    
    return {
        'test': 'inspect_history_rank3',
        'rank': info['rank'],
        'is_cubic': info['is_cubic'],
        'rank_correct': info['rank'] == 3,
        'is_cubic_correct': info['is_cubic'] == True,
    }


# =============================================================================
# TESTS EXTRACTOR
# =============================================================================

def test_compute_projection():
    """Test projections temporelles."""
    history = np.array([
        [[1, 2], [3, 4]],  # t=0
        [[5, 6], [7, 8]],  # t=1
        [[9, 10], [11, 12]],  # t=2
    ])
    
    initial = compute_projection(history, 'initial')
    final = compute_projection(history, 'final')
    mean = compute_projection(history, 'mean')
    
    return {
        'test': 'compute_projection',
        'initial_correct': np.array_equal(initial, [[1, 2], [3, 4]]),
        'final_correct': np.array_equal(final, [[9, 10], [11, 12]]),
        'mean_shape': mean.shape,
        'mean_value': float(np.mean(mean)),
    }


def test_extract_universal_features():
    """Test extract_universal_features avec config."""
    history = np.random.rand(51, 10, 10)
    
    config = {
        'functions': [
            {
                'name': 'euclidean_norm',
                'projections': ['initial', 'final', 'mean'],
            },
            {
                'name': 'entropy',
                'projections': ['initial', 'final'],
                'params': {'bins': 50},
            }
        ]
    }
    
    features = extract_universal_features(history, config)
    
    expected_keys = [
        'euclidean_norm_initial',
        'euclidean_norm_final',
        'euclidean_norm_mean',
        'entropy_initial',
        'entropy_final',
    ]
    
    return {
        'test': 'extract_universal_features',
        'n_features': len(features),
        'expected_n_features': 5,
        'has_all_keys': all(k in features for k in expected_keys),
        'all_float': all(isinstance(v, float) for v in features.values()),
        'sample_keys': list(features.keys())[:3],
    }


# =============================================================================
# TESTS HUB
# =============================================================================

def test_extract_features_rank2():
    """Test extract_features complet rank 2."""
    history = np.random.rand(101, 20, 20)
    
    # Load config depuis YAML
    config_path = Path('featuring/configs/minimal/universal.yaml')
    
    print(f"  Loading config from: {config_path}")
    print(f"  Path exists: {config_path.exists()}")
    
    universal_config = load_yaml(config_path)
    print(f"  Loaded config keys: {list(universal_config.keys())}")
    print(f"  Config content: {universal_config}")
    
    config = {'universal': universal_config}
    
    features = extract_features(history, config)
    
    print(f"  Extracted features: {list(features.keys())}")
    
    return {
        'test': 'extract_features_rank2',
        'n_features': len(features),
        'has_nan_inf_flag': 'has_nan_inf' in features,
        'has_nan_inf': features.get('has_nan_inf'),
        'has_norm_features': 'euclidean_norm_initial' in features,
        'has_entropy_features': 'entropy_initial' in features,
        'sample_features': {k: features[k] for k in list(features.keys())[:3]},
        'all_feature_keys': list(features.keys()),
    }


def test_extract_features_rank3():
    """Test extract_features complet rank 3."""
    history = np.random.rand(51, 5, 5, 5)
    
    config_path = Path('featuring/configs/minimal/universal.yaml')
    universal_config = load_yaml(config_path)
    
    config = {'universal': universal_config}
    
    features = extract_features(history, config)
    
    return {
        'test': 'extract_features_rank3',
        'n_features': len(features),
        'functions_work_rank3': len(features) > 0,
        'all_finite': all(np.isfinite(v) for v in features.values() if isinstance(v, (int, float))),
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_universal_functions,
        test_universal_functions_rank3,
        test_inspect_history_rank2,
        test_inspect_history_rank3,
        test_compute_projection,
        test_extract_universal_features,
        test_extract_features_rank2,
        test_extract_features_rank3,
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
    print("=== TESTS featuring_lite ===\n")
    
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
