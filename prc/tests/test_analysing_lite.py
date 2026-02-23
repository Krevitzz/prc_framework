"""
test_analysing_lite.py

Tests : prc.analysing (clustering ML)

Méthodologie : Observations pures, validation clustering
"""

from pathlib import Path
import sys
import numpy as np

# Ajouter prc/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysing.clustering_lite import (
    extract_features_matrix,
    run_kmeans_clustering,
)
from analysing.hub_analysing import run_analysing


# =============================================================================
# TESTS FEATURES MATRIX
# =============================================================================

def test_extract_features_matrix():
    """Test extraction matrice features."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_final': 10.0,
                'entropy_initial': 2.5,
                'has_nan_inf': False,  # Flag, devrait être filtré
            }
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {
                'euclidean_norm_final': 12.0,
                'entropy_initial': 2.7,
                'has_nan_inf': False,
            }
        },
    ]
    
    matrix, feature_names, valid_indices = extract_features_matrix(rows)
    
    return {
        'test': 'extract_features_matrix',
        'n_samples': matrix.shape[0],
        'n_features': matrix.shape[1],
        'feature_names': feature_names,
        'n_feature_names': len(feature_names),
        'flags_filtered': 'has_nan_inf' not in feature_names,
        'valid_indices': valid_indices,
    }


def test_extract_features_skip_nan():
    """Test extraction skip NaN."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 10.0, 'entropy_initial': 2.5}
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': np.nan, 'entropy_initial': 2.7}  # NaN → skip
        },
        {
            'composition': {'gamma_id': 'GAM-003', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 12.0, 'entropy_initial': 3.0}
        },
    ]
    
    matrix, feature_names, valid_indices = extract_features_matrix(rows)
    
    return {
        'test': 'extract_features_skip_nan',
        'n_samples': matrix.shape[0],
        'expected_samples': 2,  # 3 rows, 1 NaN skipped
        'correct_n_samples': matrix.shape[0] == 2,
        'valid_indices': valid_indices,
        'correct_indices': valid_indices == [0, 2],
    }


def test_extract_features_incomplete():
    """Test extraction skip incomplete features."""
    rows = [
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 10.0, 'entropy_initial': 2.5}
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'euclidean_norm_final': 12.0}  # entropy_initial manquant → skip
        },
    ]
    
    matrix, feature_names, valid_indices = extract_features_matrix(rows)
    
    return {
        'test': 'extract_features_incomplete',
        'n_samples': matrix.shape[0],
        'expected_samples': 1,  # 2 rows, 1 incomplete skipped
        'correct_n_samples': matrix.shape[0] == 1,
        'valid_indices': valid_indices,
    }


# =============================================================================
# TESTS CLUSTERING
# =============================================================================

def test_run_kmeans_clustering():
    """Test KMeans clustering."""
    # Mock rows avec patterns clairs
    rows = [
        # Cluster 0 : norms faibles
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'norm': 5.0, 'entropy': 2.0}
        },
        {
            'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'norm': 6.0, 'entropy': 2.1}
        },
        # Cluster 1 : norms élevées
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'norm': 20.0, 'entropy': 3.5}
        },
        {
            'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
            'features': {'norm': 22.0, 'entropy': 3.6}
        },
    ]
    
    clustering = run_kmeans_clustering(rows, n_clusters=2, random_state=42)
    
    return {
        'test': 'run_kmeans_clustering',
        'n_clusters': clustering['n_clusters'],
        'n_samples': clustering['n_samples'],
        'n_features': clustering['n_features'],
        'n_labels': len(clustering['labels']),
        'labels': clustering['labels'],
        'inertia': clustering['inertia'],
        'correct_n_samples': clustering['n_samples'] == 4,
        'correct_n_features': clustering['n_features'] == 2,
    }


def test_clustering_centroids():
    """Test clustering centroids."""
    rows = [
        {'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'x': 0.0, 'y': 0.0}},
        {'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'x': 1.0, 'y': 0.0}},
        {'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'x': 10.0, 'y': 10.0}},
        {'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'x': 11.0, 'y': 10.0}},
    ]
    
    clustering = run_kmeans_clustering(rows, n_clusters=2, random_state=42)
    
    centroids = clustering['centroids']
    
    return {
        'test': 'clustering_centroids',
        'n_centroids': len(centroids),
        'centroid_0': centroids[0],
        'centroid_1': centroids[1],
        'correct_n_centroids': len(centroids) == 2,
    }


# =============================================================================
# TESTS HUB ANALYSING
# =============================================================================

def test_run_analysing():
    """Test run_analysing complet."""
    rows = [
        {'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'norm': 5.0, 'entropy': 2.0}},
        {'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'norm': 6.0, 'entropy': 2.1}},
        {'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'norm': 20.0, 'entropy': 3.5}},
        {'composition': {'gamma_id': 'GAM-002', 'encoding_id': 'SYM-002', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'norm': 22.0, 'entropy': 3.6}},
    ]
    
    analysing = run_analysing(rows, n_clusters=2)
    
    return {
        'test': 'run_analysing',
        'n_observations': analysing['n_observations'],
        'has_clustering': 'clustering' in analysing,
        'clustering_not_none': analysing['clustering'] is not None,
        'n_clusters': analysing['clustering']['n_clusters'] if analysing['clustering'] else None,
        'n_samples_clustered': analysing['clustering']['n_samples'] if analysing['clustering'] else None,
    }


def test_analysing_insufficient_samples():
    """Test analysing avec pas assez de samples."""
    rows = [
        {'composition': {'gamma_id': 'GAM-001', 'encoding_id': 'SYM-001', 'modifier_id': 'M0', 'n_dof': 50, 'max_iterations': 1000},
         'features': {'norm': 5.0, 'entropy': 2.0}},
    ]
    
    # n_clusters=3 mais seulement 1 sample → fail
    analysing = run_analysing(rows, n_clusters=3)
    
    return {
        'test': 'analysing_insufficient_samples',
        'clustering_is_none': analysing['clustering'] is None,
        'handled_gracefully': analysing['clustering'] is None,
    }


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_all_tests():
    """Execute tous les tests et affiche résultats."""
    tests = [
        test_extract_features_matrix,
        test_extract_features_skip_nan,
        test_extract_features_incomplete,
        test_run_kmeans_clustering,
        test_clustering_centroids,
        test_run_analysing,
        test_analysing_insufficient_samples,
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
    print("=== TESTS analysing_lite ===\n")
    
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
