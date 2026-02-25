"""
prc.analysing.clustering_lite

Responsabilité : Clustering simple (KMeans) sur features COMMUNES cross-runs

Minimal : KMeans sklearn
"""

import numpy as np
from typing import Dict, List, Set
from sklearn.cluster import KMeans
from featuring.layers_lite import group_rows_by_layers


def _extract_common_features(rows: List[Dict]) -> Set[str]:
    """
    Identifie features présentes dans TOUS les runs.
    
    Args:
        rows : Liste {composition, features}
    
    Returns:
        Set features communes (intersection)
    
    Notes:
        - Exclut features non numériques (has_*)
        - Retourne intersection stricte
    """
    if len(rows) == 0:
        return set()
    
    # Features premier run
    common = set(rows[0]['features'].keys())
    
    # Intersection avec tous les autres
    for row in rows[1:]:
        common &= set(row['features'].keys())
    
    # Filtrer features non numériques
    common = {k for k in common if not k.startswith('has_')}
    
    return common


def run_clustering(rows: List[Dict], n_clusters: int = 3) -> Dict:
    """
    Clustering KMeans sur features COMMUNES.
    
    Args:
        rows       : Liste {composition, features}
        n_clusters : Nombre clusters (défaut 3)
    
    Returns:
        {
            'n_clusters': int,
            'n_samples': int,
            'n_features': int,
            'labels': List[int],
            'centroids': List[List[float]],
            'inertia': float,
            'feature_names': List[str]
        }
        ou None si échec
    
    Notes:
        - Compare uniquement features présentes dans TOUS runs
        - Skip runs avec NaN/Inf
        - Minimum 2 samples pour clustering
    """
    # Identifier features communes
    common_features = _extract_common_features(rows)
    
    if len(common_features) == 0:
        print("  WARNING: Aucune feature commune — clustering skipped")
        return None
    
    # Trier pour ordre stable
    feature_names = sorted(common_features)
    
    # Extraire features matrix
    features_matrix = []
    valid_indices = []
    
    for i, row in enumerate(rows):
        features = row['features']
        
        # Extraire valeurs features communes
        vector = [features[k] for k in feature_names]
        
        # Skip si NaN/Inf présents
        if not all(np.isfinite(v) for v in vector):
            continue
        
        features_matrix.append(vector)
        valid_indices.append(i)
    
    n_valid = len(features_matrix)
    
    if n_valid < n_clusters:
        print(f"  WARNING: Pas assez de samples valides ({n_valid}) pour {n_clusters} clusters")
        return None
    
    # KMeans
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_matrix)
        
        print(f"  Samples clustered: {n_valid}")
        print(f"  Features used: {len(feature_names)}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        
        return {
            'n_clusters': n_clusters,
            'n_samples': n_valid,
            'n_features': len(feature_names),
            'labels': labels.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_),
            'feature_names': feature_names,
            'valid_indices': valid_indices
        }
    
    except Exception as e:
        print(f"  WARNING: Clustering failed - {str(e)}")
        return None


def run_clustering_stratified(rows: List[Dict], n_clusters: int = 3) -> Dict:
    """
    Clustering par layers (générique, extensible automatiquement).
    
    Args:
        rows : Liste {composition, features, layers}
        n_clusters : Nombre clusters (défaut 3)
    
    Returns:
        {
            'universal': {...},
            'matrix_2d': {...},
            'tensor_3d': {...},
            # Nouveaux layers auto-ajoutés
        }
    
    Notes:
        - Extensible : Nouveau layer = 0 ligne code
        - Skip layers avec <n_clusters samples
    """

    
    # Grouper rows par layer
    layers_groups = group_rows_by_layers(rows)
    
    # Clustering par layer (boucle générique)
    results = {}
    
    for layer_name, layer_rows in layers_groups.items():
        # Skip si pas assez samples
        if len(layer_rows) < n_clusters:
            print(f"  [SKIP] {layer_name}: {len(layer_rows)} samples < {n_clusters}")
            continue
        
        # Clustering
        result = run_clustering(layer_rows, n_clusters)
        
        if result is not None:
            results[layer_name] = result
            print(f"  ✓ {layer_name}: {len(layer_rows)} samples, {result['n_features']} features")
    
    return results