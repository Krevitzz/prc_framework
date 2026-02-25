"""
prc.profiling.aggregation_lite

Responsabilité : Aggregation basique cross-runs (median, IQR, n_runs)

Minimal : 3 stats uniquement
"""

import numpy as np
from typing import Dict, List

def find_feature_variants(features: Dict, base_name: str) -> List[str]:
    """
    Trouve variantes feature avec projections (_initial, _final, _mean).
    
    Args:
        features : Dict features d'un run
        base_name : Nom base (ex: 'euclidean_norm', 'trace')
    
    Returns:
        Liste variantes trouvées (ex: ['euclidean_norm_final', 'euclidean_norm_initial'])
    
    Examples:
        >>> find_feature_variants({'euclidean_norm_final': 12.3}, 'euclidean_norm')
        ['euclidean_norm_final']
    """
    variants = []
    
    # Chercher variantes avec suffixes
    for suffix in ['_initial', '_final', '_mean', '_max', '_min']:
        variant = base_name + suffix
        if variant in features:
            variants.append(variant)
    
    # Chercher aussi base sans suffixe
    if base_name in features:
        variants.append(base_name)
    
    return variants
    
def aggregate_feature_by_entity(
    rows: List[Dict],
    entity_key: str,
    feature_name: str
) -> Dict[str, Dict]:
    """
    Agrège feature par entité (median, Q1, Q3, n_runs).
    
    Args:
        rows        : Liste {composition, features}
        entity_key  : 'gamma_id' | 'encoding_id' | 'modifier_id'
        feature_name: Nom feature à agréger (ex: 'euclidean_norm_final')
    
    Returns:
        {
            'GAM-001': {
                'median': 12.3,
                'q1': 10.5,
                'q3': 15.8,
                'n_runs': 21
            },
            ...
        }
    
    Notes:
        - Skip si feature absente ou NaN
        - Calcul sur toutes valeurs valides cross-runs
    
    Examples:
        >>> aggregate_feature_by_entity(rows, 'gamma_id', 'euclidean_norm_final')
    """
    # Groupby entity_key
    grouped = {}
    
    for row in rows:
        entity_id = row['composition'][entity_key]
        features = row['features']
        
        # Chercher variantes projections
        variants = find_feature_variants(features, feature_name)
        
        for variant in variants:
            value = features[variant]
            
            # Skip NaN/Inf
            if not np.isfinite(value):
                continue
            
            if entity_id not in grouped:
                grouped[entity_id] = []
            
            grouped[entity_id].append(value)

    
    # Aggregate
    aggregated = {}
    
    for entity_id, values in grouped.items():
        if len(values) == 0:
            continue
        
        values_array = np.array(values)
        
        aggregated[entity_id] = {
            'median': float(np.median(values_array)),
            'q1': float(np.percentile(values_array, 25)),
            'q3': float(np.percentile(values_array, 75)),
            'n_runs': len(values),
        }
    
    return aggregated


def aggregate_all_features_by_entity(
    rows: List[Dict],
    entity_key: str
) -> Dict[str, Dict[str, Dict]]:
    """
    Agrège toutes features par entité.
    
    Args:
        rows       : Liste {composition, features}
        entity_key : 'gamma_id' | 'encoding_id' | 'modifier_id'
    
    Returns:
        {
            'GAM-001': {
                'euclidean_norm_final': {median, q1, q3, n_runs},
                'entropy_initial': {median, q1, q3, n_runs},
                ...
            },
            ...
        }
    
    Notes:
        - Détecte features automatiquement depuis rows
        - Skip features avec <2 valeurs valides
    """
    # Détecter features disponibles
    all_features = set()
    for row in rows:
        all_features.update(row['features'].keys())
    
    # Filtrer flags non numériques
    numeric_features = sorted([
        f for f in all_features 
        if not f.startswith('has_')
    ])
    
    # Aggregate chaque feature
    result = {}
    
    for feature_name in numeric_features:
        feature_agg = aggregate_feature_by_entity(rows, entity_key, feature_name)
        
        # Merge dans result
        for entity_id, stats in feature_agg.items():
            if entity_id not in result:
                result[entity_id] = {}
            
            result[entity_id][feature_name] = stats
    
    return result
