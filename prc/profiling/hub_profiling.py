"""
prc.profiling.hub_profiling

Responsabilité : Orchestration profiling cross-runs (aggregation par axes)

Minimal : gamma, encoding, modifier uniquement
"""

from typing import Dict, List

from profiling.aggregation_lite import aggregate_all_features_by_entity


def run_profiling(rows: List[Dict]) -> Dict:
    """
    Exécute profiling complet (gamma, encoding, modifier).
    
    Args:
        rows : Liste {composition, features}
    
    Returns:
        {
            'gamma': {
                'GAM-001': {
                    'euclidean_norm_final': {median, q1, q3, n_runs},
                    ...
                },
                ...
            },
            'encoding': {...},
            'modifier': {...},
            'n_observations': int,
        }
    
    Workflow:
        1. Aggregate par gamma_id
        2. Aggregate par encoding_id
        3. Aggregate par modifier_id
        4. Return dict unifié
    
    Notes:
        - Skip axes avec <2 entités uniques
        - Features détectées automatiquement
    
    Examples:
        >>> profiling = run_profiling(rows)
        >>> profiling['gamma']['GAM-001']['euclidean_norm_final']['median']
        12.3
    """
    print(f"=== Profiling cross-runs ===")
    print(f"Observations: {len(rows)}\n")
    
    result = {
        'n_observations': len(rows),
    }
    
    # 1. Aggregate par gamma
    print("Aggregating by gamma...")
    gamma_profiles = aggregate_all_features_by_entity(rows, 'gamma_id')
    result['gamma'] = gamma_profiles
    print(f"  {len(gamma_profiles)} gammas profiled")
    
    # 2. Aggregate par encoding
    print("Aggregating by encoding...")
    encoding_profiles = aggregate_all_features_by_entity(rows, 'encoding_id')
    result['encoding'] = encoding_profiles
    print(f"  {len(encoding_profiles)} encodings profiled")
    
    # 3. Aggregate par modifier
    print("Aggregating by modifier...")
    modifier_profiles = aggregate_all_features_by_entity(rows, 'modifier_id')
    result['modifier'] = modifier_profiles
    print(f"  {len(modifier_profiles)} modifiers profiled")
    
    return result
