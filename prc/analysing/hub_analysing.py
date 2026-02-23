"""
prc.analysing.hub_analysing

Responsabilité : Orchestration analysing (patterns ML)

Minimal : clustering uniquement
"""

from typing import Dict, List

from analysing.clustering_lite import run_clustering


def run_analysing(rows: List[Dict], n_clusters: int = 3) -> Dict:
    """
    Orchestration analysing patterns ML.
    
    Args:
        rows       : Liste {composition, features}
        n_clusters : Nombre clusters KMeans (défaut 3)
    
    Returns:
        {
            'n_observations': int,
            'clustering': Dict ou None
        }
    
    Workflow:
        1. Clustering similarity grouping (features communes)
        2. (Futur : variance analysis, interactions)
    """
    print(f"\n=== Analysing patterns ML ===")
    print(f"Observations: {len(rows)}")
    
    # Clustering
    print(f"\nRunning KMeans clustering (n_clusters={n_clusters})...")
    clustering_results = run_clustering(rows, n_clusters)
    
    return {
        'n_observations': len(rows),
        'clustering': clustering_results
    }
