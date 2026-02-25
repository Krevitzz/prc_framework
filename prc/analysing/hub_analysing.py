"""
prc.analysing.hub_analysing

Responsabilité : Orchestration analysing (patterns ML)

Minimal : clustering uniquement
"""

from typing import Dict, List

from analysing.clustering_lite import run_clustering, run_clustering_stratified


def run_analysing(rows, n_clusters=3, stratified=True):
    """
    Analyse patterns ML.
    
    Args:
        rows : Liste {composition, features, layers}
        n_clusters : Nombre clusters (défaut 3)
        stratified : Si True, analyses par layers (recommandé)
    """
    print(f"\n=== Analysing patterns ML ===")
    print(f"Observations: {len(rows)}")
    
    if stratified:
        print("Mode: Stratified by layers")
        clustering_results = run_clustering_stratified(rows, n_clusters)
        
        return {
            'n_observations': len(rows),
            'clustering_stratified': clustering_results,
            'strategy': 'stratified'
        }
    else:
        print("Mode: Unified (features communes)")
        clustering_results = run_clustering(rows, n_clusters)
        
        return {
            'n_observations': len(rows),
            'clustering': clustering_results,
            'strategy': 'unified'
        }
