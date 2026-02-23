"""
prc.analysing.concordance_lite

Responsabilité : Analyses concordance cross-phases (inter-run)

Minimal : stub kappa + DTW (mécanisme uniquement)
"""

import numpy as np
from typing import Dict, List


def compute_kappa_stub(labels_phase1: List[str], labels_phase2: List[str]) -> float:
    """
    Stub Cohen's Kappa (concordance classifications).
    
    Args:
        labels_phase1 : Labels phase 1 (ex: régimes)
        labels_phase2 : Labels phase 2
    
    Returns:
        Kappa ∈ [-1, 1]
        - 1.0 = accord parfait
        - 0.0 = accord aléatoire
        - <0 = désaccord
    
    Notes:
        - STUB : retourne 0.5 fixe
        - FULL : implémenter sklearn.metrics.cohen_kappa_score
    """
    # TODO: Implémenter vraie formule
    # from sklearn.metrics import cohen_kappa_score
    # return cohen_kappa_score(labels_phase1, labels_phase2)
    
    return 0.5  # Stub fixe


def compute_dtw_stub(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Stub DTW distance (concordance trajectoires temporelles).
    
    Args:
        trajectory1 : Trajectoire phase 1 (T, n_features)
        trajectory2 : Trajectoire phase 2 (T, n_features)
    
    Returns:
        Distance DTW (plus petit = plus similaire)
    
    Notes:
        - STUB : retourne euclidean simple
        - FULL : implémenter dtaidistance ou fastdtw
    """
    # TODO: Implémenter vraie DTW
    # from dtaidistance import dtw
    # return dtw.distance(trajectory1.flatten(), trajectory2.flatten())
    
    # Stub : distance euclidienne moyenne
    return float(np.linalg.norm(trajectory1 - trajectory2))


def run_concordance_cross_phases(
    phases_data: Dict[str, List[Dict]]
) -> Dict:
    """
    Analyse concordance entre phases.
    
    Args:
        phases_data : {
            'r0': [rows...],
            'r1': [rows...],
            ...
        }
    
    Returns:
        {
            'kappa': {
                'r0_r1': 0.5,
                'r1_r2': 0.7,
                ...
            },
            'trajectories': {
                'r0_r1': {'dtw_distance': 12.3, 'n_samples': 100},
                ...
            },
            'n_phases': int,
        }
    
    Notes:
        - STUB LITE : mécanisme validé, calculs simplifiés
        - FULL : kappa régimes, DTW timelines, trajectoires complètes
    """
    phase_names = sorted(phases_data.keys())
    n_phases = len(phase_names)
    
    if n_phases < 2:
        return {
            'kappa': {},
            'trajectories': {},
            'n_phases': n_phases,
            'warning': 'Besoin ≥2 phases pour concordance',
        }
    
    result = {
        'kappa': {},
        'trajectories': {},
        'n_phases': n_phases,
    }
    
    # Pairwise concordance
    for i in range(len(phase_names) - 1):
        phase1 = phase_names[i]
        phase2 = phase_names[i + 1]
        
        rows1 = phases_data[phase1]
        rows2 = phases_data[phase2]
        
        # Stub kappa (régimes fictifs)
        kappa = compute_kappa_stub(
            ['regime_a'] * len(rows1),
            ['regime_a'] * len(rows2)
        )
        
        result['kappa'][f'{phase1}_{phase2}'] = kappa
        
        # Stub DTW (trajectoires fictives)
        traj1 = np.random.randn(10, 5)  # Stub
        traj2 = np.random.randn(10, 5)  # Stub
        dtw_dist = compute_dtw_stub(traj1, traj2)
        
        result['trajectories'][f'{phase1}_{phase2}'] = {
            'dtw_distance': dtw_dist,
            'n_samples': min(len(rows1), len(rows2)),
        }
    
    return result
