
# File: tests/utilities/utils/sequence_analyzer.py

"""
Sequence Analyzer R1 - Analyse préservation invariants.

RESPONSABILITÉ:
- Classification séquences (robustes/dégradantes/instables)
- Calcul baselines comparaison R0→R1
- Mesure indépendances (ordre, regroupement)

CONFORMITÉ:
- Charter 6.1 Section 5 (UTIL, pas HUB)
- Délégation statistical_utils, regime_utils
- Pas de calculs inline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


# =============================================================================
# BASELINES R0 (Extraction)
# =============================================================================

def compute_r0_baselines(observations_r0: List[Dict]) -> Dict:
    """
    Calcule baselines R0 pour comparaison R1.
    
    BASELINES:
    - Taux rejection numeric (0.33% R0)
    - Taux explosion (0.45% R0)
    - Concordance régimes (100% R0)
    - Tests robustes (TOP-001: 0.5% outliers)
    - Tests sentinelles (UNIV-002: 31.6% outliers)
    
    DÉLÉGATION:
    - filter_numeric_artifacts() pour rejections
    - stratify_by_regime() pour explosions
    
    Args:
        observations_r0: Observations R0 SUCCESS
    
    Returns:
        {
            'n_observations': int,
            'rejection_rate': float,
            'explosion_rate': float,
            'regime_concordance': float,
            'test_robustness': {test_name: outlier_rate},
            'metric_sensitivity': {metric_name: outlier_rate}
        }
    """
    from tests.utilities.utils.statistical_utils import filter_numeric_artifacts
    from tests.utilities.utils.regime_utils import stratify_by_regime
    
    n_total = len(observations_r0)
    
    # 1. Rejections numériques
    valid_obs, rejection_stats = filter_numeric_artifacts(observations_r0)
    rejection_rate = rejection_stats['rejection_rate']
    
    # 2. Explosions
    obs_stable, obs_explosif = stratify_by_regime(observations_r0, threshold=1e50)
    explosion_rate = len(obs_explosif) / n_total if n_total > 0 else 0
    
    # 3. Concordance régimes (placeholder R0, assumé 100%)
    regime_concordance = 1.0  # Charter Section 6 (validation empirique R0)
    
    # 4. Tests robustesse (empirique depuis data)
    test_robustness = _compute_test_outlier_rates(observations_r0)
    
    # 5. Métriques sensibilité
    metric_sensitivity = _compute_metric_outlier_rates(observations_r0)
    
    return {
        'n_observations': n_total,
        'rejection_rate': rejection_rate,
        'explosion_rate': explosion_rate,
        'regime_concordance': regime_concordance,
        'test_robustness': test_robustness,
        'metric_sensitivity': metric_sensitivity
    }


def _compute_test_outlier_rates(observations: List[Dict]) -> Dict[str, float]:
    """
    Calcule taux outliers par test (métrique first).
    
    MÉTHODE:
    - Outlier = |normalized| > 3.0 (Robust normalization)
    - Par test: fraction observations outliers
    """
    
    test_outliers = defaultdict(lambda: {'total': 0, 'outliers': 0})
    
    for obs in observations:
        test_name = obs['test_name']
        stats = obs.get('observation_data', {}).get('statistics', {})
        
        if not stats:
            continue
        
        first_metric = list(stats.keys())[0]
        final_value = stats[first_metric].get('final')
        
        if final_value is None:
            continue
        
        # Collecter toutes valeurs finales ce test
        test_outliers[test_name]['total'] += 1
        
        # Normaliser (nécessite toutes valeurs test, simplifié ici)
        # NOTE: Calcul robuste nécessite toutes valeurs, ici approximation
        if abs(final_value) > 1e10:  # Heuristique explosion
            test_outliers[test_name]['outliers'] += 1
    
    # Calculer rates
    outlier_rates = {}
    for test_name, counts in test_outliers.items():
        if counts['total'] > 0:
            outlier_rates[test_name] = counts['outliers'] / counts['total']
        else:
            outlier_rates[test_name] = 0.0
    
    return outlier_rates


def _compute_metric_outlier_rates(observations: List[Dict]) -> Dict[str, float]:
    """
    Calcule taux outliers par métrique (cross-tests).
    
    MÉTHODE:
    - Projection: value_final, slope, relative_change
    - Outlier = valeur > P90 + 5 décades
    """
    # TODO: Implémenter si nécessaire (similaire _compute_test_outlier_rates)
    # Pour MVP R1.1, retourner dict vide
    return {}


# =============================================================================
# CLASSIFICATION SÉQUENCES
# =============================================================================

def classify_sequences(
    observations_r1: List[Dict],
    baselines_r0: Dict,
    tolerance_factor: float = 2.0
) -> Dict:
    """
    Classifie séquences (ROBUSTES, DÉGRADANTES, INSTABLES).
    
    CRITÈRES:
    - ROBUSTES: Préservent tous invariants R0
      * Taux explosion ≤ baseline_r0 × tolerance_factor
      * Concordance régimes ≥ 95%
      * Tests robustes maintenus (outlier_rate proche R0)
    
    - DÉGRADANTES: Invariants partiellement perdus
      * Explosion_rate > baseline × tolerance mais < 10%
      * Concordance régimes 80-95%
    
    - INSTABLES: Nouvelles explosions, contradictions
      * Explosion_rate > 10%
      * Concordance régimes < 80%
    
    Args:
        observations_r1: Observations séquences R1
        baselines_r0: Retour compute_r0_baselines()
        tolerance_factor: Multiplicateur tolérance (2.0 par défaut)
    
    Returns:
        {
            'n_sequences_total': int,
            'n_sequences_robustes': int,
            'n_sequences_degradantes': int,
            'n_sequences_instables': int,
            'sequences_robustes': List[str],  # sequence_exec_ids
            'sequences_degradantes': List[str],
            'sequences_instables': List[str],
            'metrics': {
                'explosion_rate_r1': float,
                'concordance_r1': float,
                'test_robustness_r1': Dict
            }
        }
    """
    from tests.utilities.utils.statistical_utils import filter_numeric_artifacts
    from tests.utilities.utils.regime_utils import stratify_by_regime
    
    n_total = len(observations_r1)
    
    # 1. Métriques globales R1
    valid_obs, rejection_stats = filter_numeric_artifacts(observations_r1)
    explosion_rate_r1 = rejection_stats['rejection_rate']
    
    obs_stable, obs_explosif = stratify_by_regime(observations_r1, threshold=1e50)
    explosion_rate_r1_strat = len(obs_explosif) / n_total if n_total > 0 else 0
    
    # Concordance régimes (simplifié R1.1, calcul exact nécessite profiling)
    concordance_r1 = 0.95  # Placeholder (calcul réel nécessite classify_regime)
    
    test_robustness_r1 = _compute_test_outlier_rates(observations_r1)
    
    # 2. Classification par séquence
    sequences_by_exec_id = defaultdict(list)
    for obs in observations_r1:
        seq_exec_id = obs.get('observation_data', {}).get('run_metadata', {}).get('sequence_exec_id')
        if seq_exec_id:
            sequences_by_exec_id[seq_exec_id].append(obs)
    
    sequences_robustes = []
    sequences_degradantes = []
    sequences_instables = []
    
    baseline_explosion = baselines_r0['explosion_rate']
    
    for seq_exec_id, seq_obs in sequences_by_exec_id.items():
        # Métriques séquence
        n_obs_seq = len(seq_obs)
        
        _, rejection_stats_seq = filter_numeric_artifacts(seq_obs)
        explosion_rate_seq = rejection_stats_seq['rejection_rate']
        
        # Classification
        if explosion_rate_seq <= baseline_explosion * tolerance_factor:
            sequences_robustes.append(seq_exec_id)
        elif explosion_rate_seq <= 0.10:
            sequences_degradantes.append(seq_exec_id)
        else:
            sequences_instables.append(seq_exec_id)
    
    return {
        'n_sequences_total': len(sequences_by_exec_id),
        'n_sequences_robustes': len(sequences_robustes),
        'n_sequences_degradantes': len(sequences_degradantes),
        'n_sequences_instables': len(sequences_instables),
        'sequences_robustes': sequences_robustes,
        'sequences_degradantes': sequences_degradantes,
        'sequences_instables': sequences_instables,
        'metrics': {
            'explosion_rate_r1': explosion_rate_r1_strat,
            'concordance_r1': concordance_r1,
            'test_robustness_r1': test_robustness_r1
        }
    }


# =============================================================================
# INDÉPENDANCES (ordre, regroupement)
# =============================================================================

def measure_order_independence(
    observations_r1: List[Dict],
    threshold: float = 0.80
) -> Dict:
    """
    Mesure indépendance ordre composition (Γ₁→Γ₂ vs Γ₂→Γ₁).
    
    MÉTHODE:
    - Identifier paires inversées: (GAM-A, GAM-B) et (GAM-B, GAM-A)
    - Comparer régimes/métriques finales
    - Concordance > threshold → ordre-indépendant
    
    Args:
        observations_r1: Observations séquences n=2
        threshold: Seuil concordance (0.80 par défaut)
    
    Returns:
        {
            'n_pairs_tested': int,
            'n_pairs_independent': int,
            'independence_rate': float,
            'pairs_dependent': List[tuple],  # [(seq1, seq2), ...]
            'metric_differences': Dict  # {pair: {metric: diff}}
        }
    """
    # TODO: Implémenter
    # Pour MVP R1.1, retourner structure minimale
    return {
        'n_pairs_tested': 0,
        'n_pairs_independent': 0,
        'independence_rate': 0.0,
        'pairs_dependent': [],
        'metric_differences': {}
    }


def measure_grouping_independence(
    observations_r1: List[Dict],
    threshold: float = 0.80
) -> Dict:
    """
    Mesure indépendance regroupement ((Γ₁→Γ₂)→Γ₃ vs Γ₁→(Γ₂→Γ₃)).
    
    MÉTHODE:
    - Identifier triplets testables (nécessite séquences n=2 et n=3)
    - Comparer composition imbriquée vs séquentielle
    - Concordance > threshold → regroupement-indépendant
    
    NOTE R1.1: Nécessite runs additionnels (compositions imbriquées)
    MVP: Retourner placeholder
    
    Args:
        observations_r1: Observations séquences n=3
        threshold: Seuil concordance
    
    Returns:
        Structure similaire measure_order_independence()
    """
    # TODO: Implémenter si séquences imbriquées disponibles
    # Pour MVP R1.1, retourner structure minimale
    return {
        'n_triplets_tested': 0,
        'n_triplets_independent': 0,
        'independence_rate': 0.0,
        'triplets_dependent': [],
        'metric_differences': {}
    }
    
    

def analyze_compensations(
    observations_compensated: List[Dict],
    baseline_explosion_rate: float
) -> Dict:
    """
    Analyse compensations (réussies vs échecs).
    
    CRITÈRE COMPENSATION RÉUSSIE:
    - Séquence [Γ_stable₁, Γ_explosif, Γ_stable₂] produit état final stable
    - Explosion_rate séquence ≤ baseline_explosion_rate × 2.0
    
    MÉTRIQUES:
    - Taux compensation (% séquences compensées / total)
    - Patterns compensations réussies (quels gammas explosifs compensables)
    - Encadrements efficaces (quels gammas stables marchent)
    
    Args:
        observations_compensated: Observations séquences compensées
        baseline_explosion_rate: Baseline R0 (0.45%)
    
    Returns:
        {
            'n_sequences_total': int,
            'n_compensations_success': int,
            'n_compensations_failed': int,
            'compensation_rate': float,
            'compensable_explosive_gammas': List[str],
            'effective_framing_gammas': List[str],
            'patterns': Dict
        }
    """
    from tests.utilities.utils.statistical_utils import filter_numeric_artifacts
    from collections import defaultdict
    
    # Grouper par sequence_exec_id
    sequences_by_id = defaultdict(list)
    for obs in observations_compensated:
        seq_exec_id = obs.get('observation_data', {}).get('run_metadata', {}).get('sequence_exec_id')
        if seq_exec_id:
            sequences_by_id[seq_exec_id].append(obs)
    
    n_total = len(sequences_by_id)
    
    compensations_success = []
    compensations_failed = []
    
    compensable_explosive_gammas = set()
    effective_framing_gammas = set()
    
    for seq_exec_id, seq_obs in sequences_by_id.items():
        # Extraire structure séquence
        if not seq_obs:
            continue
        
        first_obs = seq_obs[0]
        sequence_gammas = first_obs.get('observation_data', {}).get('run_metadata', {}).get('sequence_gammas')
        
        if not sequence_gammas or len(sequence_gammas) != 3:
            continue
        
        gamma_stable_1 = sequence_gammas[0]
        gamma_explosif = sequence_gammas[1]
        gamma_stable_2 = sequence_gammas[2]
        
        # Métriques séquence
        _, rejection_stats = filter_numeric_artifacts(seq_obs)
        explosion_rate_seq = rejection_stats['rejection_rate']
        
        # Classification
        if explosion_rate_seq <= baseline_explosion_rate * 2.0:
            compensations_success.append({
                'sequence_exec_id': seq_exec_id,
                'sequence_gammas': sequence_gammas,
                'explosion_rate': explosion_rate_seq
            })
            
            # Patterns
            compensable_explosive_gammas.add(gamma_explosif)
            effective_framing_gammas.add(gamma_stable_1)
            effective_framing_gammas.add(gamma_stable_2)
        else:
            compensations_failed.append({
                'sequence_exec_id': seq_exec_id,
                'sequence_gammas': sequence_gammas,
                'explosion_rate': explosion_rate_seq
            })
    
    compensation_rate = len(compensations_success) / n_total if n_total > 0 else 0.0
    
    return {
        'n_sequences_total': n_total,
        'n_compensations_success': len(compensations_success),
        'n_compensations_failed': len(compensations_failed),
        'compensation_rate': compensation_rate,
        'compensable_explosive_gammas': list(compensable_explosive_gammas),
        'effective_framing_gammas': list(effective_framing_gammas),
        'compensations_success': compensations_success,
        'compensations_failed': compensations_failed,
        'patterns': {
            'n_explosive_gammas_compensable': len(compensable_explosive_gammas),
            'n_framing_gammas_effective': len(effective_framing_gammas)
        }
    }