
# File: tests/utilities/utils/convergence_evaluator.py

"""
Convergence Evaluator R1.3 - Évaluation réduction espace candidat.

RESPONSABILITÉ:
- Compilation contraintes cumulatives R0 + R1
- Scoring admissibilité gammas
- Détection structure convergente (unique, famille, classes)

CONFORMITÉ:
- Charter 6.1 Section 5 (UTIL, pas HUB)
- Délégation stricte (pas calculs inline)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


# =============================================================================
# COMPILATION CONTRAINTES
# =============================================================================

def compile_constraints_r0(
    gamma_stability_stats: pd.DataFrame
) -> Dict:
    """
    Compile contraintes R0.
    
    CONTRAINTES R0:
    1. Explosion rate (≤ 10% pour inclusion R1.1)
    2. Nombre encodings affectés
    3. Combinaisons explosives (26 totales)
    
    Args:
        gamma_stability_stats: DataFrame gamma_stability_stats_r0.csv
    
    Returns:
        {
            'gamma_id': {
                'explosion_rate': float,
                'n_encodings_affected': int,
                'recommendation': 'INCLUDE' | 'EXCLUDE'
            }
        }
    """
    constraints = {}
    
    for _, row in gamma_stability_stats.iterrows():
        gamma_id = row['gamma_id']
        
        constraints[gamma_id] = {
            'explosion_rate': row['explosion_rate'],
            'n_encodings_affected': row['n_encodings_affected'],
            'recommendation': row['recommendation']
        }
    
    return constraints


def compile_constraints_r1_1(
    classification_sequences: Dict
) -> Dict:
    """
    Compile contraintes R1.1 (composition robuste).
    
    CONTRAINTES R1.1:
    - Nombre séquences robustes incluant gamma
    - Nombre séquences dégradantes incluant gamma
    - Nombre séquences instables incluant gamma
    
    NOTE: Nécessite extraction gammas depuis séquences classifiées
    
    Args:
        classification_sequences: classification_sequences_r1.json
    
    Returns:
        {
            'gamma_id': {
                'n_sequences_robustes': int,
                'n_sequences_degradantes': int,
                'n_sequences_instables': int,
                'robustness_rate': float
            }
        }
    """
    from collections import Counter
    
    # Extraire gammas depuis séquences (nécessite DB query)
    # Pour MVP, utiliser sequences_robustes directement
    
    # TODO: Implémenter extraction gammas depuis DB
    # Placeholder: Retourner structure minimale
    
    constraints = {}
    
    # NOTE: Implémentation complète nécessite:
    # 1. Query DB: SELECT sequence_gammas FROM sequences WHERE sequence_exec_id IN (...)
    # 2. Parse JSON sequence_gammas
    # 3. Compter occurrences par gamma
    
    return constraints  # Placeholder


def compile_constraints_r1_2(
    compensation_analysis: Dict
) -> Dict:
    """
    Compile contraintes R1.2 (compensation).
    
    CONTRAINTES R1.2:
    - Gamma compensable (présent dans compensable_explosive_gammas)
    - Gamma encadrant efficace (présent dans effective_framing_gammas)
    
    Args:
        compensation_analysis: compensation_analysis.json
    
    Returns:
        {
            'gamma_id': {
                'is_compensable': bool,
                'is_effective_framing': bool
            }
        }
    """
    constraints = {}
    
    compensable = set(compensation_analysis.get('compensable_explosive_gammas', []))
    effective_framing = set(compensation_analysis.get('effective_framing_gammas', []))
    
    all_gammas = compensable | effective_framing
    
    for gamma_id in all_gammas:
        constraints[gamma_id] = {
            'is_compensable': gamma_id in compensable,
            'is_effective_framing': gamma_id in effective_framing
        }
    
    return constraints


def compile_all_constraints(
    constraints_r0: Dict,
    constraints_r1_1: Dict,
    constraints_r1_2: Dict = None
) -> Dict:
    """
    Fusionne toutes contraintes.
    
    Args:
        constraints_r0: Retour compile_constraints_r0()
        constraints_r1_1: Retour compile_constraints_r1_1()
        constraints_r1_2: Retour compile_constraints_r1_2() (optionnel)
    
    Returns:
        {
            'gamma_id': {
                'r0': {...},
                'r1_1': {...},
                'r1_2': {...}  # Si disponible
            }
        }
    """
    all_gammas = set(constraints_r0.keys()) | set(constraints_r1_1.keys())
    
    if constraints_r1_2:
        all_gammas |= set(constraints_r1_2.keys())
    
    compiled = {}
    
    for gamma_id in all_gammas:
        compiled[gamma_id] = {
            'r0': constraints_r0.get(gamma_id, {}),
            'r1_1': constraints_r1_1.get(gamma_id, {}),
        }
        
        if constraints_r1_2:
            compiled[gamma_id]['r1_2'] = constraints_r1_2.get(gamma_id, {})
    
    return compiled


# =============================================================================
# SCORING ADMISSIBILITÉ
# =============================================================================

def compute_admissibility_score(
    gamma_id: str,
    constraints: Dict,
    weights: Dict = None
) -> float:
    """
    Calcule score admissibilité gamma.
    
    COMPOSANTES (poids par défaut):
    1. R0 explosion_rate: -10 × explosion_rate (pénalité)
    2. R0 recommendation: +5 si INCLUDE, -5 si EXCLUDE
    3. R1.1 robustness_rate: +10 × robustness_rate (bonus)
    4. R1.2 compensable: +3 (bonus si compensable)
    5. R1.2 effective_framing: +3 (bonus si encadrant efficace)
    
    Score normalisé: [0, 1] (0 = inadmissible, 1 = optimal)
    
    Args:
        gamma_id: ID gamma
        constraints: Contraintes compilées pour ce gamma
        weights: Poids custom (optionnel)
    
    Returns:
        Score admissibilité [0, 1]
    """
    if weights is None:
        weights = {
            'r0_explosion_penalty': -10.0,
            'r0_recommendation': 5.0,
            'r1_1_robustness': 10.0,
            'r1_2_compensable': 3.0,
            'r1_2_effective_framing': 3.0
        }
    
    score = 0.0
    
    # R0 contraintes
    r0 = constraints.get('r0', {})
    
    explosion_rate = r0.get('explosion_rate', 0.0)
    score += weights['r0_explosion_penalty'] * explosion_rate
    
    recommendation = r0.get('recommendation', 'EXCLUDE')
    if recommendation == 'INCLUDE':
        score += weights['r0_recommendation']
    else:
        score -= weights['r0_recommendation']
    
    # R1.1 contraintes
    r1_1 = constraints.get('r1_1', {})
    
    robustness_rate = r1_1.get('robustness_rate', 0.0)
    score += weights['r1_1_robustness'] * robustness_rate
    
    # R1.2 contraintes (si disponibles)
    r1_2 = constraints.get('r1_2', {})
    
    if r1_2:
        if r1_2.get('is_compensable', False):
            score += weights['r1_2_compensable']
        
        if r1_2.get('is_effective_framing', False):
            score += weights['r1_2_effective_framing']
    
    # Normaliser [0, 1]
    # Max théorique: +5 (INCLUDE) + 10 (robustness=1.0) + 3 + 3 = 21
    # Min théorique: -10 (explosion=1.0) - 5 (EXCLUDE) + 0 = -15
    # Range: 36
    
    score_normalized = (score - (-15)) / 36.0
    score_normalized = np.clip(score_normalized, 0.0, 1.0)
    
    return score_normalized


def score_all_gammas(
    compiled_constraints: Dict,
    weights: Dict = None
) -> Dict:
    """
    Calcule scores admissibilité tous gammas.
    
    Args:
        compiled_constraints: Retour compile_all_constraints()
        weights: Poids custom (optionnel)
    
    Returns:
        {
            'gamma_id': {
                'score': float,
                'rank': int,
                'constraints': {...}
            }
        }
    """
    scores = {}
    
    for gamma_id, constraints in compiled_constraints.items():
        score = compute_admissibility_score(gamma_id, constraints, weights)
        
        scores[gamma_id] = {
            'score': score,
            'constraints': constraints
        }
    
    # Ranking
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for rank, (gamma_id, data) in enumerate(ranked, 1):
        scores[gamma_id]['rank'] = rank
    
    return scores


# =============================================================================
# RÉDUCTION ESPACE
# =============================================================================

def filter_survivors(
    scores: Dict,
    threshold: float = 0.5
) -> Tuple[List[str], Dict]:
    """
    Filtre gammas survivants (score > threshold).
    
    Args:
        scores: Retour score_all_gammas()
        threshold: Seuil admissibilité (0.5 par défaut)
    
    Returns:
        (gamma_ids_survivors, stats)
        stats = {
            'n_candidates_initial': int,
            'n_survivors': int,
            'reduction_factor': float,
            'threshold_used': float
        }
    """
    n_initial = len(scores)
    
    survivors = [
        gamma_id for gamma_id, data in scores.items()
        if data['score'] > threshold
    ]
    
    n_survivors = len(survivors)
    reduction_factor = n_survivors / n_initial if n_initial > 0 else 0.0
    
    stats = {
        'n_candidates_initial': n_initial,
        'n_survivors': n_survivors,
        'reduction_factor': reduction_factor,
        'threshold_used': threshold
    }
    
    return survivors, stats


# =============================================================================
# DÉTECTION STRUCTURE
# =============================================================================

def detect_convergent_structure(
    survivors: List[str],
    gamma_metadata: Dict
) -> Tuple[str, str]:
    """
    Détecte structure convergente (unique, famille, classes).
    
    CLASSIFICATION:
    - UNIQUE: 1 gamma survivant
    - TIGHT_CLASS: 2-3 gammas, même famille (ex: tous markovian)
    - PARAMETRIC_FAMILY: 2-5 gammas, variations paramétriques détectables
    - TOPOLOGICAL_FAMILY: 2-5 gammas, dépendance topologie (d_applicability différentes)
    - CORE_EXTENSIONS: ≥2 gammas, noyau commun + extensions identifiables
    - MULTIPLE_UNCORRELATED: >5 gammas, aucune structure évidente
    - NONE: Aucun survivant (incompatibilité)
    
    Args:
        survivors: Liste gamma_ids survivants
        gamma_metadata: Métadonnées gammas (METADATA dicts)
    
    Returns:
        (structure_type, description)
    """
    n = len(survivors)
    
    if n == 0:
        return 'NONE', 'Aucun gamma survivant (incompatibilité contraintes)'
    
    if n == 1:
        return 'UNIQUE', f'Gamma unique: {survivors[0]}'
    
    if n <= 3:
        # Vérifier même famille
        families = [gamma_metadata.get(gid, {}).get('family', 'unknown') for gid in survivors]
        
        if len(set(families)) == 1 and families[0] != 'unknown':
            return 'TIGHT_CLASS', f"Classe étroite ({n} gammas, famille {families[0]})"
        else:
            return 'PARAMETRIC_FAMILY', f"Famille paramétrée ({n} gammas, familles mixtes)"
    
    if n <= 5:
        # Vérifier dépendance topologie
        d_applicabilities = [
            set(gamma_metadata.get(gid, {}).get('d_applicability', []))
            for gid in survivors
        ]
        
        if len(set(tuple(sorted(d)) for d in d_applicabilities)) > 1:
            return 'TOPOLOGICAL_FAMILY', f"Famille topologique ({n} gammas, d_applicability différentes)"
        else:
            return 'PARAMETRIC_FAMILY', f"Famille paramétrée ({n} gammas)"
    
    # n > 5
    return 'MULTIPLE_UNCORRELATED', f"Multiple non corrélés ({n} gammas, pas de structure évidente)"


def identify_common_properties(
    survivors: List[str],
    gamma_metadata: Dict
) -> Dict:
    """
    Identifie propriétés communes gammas survivants.
    
    PROPRIÉTÉS:
    - Familles représentées
    - d_applicability communes
    - Paramètres typiques (si paramétriques)
    
    Args:
        survivors: Liste gamma_ids survivants
        gamma_metadata: Métadonnées gammas
    
    Returns:
        {
            'families': List[str],
            'd_applicability_common': List[str],
            'parameters_ranges': Dict  # Si paramétrique
        }
    """
    families = []
    d_applicabilities = []
    
    for gamma_id in survivors:
        metadata = gamma_metadata.get(gamma_id, {})
        
        family = metadata.get('family', 'unknown')
        families.append(family)
        
        d_app = metadata.get('d_applicability', [])
        d_applicabilities.append(set(d_app))
    
    # Familles uniques
    families_unique = list(set(families))
    
    # d_applicability commune (intersection)
    if d_applicabilities:
        d_app_common = set.intersection(*d_applicabilities)
    else:
        d_app_common = set()
    
    return {
        'families': families_unique,
        'd_applicability_common': list(d_app_common),
        'n_families': len(families_unique),
        'parameters_ranges': {}  # TODO: Implémenter si paramétriques détectés
    }