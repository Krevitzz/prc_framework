# tests/utilities/scoring.py
"""
Scoring pathologies Charter 5.4.

Principe : Scores [0,1] détectent pathologies, pas qualités.
- 0 = aucun signal pathologique (sain)
- 1 = pathologie maximale (toxique)
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from .config_loader import get_loader


# =============================================================================
# TYPES PATHOLOGIES
# =============================================================================

def score_s1_collapse(value: float, rule: dict) -> Tuple[float, bool]:
    """
    S1 COLLAPSE : Valeur trop faible indique effondrement structural.
    
    Args:
        value: Valeur observée
        rule: {threshold_low, critical_low, mode, weight}
    
    Returns:
        (score [0,1], flag)
    """
    threshold_low = rule['threshold_low']
    critical_low = rule.get('critical_low', threshold_low * 0.1)
    mode = rule.get('mode', 'soft')
    
    if mode == 'hard':
        score = 1.0 if value < threshold_low else 0.0
        flag = value < critical_low
    else:  # soft
        if value >= threshold_low:
            score = 0.0
            flag = False
        elif value <= critical_low:
            score = 1.0
            flag = True
        else:
            score = (threshold_low - value) / (threshold_low - critical_low)
            flag = False
    
    return float(score), flag


def score_s2_explosion(value: float, rule: dict) -> Tuple[float, bool]:
    """
    S2 EXPLOSION : Valeur trop élevée indique divergence non contrôlée.
    
    Args:
        value: Valeur observée
        rule: {threshold_high, critical_high, mode, weight}
    
    Returns:
        (score [0,1], flag)
    """
    threshold_high = rule['threshold_high']
    critical_high = rule.get('critical_high', threshold_high * 10)
    mode = rule.get('mode', 'soft')
    
    if mode == 'hard':
        score = 1.0 if value > threshold_high else 0.0
        flag = value > critical_high
    else:  # soft
        if value <= threshold_high:
            score = 0.0
            flag = False
        elif value >= critical_high:
            score = 1.0
            flag = True
        else:
            score = (value - threshold_high) / (critical_high - threshold_high)
            flag = False
    
    return float(score), flag


def score_s3_plateau(value: float, rule: dict) -> Tuple[float, bool]:
    """
    S3 PLATEAU : Valeur dans intervalle toxique (trop stable/uniforme).
    
    Args:
        value: Valeur observée
        rule: {interval_toxic: [low, high], mode, weight}
    
    Returns:
        (score [0,1], flag)
    """
    interval = rule['interval_toxic']
    interval_low, interval_high = interval
    
    if interval_low <= value <= interval_high:
        score = 1.0
        flag = True
    else:
        dist = min(abs(value - interval_low), abs(value - interval_high))
        interval_width = interval_high - interval_low
        score = max(0, 1 - dist / max(interval_width, 0.1))
        flag = False
    
    return float(score), flag


def score_s4_instability(
    value: float,
    rule: dict,
    context: dict
) -> Tuple[float, bool]:
    """
    S4 INSTABILITY : Variation différentielle trop rapide.
    
    Args:
        value: Valeur actuelle
        rule: {delta_critical, weight}
        context: {value_prev} pour calculer delta
    
    Returns:
        (score [0,1], flag)
    """
    delta_critical = rule['delta_critical']
    value_prev = context.get('value_prev', value)
    
    delta = abs(value - value_prev)
    score = max(0, min(1, delta / delta_critical))
    flag = delta > delta_critical
    
    return float(score), flag


def score_mapping(
    value: Union[float, str],
    rule: dict
) -> Tuple[float, bool]:
    """
    MAPPING : Transitions qualitatives catégorielles.
    
    Args:
        value: Valeur catégorielle (str) ou numérique
        rule: {mapping: {value: score}, weight}
    
    Returns:
        (score [0,1], flag)
    """
    mapping = rule['mapping']
    score = mapping.get(value, 0.5)  # Default 0.5 si non mappé
    flag = score >= 0.8
    
    return float(score), flag


# =============================================================================
# EXTRACTION VALEURS
# =============================================================================

def extract_value_from_observation(
    observation: dict,
    metric_key: str,
    source: str
) -> Union[float, str]:
    """
    Extrait valeur depuis observation selon path source.
    
    Args:
        observation: Dict observation depuis test_engine
        metric_key: Nom métrique (ex: "frobenius_norm")
        source: Path extraction (ex: "statistics.frobenius_norm.final")
    
    Returns:
        Valeur extraite
    
    Raises:
        KeyError: Si path invalide
        ValueError: Si source mal formée
    
    Examples:
        >>> extract_value_from_observation(
        ...     obs,
        ...     'frobenius_norm',
        ...     'statistics.frobenius_norm.final'
        ... )
        1250.5
    """
    parts = source.split('.')
    
    if parts[0] not in ['statistics', 'evolution']:
        raise ValueError(
            f"Source doit commencer par 'statistics' ou 'evolution', reçu: {source}"
        )
    
    current = observation
    for part in parts:
        if part not in current:
            raise KeyError(
                f"Path '{source}' invalide : '{part}' non trouvé dans observation"
            )
        current = current[part]
    
    return current


# =============================================================================
# SCORING MÉTRIQUE
# =============================================================================

def score_metric(
    metric_key: str,
    observation: dict,
    scoring_rule: dict,
    context: dict = None
) -> dict:
    """
    Calcule pathology_score pour une métrique.
    
    Args:
        metric_key: Nom métrique (ex: "frobenius_norm")
        observation: Dict observation depuis test_engine
        scoring_rule: Config YAML règle scoring
        context: Contexte optionnel (historique)
    
    Returns:
        {
            'metric_key': str,
            'value': float | str,
            'score': float [0,1],
            'flag': bool,
            'pathology_type': str,
            'weight': float,
            'source': str
        }
    
    Raises:
        KeyError: Si métrique absente
        ValueError: Si pathology_type inconnu
    """
    # Extraction valeur
    source = scoring_rule.get('source', f"statistics.{metric_key}.final")
    value = extract_value_from_observation(observation, metric_key, source)
    
    # Type pathologie
    pathology_type = scoring_rule['pathology_type']
    weight = scoring_rule.get('weight', 1.0)
    
    # Scoring selon type
    if pathology_type == 'S1_COLLAPSE':
        score, flag = score_s1_collapse(value, scoring_rule)
    
    elif pathology_type == 'S2_EXPLOSION':
        score, flag = score_s2_explosion(value, scoring_rule)
    
    elif pathology_type == 'S3_PLATEAU':
        score, flag = score_s3_plateau(value, scoring_rule)
    
    elif pathology_type == 'S4_INSTABILITY':
        score, flag = score_s4_instability(value, scoring_rule, context or {})
    
    elif pathology_type == 'MAPPING':
        score, flag = score_mapping(value, scoring_rule)
    
    else:
        raise ValueError(f"Type pathologie inconnu: {pathology_type}")
    
    return {
        'metric_key': metric_key,
        'value': value,
        'score': score,
        'flag': flag,
        'pathology_type': pathology_type,
        'weight': weight,
        'source': source
    }


# =============================================================================
# AGRÉGATION TEST
# =============================================================================

def aggregate_test_score(
    metric_scores: Dict[str, dict],
    aggregation_mode: str = "max"
) -> dict:
    """
    Agrège scores métriques en score test.
    
    Args:
        metric_scores: {metric_key: score_dict}
        aggregation_mode: "max" | "weighted_mean" | "weighted_max"
    
    Returns:
        {
            'test_score': float [0,1],
            'pathology_flags': list[str],
            'critical_metrics': list[str],
            'aggregation_mode': str,
            'metric_scores': dict
        }
    
    Raises:
        ValueError: Si aggregation_mode inconnu
    """
    if not metric_scores:
        return {
            'test_score': 0.0,
            'pathology_flags': [],
            'critical_metrics': [],
            'aggregation_mode': aggregation_mode,
            'metric_scores': {}
        }
    
    scores = [m['score'] for m in metric_scores.values()]
    weights = [m['weight'] for m in metric_scores.values()]
    
    # Agrégation
    if aggregation_mode == 'max':
        test_score = max(scores)
    
    elif aggregation_mode == 'weighted_mean':
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        test_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    elif aggregation_mode == 'weighted_max':
        max_weight = max(weights)
        weighted_scores = [s * (w / max_weight) for s, w in zip(scores, weights)]
        test_score = max(weighted_scores)
    
    else:
        raise ValueError(f"Mode agrégation inconnu: {aggregation_mode}")
    
    # Flags et métriques critiques
    pathology_flags = [k for k, m in metric_scores.items() if m['flag']]
    critical_metrics = [k for k, m in metric_scores.items() if m['score'] >= 0.8]
    
    return {
        'test_score': float(test_score),
        'pathology_flags': pathology_flags,
        'critical_metrics': critical_metrics,
        'aggregation_mode': aggregation_mode,
        'metric_scores': metric_scores
    }


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def score_observation(
    observation: dict,
    test_module,
    scoring_config_id: str
) -> dict:
    """
    Convertit observation → scores pathologies.
    
    Args:
        observation: Dict v2 depuis test_engine (avec exec_id)
        test_module: Module test importé
        scoring_config_id: ID config scoring
    
    Returns:
        {
            'exec_id': int,
            'test_name': str,
            'config_params_id': str,
            'config_scoring_id': str,
            'test_score': float [0,1],
            'aggregation_mode': str,
            'metric_scores': dict,
            'pathology_flags': list[str],
            'critical_metrics': list[str],
            'test_weight': float
        }
    
    Raises:
        KeyError: Si test_module.TEST_ID absent de config
        ValueError: Si observation invalide
    """
    # Validation
    if observation['status'] not in ['SUCCESS']:
        return {
            'exec_id': observation.get('exec_id'),
            'test_name': test_module.TEST_ID,
            'config_params_id': observation['config_params_id'],
            'config_scoring_id': scoring_config_id,
            'test_score': 0.0,
            'aggregation_mode': 'max',
            'metric_scores': {},
            'pathology_flags': [],
            'critical_metrics': [],
            'test_weight': 1.0,
            'skipped_reason': observation.get('message', 'Non SUCCESS')
        }
    
    # Charger config scoring
    loader = get_loader()
    config = loader.load(
        config_type='scoring',
        config_id=scoring_config_id,
        test_id=test_module.TEST_ID
    )
    
    if test_module.TEST_ID not in config['tests']:
        raise KeyError(
            f"Test {test_module.TEST_ID} absent de config {scoring_config_id}"
        )
    
    test_config = config['tests'][test_module.TEST_ID]
    scoring_rules = test_config.get('scoring_rules', {})
    aggregation_mode = test_config.get('aggregation_mode', 'max')
    
    # Scorer chaque métrique
    metric_scores = {}
    
    for metric_key in test_module.COMPUTATION_SPECS.keys():
        if metric_key not in scoring_rules:
            continue  # Métrique non scorée
        
        try:
            score_dict = score_metric(
                metric_key,
                observation,
                scoring_rules[metric_key]
            )
            metric_scores[metric_key] = score_dict
        
        except Exception as e:
            print(f"[scoring] Erreur métrique {metric_key}: {e}")
            continue
    
    # Agrégation
    aggregated = aggregate_test_score(metric_scores, aggregation_mode)
    
    return {
        'exec_id': observation.get('exec_id'),
        'test_name': test_module.TEST_ID,
        'config_params_id': observation['config_params_id'],
        'config_scoring_id': scoring_config_id,
        'test_score': aggregated['test_score'],
        'aggregation_mode': aggregated['aggregation_mode'],
        'metric_scores': aggregated['metric_scores'],
        'pathology_flags': aggregated['pathology_flags'],
        'critical_metrics': aggregated['critical_metrics'],
        'test_weight': test_config.get('test_weight', 1.0)
    }