# tests/utilities/scoring.py
"""
Conversion observations → scores pathologies R0.

Architecture Charter 5.4 - Section 12.8
"""

import json
from typing import Dict, Any
from tests.utilities.config_loader import get_loader
from tests.utilities.scoring_engine import get_scoring_engine


def score_observation(
    observation: dict,
    test_module,
    context: dict,
    params_config_id: str,
    scoring_config_id: str
) -> dict:
    """
    Convertit observation → scores pathologies R0.
    
    Args:
        observation: Dict v2 depuis test_engine.execute_test()
        test_module: Module test importé
        context: {d_base_id, gamma_id, ...}
        params_config_id: ID config params utilisée
        scoring_config_id: ID config scoring
    
    Returns:
        {
            'test_name': str,
            'params_config_id': str,
            'scoring_config_id': str,
            'test_score': float [0,1],
            'aggregation_mode': str,
            'test_weight': float,
            'metric_scores': dict,
            'pathology_flags': list[str],
            'critical_metrics': list[str]
        }
    """
    # 1. Charger config scoring
    loader = get_loader()
    scoring_config = loader.load(
        config_type='scoring',
        config_id=scoring_config_id,
        test_id=test_module.TEST_ID
    )
    
    # Vérifier que test est dans config
    if test_module.TEST_ID not in scoring_config['tests']:
        raise ValueError(
            f"Test {test_module.TEST_ID} not found in scoring config {scoring_config_id}. "
            f"Available tests: {list(scoring_config['tests'].keys())}"
        )
    
    test_config = scoring_config['tests'][test_module.TEST_ID]
    scoring_rules = test_config['scoring_rules']
    
    # Récupérer poids test
    test_weight = scoring_config['test_weights'].get(test_module.TEST_ID, 1.0)
    
    # Récupérer mode agrégation
    aggregation_mode = scoring_config.get('aggregation_mode', 'max')
    
    # 2. Vérifier que observation est SUCCESS
    if observation['status'] != 'SUCCESS':
        # Si NOT_APPLICABLE ou ERROR, retour avec score 0 (sain par défaut)
        return {
            'test_name': test_module.TEST_ID,
            'params_config_id': params_config_id,
            'scoring_config_id': scoring_config_id,
            'test_score': 0.0,
            'aggregation_mode': aggregation_mode,
            'test_weight': test_weight,
            'metric_scores': {},
            'pathology_flags': [],
            'critical_metrics': [],
            'status': observation['status'],
            'message': f"Observation status: {observation['status']}"
        }
    
    # 3. Scorer chaque métrique
    engine = get_scoring_engine()
    metric_scores = {}
    
    statistics = observation.get('statistics', {})
    evolution = observation.get('evolution', {})
    
    for metric_key, rule in scoring_rules.items():
        # Déterminer source métrique (statistics ou evolution)
        if metric_key in statistics:
            # Prendre valeur depuis statistics (ex: stat_final)
            metric_source = statistics[metric_key]
            
            # Extraire valeur pertinente
            if isinstance(metric_source, dict):
                # statistics contient des dicts {initial, final, min, max, ...}
                # Identifier quelle clé utiliser
                if metric_key.startswith('stat_initial'):
                    metric_value = metric_source.get('initial')
                elif metric_key.startswith('stat_final'):
                    metric_value = metric_source.get('final')
                elif metric_key.startswith('stat_min'):
                    metric_value = metric_source.get('min')
                elif metric_key.startswith('stat_max'):
                    metric_value = metric_source.get('max')
                elif metric_key.startswith('stat_mean'):
                    metric_value = metric_source.get('mean')
                elif metric_key.startswith('stat_std'):
                    metric_value = metric_source.get('std')
                elif metric_key.startswith('stat_median'):
                    metric_value = metric_source.get('median')
                else:
                    # Par défaut, chercher 'final'
                    metric_value = metric_source.get('final')
            else:
                metric_value = metric_source
        
        elif metric_key in evolution:
            # Prendre valeur depuis evolution
            metric_source = evolution[metric_key]
            
            if isinstance(metric_source, dict):
                # evolution contient des dicts {transition, trend, slope, ...}
                if metric_key.startswith('evolution_transition'):
                    metric_value = metric_source.get('transition')
                elif metric_key.startswith('evolution_trend'):
                    metric_value = metric_source.get('trend')
                elif metric_key.startswith('evolution_slope'):
                    metric_value = metric_source.get('slope')
                elif metric_key.startswith('evolution_volatility'):
                    metric_value = metric_source.get('volatility')
                elif metric_key.startswith('evolution_trend_coefficient'):
                    metric_value = metric_source.get('slope')  # Alias
                else:
                    # Par défaut, chercher 'transition'
                    metric_value = metric_source.get('transition')
            else:
                metric_value = metric_source
        
        else:
            # Métrique non trouvée
            print(f"[WARNING] Métrique {metric_key} non trouvée dans observation")
            continue
        
        if metric_value is None:
            print(f"[WARNING] Métrique {metric_key} value is None, skipping")
            continue
        
        # Contexte pour S4_INSTABILITY (si besoin delta)
        # TODO: Implémenter si nécessaire (nécessite historique métrique)
        context_metric = {}
        
        # Scorer métrique
        try:
            scored = engine.score_metric(
                metric_key=metric_key,
                metric_value=metric_value,
                scoring_rule=rule,
                context=context_metric
            )
            
            metric_scores[metric_key] = scored
        
        except Exception as e:
            print(f"[ERROR] Scoring métrique {metric_key}: {e}")
            continue
    
    # 4. Agréger métriques → test_score
    aggregated = engine.aggregate_test_score(
        metric_scores=metric_scores,
        aggregation_mode=aggregation_mode
    )
    
    return {
        'test_name': test_module.TEST_ID,
        'params_config_id': params_config_id,
        'scoring_config_id': scoring_config_id,
        'test_score': aggregated['test_score'],
        'aggregation_mode': aggregated['aggregation_mode'],
        'test_weight': test_weight,
        'metric_scores': aggregated['metric_scores'],
        'pathology_flags': aggregated['pathology_flags'],
        'critical_metrics': aggregated['critical_metrics']
    }