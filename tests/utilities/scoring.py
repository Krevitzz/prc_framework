# tests/utilities/scoring.py

def score_observation(
    observation: dict,
    test_spec,
    context: dict,
    scoring_config_id: str
) -> dict:
    """
    Convertit observation → scores 0-1.
    
    Args:
        observation: Dict v2 depuis run_observation()
        test_spec: TestSpecification instance
        context: {d_base_id, gamma_id, ...}
        scoring_config_id: ID config scoring
    
    Returns:
        {
            'test_name': str,
            'config_params_id': str,
            'config_scoring_id': str,
            'test_weight': float,
            'metric_scores': {
                metric_key: {
                    'value': float,
                    'score': float,  # 0-1
                    'weight': float,
                    'skipped': bool,
                }
            },
            'weighted_average': float,  # 0-1
        }
    """
    # Charger config scoring
    config = load_scoring_config(test_spec.TEST_ID, scoring_config_id)
    
    # Scorer chaque métrique
    metric_scores = {}
    for metric_key in test_spec.SCORING_SPEC['available_metrics']:
        # Skip si métrique SKIPPED ou NOT_APPLICABLE
        if observation['statistics'].get(metric_key) is None:
            continue
        
        # Vérifier si métrique skipped
        if observation['evolution'].get(metric_key) == 'skipped':
            metric_scores[metric_key] = {
                'value': 0.0,
                'score': 0.0,
                'weight': 0.0,
                'skipped': True,
            }
            continue
        
        # Récupérer valeur
        value = observation['statistics'].get(metric_key)
        if value is None:
            value = observation['evolution'].get(metric_key)
        
        # Appliquer règle scoring
        scoring_rule = config['scoring_rules'].get(metric_key)
        if scoring_rule is None:
            # Pas de règle définie → skip
            continue
        
        score = apply_scoring_rule(value, scoring_rule)
        weight = config['metric_weights'].get(metric_key, 1.0)
        
        metric_scores[metric_key] = {
            'value': value,
            'score': score,
            'weight': weight,
            'skipped': False,
        }
    
    # Calculer moyenne pondérée
    weighted_sum = sum(m['score'] * m['weight'] 
                      for m in metric_scores.values() 
                      if not m['skipped'])
    weight_sum = sum(m['weight'] 
                    for m in metric_scores.values() 
                    if not m['skipped'])
    
    weighted_average = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    return {
        'test_name': test_spec.TEST_ID,
        'config_params_id': observation['config_params_id'],
        'config_scoring_id': scoring_config_id,
        'test_weight': config['test_weight'],
        'metric_scores': metric_scores,
        'weighted_average': weighted_average,
    }


def apply_scoring_rule(value, rule: dict) -> float:
    """
    Applique règle de notation.
    
    Args:
        value: Valeur observée (float ou str)
        rule: Dict avec type règle
    
    Returns:
        Score 0-1
    """
    if 'ranges' in rule:
        # Règle par tranches
        for range_spec in rule['ranges']:
            min_val = range_spec.get('min', float('-inf'))
            max_val = range_spec.get('max', float('inf'))
            if min_val <= value < max_val:
                return range_spec['score']
        return 0.0  # Hors tranches
    
    elif 'linear' in rule:
        # Interpolation linéaire
        linear = rule['linear']
        min_val = linear['min_val']
        max_val = linear['max_val']
        min_score = linear['min_score']
        max_score = linear['max_score']
        
        if value <= min_val:
            return min_score
        elif value >= max_val:
            return max_score
        else:
            # Interpolation
            t = (value - min_val) / (max_val - min_val)
            return min_score + t * (max_score - min_score)
    
    elif 'mapping' in rule:
        # Mapping direct
        return rule['mapping'].get(value, 0.0)
    
    else:
        raise ValueError(f"Unknown scoring rule type: {rule}")