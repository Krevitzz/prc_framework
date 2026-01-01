# tests/utilities/verdict_engine.py
from tests.utilities.config_loader import get_loader

def compute_gamma_verdict(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str,
    thresholds_config_id: str
) -> dict:
    """
    Calcule verdict GLOBAL pour une gamma.
    
    Agrège TOUS tests sur TOUS runs.
    
    Returns:
        {
            'gamma_id': str,
            'params_config_id': str,
            'scoring_config_id': str,
            'thresholds_config_id': str,
            'majority_pct': float,
            'robustness_pct': float,
            'global_score': float,
            'verdict': str,
            'verdict_reason': str,
        }
    """
    # Charger thresholds (global uniquement, pas de test_id)
    loader = get_loader()
    thresholds = loader.load(
        config_type='thresholds',
        config_id=thresholds_config_id
    )
    
    # Récupérer critères
    survives_criteria = thresholds['survives']
    flagged_criteria = thresholds['flagged']
    
    # Calculer 3 critères
    majority_pct = calculate_majority(
        gamma_id, params_config_id, scoring_config_id, thresholds
    )
    
    robustness_pct = calculate_robustness(
        gamma_id, params_config_id, scoring_config_id, thresholds
    )
    
    global_score = calculate_global_score(
        gamma_id, params_config_id, scoring_config_id
    )
    
    # Appliquer règles verdict
    verdict, reason = apply_verdict_rules(
        majority_pct, robustness_pct, global_score, thresholds
    )
    
    return {
        'gamma_id': gamma_id,
        'params_config_id': params_config_id,
        'scoring_config_id': scoring_config_id,
        'thresholds_config_id': thresholds_config_id,
        'majority_pct': majority_pct,
        'robustness_pct': robustness_pct,
        'global_score': global_score,
        'verdict': verdict,
        'verdict_reason': reason,
    }
	
def compute_config_score(
    exec_id: int,
    params_config_id: str,
    scoring_config_id: str
) -> float:
    """
    Score 0-1 d'une config = moyenne pondérée de tous ses tests.
    
    Args:
        exec_id: ID run dans db_raw
        params_config_id: Config params utilisée
        scoring_config_id: Config scoring utilisée
    
    Returns:
        float: Score 0-1
    """
    # Charger tous scores tests pour ce run
    test_scores = db.query("""
        SELECT test_name, weighted_average, test_weight
        FROM TestScores
        WHERE exec_id = ? 
          AND config_params_id = ?
          AND config_scoring_id = ?
    """, exec_id, params_config_id, scoring_config_id)
    
    # Moyenne pondérée
    weighted_sum = sum(t['weighted_average'] * t['test_weight'] 
                      for t in test_scores)
    weight_sum = sum(t['test_weight'] for t in test_scores)
    
    config_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    return config_score
	
def calculate_majority(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str,
    thresholds: dict
) -> float:
    """
    % configs dont score ≥ seuil global.
    
    Returns:
        float: Pourcentage 0-100
    """
    threshold = thresholds['global']['score_threshold']
    
    # Charger toutes configs pour cette gamma
    all_exec_ids = db.query("""
        SELECT id FROM Executions WHERE gamma_id = ?
    """, gamma_id)
    
    passing_count = 0
    for exec_id in all_exec_ids:
        config_score = compute_config_score(
            exec_id, params_config_id, scoring_config_id
        )
        if config_score >= threshold:
            passing_count += 1
    
    majority_pct = (passing_count / len(all_exec_ids)) * 100 if all_exec_ids else 0
    
    return majority_pct
	
def calculate_robustness(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str,
    thresholds: dict
) -> float:
    """
    % D distincts ayant ≥1 config viable.
    
    Config viable = score ≥ seuil global.
    
    Returns:
        float: Pourcentage 0-100
    """
    threshold = thresholds['global']['score_threshold']
    
    # Grouper configs par D
    configs_by_d = db.query("""
        SELECT d_base_id, GROUP_CONCAT(id) as exec_ids
        FROM Executions
        WHERE gamma_id = ?
        GROUP BY d_base_id
    """, gamma_id)
    
    viable_d_count = 0
    for row in configs_by_d:
        d_base_id = row['d_base_id']
        exec_ids = row['exec_ids'].split(',')
        
        # Vérifier si ≥1 config de ce D passe
        has_viable_config = False
        for exec_id in exec_ids:
            config_score = compute_config_score(
                int(exec_id), params_config_id, scoring_config_id
            )
            if config_score >= threshold:
                has_viable_config = True
                break
        
        if has_viable_config:
            viable_d_count += 1
    
    total_d = len(configs_by_d)
    robustness_pct = (viable_d_count / total_d) * 100 if total_d > 0 else 0
    
    return robustness_pct

def calculate_global_score(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str
) -> float:
    """
    Moyenne de TOUS les scores configs.
    
    Returns:
        float: Score 0-1
    """
    all_exec_ids = db.query("""
        SELECT id FROM Executions WHERE gamma_id = ?
    """, gamma_id)
    
    scores = []
    for exec_id in all_exec_ids:
        config_score = compute_config_score(
            exec_id, params_config_id, scoring_config_id
        )
        scores.append(config_score)
    
    global_score = np.mean(scores) if scores else 0.0
    
    return global_score
	
# Moyenne préférée à médiane pour ne pas masquer distribution.
ef apply_verdict_rules(
    majority_pct: float,
    robustness_pct: float,
    global_score: float,
    thresholds: dict
) -> Tuple[str, str]:
    """
    Applique logique verdict.
    
    Returns:
        (verdict: str, reason: str)
    """
    criteria_survives = thresholds['global']['survives_criteria']
    criteria_flagged = thresholds['global']['flagged_criteria']
    
    # SURVIVES[R0] - OU logique
    if (global_score >= criteria_survives['score_min'] or
        robustness_pct >= criteria_survives['robustness_min'] * 100 or
        majority_pct >= criteria_survives['majority_min'] * 100):
        
        reasons = []
        if global_score >= criteria_survives['score_min']:
            reasons.append(f"score={global_score:.2f} ≥ {criteria_survives['score_min']}")
        if robustness_pct >= criteria_survives['robustness_min'] * 100:
            reasons.append(f"robustness={robustness_pct:.1f}% ≥ {criteria_survives['robustness_min']*100}%")
        if majority_pct >= criteria_survives['majority_min'] * 100:
            reasons.append(f"majority={majority_pct:.1f}% ≥ {criteria_survives['majority_min']*100}%")
        
        return "SURVIVES[R0]", " OR ".join(reasons)
    
    # FLAGGED_FOR_REVIEW - ET logique
    if (global_score < criteria_flagged['score_max'] and
        robustness_pct < criteria_flagged['robustness_max'] * 100 and
        majority_pct < criteria_flagged['majority_max'] * 100):
        
        reason = (
            f"score={global_score:.2f} < {criteria_flagged['score_max']} AND "
            f"robustness={robustness_pct:.1f}% < {criteria_flagged['robustness_max']*100}% AND "
            f"majority={majority_pct:.1f}% < {criteria_flagged['majority_max']*100}%"
        )
        
        return "FLAGGED_FOR_REVIEW", reason
    
    # Par défaut: WIP[R0-closed]
    reason = (
        f"score={global_score:.2f}, "
        f"robustness={robustness_pct:.1f}%, "
        f"majority={majority_pct:.1f}% - "
        f"ni SURVIVES ni FLAGGED"
    )
    
    return "WIP[R0-closed]", reason
	
# Ces formules sont une baseline. Raffinage futur possible (cf Section 12.7.7).