# tests/utilities/verdict_engine.py
"""
Moteur de verdict global R0.

Architecture Charter 5.4 - Section 12.9
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from tests.utilities.config_loader import get_loader


def compute_gamma_verdict(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str,
    thresholds_config_id: str,
    db_connection  # Connection to db_results
) -> dict:
    """
    Calcule verdict global pour une gamma.
    
    Agrège TOUS test_scores sur TOUS runs.
    
    Args:
        gamma_id: ID gamma (ex: "GAM-001")
        params_config_id: ID config params
        scoring_config_id: ID config scoring
        thresholds_config_id: ID config thresholds
        db_connection: Connection db_results
    
    Returns:
        {
            'gamma_id': str,
            'params_config_id': str,
            'scoring_config_id': str,
            'thresholds_config_id': str,
            'majority_pct': float,
            'robustness_pct': float,
            'global_score': float [0,1],
            'verdict': str,
            'verdict_reason': str,
            'details': dict
        }
    """
    # 1. Charger config thresholds
    loader = get_loader()
    thresholds = loader.load('thresholds', thresholds_config_id)
    
    threshold_test = thresholds['threshold_test']
    survives_criteria = thresholds['survives']
    rejected_criteria = thresholds['rejected']
    
    # 2. Récupérer tous test_scores pour cette gamma
    cursor = db_connection.cursor()
    
    cursor.execute("""
        SELECT 
            ts.exec_id,
            ts.test_name,
            ts.test_score,
            ts.test_weight,
            ts.pathology_flags,
            ts.critical_metrics,
            e.d_base_id,
            e.modifier_id,
            e.seed
        FROM TestScores ts
        JOIN Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ?
          AND ts.params_config_id = ?
          AND ts.scoring_config_id = ?
    """, (gamma_id, params_config_id, scoring_config_id))
    
    rows = cursor.fetchall()
    
    if not rows:
        return {
            'gamma_id': gamma_id,
            'params_config_id': params_config_id,
            'scoring_config_id': scoring_config_id,
            'thresholds_config_id': thresholds_config_id,
            'majority_pct': 0.0,
            'robustness_pct': 0.0,
            'global_score': 1.0,
            'verdict': 'WIP[R0-open]',
            'verdict_reason': 'Aucune donnée disponible',
            'details': {}
        }
    
    # Convertir rows en dicts
    all_scores = []
    for row in rows:
        all_scores.append({
            'exec_id': row[0],
            'test_name': row[1],
            'test_score': row[2],
            'test_weight': row[3],
            'pathology_flags': row[4],
            'critical_metrics': row[5],
            'd_base_id': row[6],
            'modifier_id': row[7],
            'seed': row[8]
        })
    
    # 3. Calculer critère MAJORITÉ
    configs_healthy = [s for s in all_scores if s['test_score'] < threshold_test]
    majority_pct = (len(configs_healthy) / len(all_scores)) * 100
    
    # 4. Calculer critère ROBUSTESSE (grouper par D)
    d_bases = {}
    for score in all_scores:
        d_id = score['d_base_id']
        if d_id not in d_bases:
            d_bases[d_id] = []
        d_bases[d_id].append(score['test_score'])
    
    viable_d = sum(1 for scores in d_bases.values() 
                   if any(s < threshold_test for s in scores))
    robustness_pct = (viable_d / len(d_bases)) * 100 if d_bases else 0
    
    # 5. Calculer GLOBAL_SCORE
    global_score = float(np.mean([s['test_score'] for s in all_scores]))
    
    # 6. Appliquer règles verdict
    verdict, verdict_reason = _apply_verdict_rules(
        global_score=global_score,
        robustness_pct=robustness_pct,
        majority_pct=majority_pct,
        survives_criteria=survives_criteria,
        rejected_criteria=rejected_criteria
    )
    
    # 7. Construire détails (pour rapport)
    critical_tests = _identify_critical_tests(all_scores, threshold=0.8)
    d_breakdown = _compute_d_breakdown(d_bases, threshold_test)
    
    details = {
        'n_total_configs': len(all_scores),
        'n_healthy_configs': len(configs_healthy),
        'n_total_d': len(d_bases),
        'n_viable_d': viable_d,
        'critical_tests': critical_tests,
        'd_breakdown': d_breakdown,
        'thresholds_used': {
            'threshold_test': threshold_test,
            'survives': survives_criteria,
            'rejected': rejected_criteria
        }
    }
    
    return {
        'gamma_id': gamma_id,
        'params_config_id': params_config_id,
        'scoring_config_id': scoring_config_id,
        'thresholds_config_id': thresholds_config_id,
        'majority_pct': majority_pct,
        'robustness_pct': robustness_pct,
        'global_score': global_score,
        'verdict': verdict,
        'verdict_reason': verdict_reason,
        'details': details
    }


def _apply_verdict_rules(
    global_score: float,
    robustness_pct: float,
    majority_pct: float,
    survives_criteria: dict,
    rejected_criteria: dict
) -> tuple[str, str]:
    """
    Applique règles verdict.
    
    Returns:
        (verdict, reason)
    """
    # Logique OU pour SURVIVES
    survives_checks = {
        'score': global_score < survives_criteria['score_max'],
        'robustness': robustness_pct >= survives_criteria['robustness_min'] * 100,
        'majority': majority_pct >= survives_criteria['majority_min'] * 100
    }
    
    if any(survives_checks.values()):
        verdict = "SURVIVES[R0]"
        
        reasons = []
        if survives_checks['score']:
            reasons.append(
                f"global_score={global_score:.3f} < {survives_criteria['score_max']}"
            )
        if survives_checks['robustness']:
            reasons.append(
                f"robustness={robustness_pct:.1f}% ≥ {survives_criteria['robustness_min']*100}%"
            )
        if survives_checks['majority']:
            reasons.append(
                f"majority={majority_pct:.1f}% ≥ {survives_criteria['majority_min']*100}%"
            )
        
        verdict_reason = "SURVIVES (logique OU): " + " OR ".join(reasons)
        return verdict, verdict_reason
    
    # Logique ET pour REJECTED
    if (global_score > rejected_criteria['score_min'] and
        robustness_pct < rejected_criteria['robustness_max'] * 100 and
        majority_pct < rejected_criteria['majority_max'] * 100):
        
        verdict = "REJECTED[R0]"
        verdict_reason = (
            f"REJECTED (logique ET): "
            f"global_score={global_score:.3f} > {rejected_criteria['score_min']} AND "
            f"robustness={robustness_pct:.1f}% < {rejected_criteria['robustness_max']*100}% AND "
            f"majority={majority_pct:.1f}% < {rejected_criteria['majority_max']*100}%"
        )
        return verdict, verdict_reason
    
    # Sinon WIP
    verdict = "WIP[R0-open]"
    verdict_reason = (
        f"WIP (ni SURVIVES ni REJECTED): "
        f"global_score={global_score:.3f}, "
        f"robustness={robustness_pct:.1f}%, "
        f"majority={majority_pct:.1f}%"
    )
    return verdict, verdict_reason


def _identify_critical_tests(
    all_scores: List[dict],
    threshold: float = 0.8
) -> List[dict]:
    """Identifie tests avec score critique."""
    critical = []
    
    for s in all_scores:
        if s['test_score'] >= threshold:
            critical.append({
                'exec_id': s['exec_id'],
                'test_name': s['test_name'],
                'test_score': s['test_score'],
                'pathology_flags': json.loads(s['pathology_flags']) if s['pathology_flags'] else [],
                'critical_metrics': json.loads(s['critical_metrics']) if s['critical_metrics'] else []
            })
    
    return critical


def _compute_d_breakdown(
    d_bases: Dict[str, List[float]],
    threshold_test: float
) -> Dict[str, dict]:
    """Calcule décomposition par D."""
    breakdown = {}
    
    for d_id, scores in d_bases.items():
        breakdown[d_id] = {
            'n_configs': len(scores),
            'n_healthy': sum(1 for s in scores if s < threshold_test),
            'mean_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }
    
    return breakdown


def generate_verdict_report(verdict_dict: dict) -> str:
    """
    Génère rapport texte détaillé.
    
    Args:
        verdict_dict: Résultat de compute_gamma_verdict()
    
    Returns:
        str: Rapport formaté
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"VERDICT REPORT: {verdict_dict['gamma_id']}")
    lines.append("=" * 80)
    lines.append("")
    
    # Configuration
    lines.append("Configuration:")
    lines.append(f"  Params:     {verdict_dict['params_config_id']}")
    lines.append(f"  Scoring:    {verdict_dict['scoring_config_id']}")
    lines.append(f"  Thresholds: {verdict_dict['thresholds_config_id']}")
    lines.append(f"  Computed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Verdict
    lines.append("=" * 80)
    lines.append(f"VERDICT: {verdict_dict['verdict']}")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("Raison:")
    lines.append(f"  {verdict_dict['verdict_reason']}")
    lines.append("")
    
    # Critères détail
    lines.append("Critères (détail):")
    
    thresholds = verdict_dict['details']['thresholds_used']
    global_score = verdict_dict['global_score']
    robustness = verdict_dict['robustness_pct']
    majority = verdict_dict['majority_pct']
    
    # Global score
    score_pass = global_score < thresholds['survives']['score_max']
    lines.append(
        f"  {'✓' if score_pass else '✗'} global_score:  "
        f"{global_score:.3f}  "
        f"[{'PASS' if score_pass else 'FAIL'}: "
        f"{'<' if score_pass else '≥'} {thresholds['survives']['score_max']}]"
    )
    
    # Robustness
    rob_pass = robustness >= thresholds['survives']['robustness_min'] * 100
    lines.append(
        f"  {'✓' if rob_pass else '✗'} robustness:    "
        f"{robustness:.1f}%  "
        f"[{'PASS' if rob_pass else 'FAIL'}: "
        f"{'≥' if rob_pass else '<'} {thresholds['survives']['robustness_min']*100}%]"
    )
    
    # Majority
    maj_pass = majority >= thresholds['survives']['majority_min'] * 100
    lines.append(
        f"  {'✓' if maj_pass else '✗'} majority:      "
        f"{majority:.1f}%  "
        f"[{'PASS' if maj_pass else 'FAIL'}: "
        f"{'≥' if maj_pass else '<'} {thresholds['survives']['majority_min']*100}%]"
    )
    lines.append("")
    
    verdict = verdict_dict['verdict']
    if verdict == "SURVIVES[R0]":
        lines.append("Verdict: SURVIVES car AU MOINS UN critère passe (logique OU)")
    elif verdict == "REJECTED[R0]":
        lines.append("Verdict: REJECTED car TOUS les critères échouent (logique ET)")
    else:
        lines.append("Verdict: WIP car ni SURVIVES ni REJECTED")
    lines.append("")
    
    # Analyse détaillée
    lines.append("=" * 80)
    lines.append("ANALYSE DÉTAILLÉE")
    lines.append("=" * 80)
    lines.append("")
    
    details = verdict_dict['details']
    
    lines.append("Statistiques globales:")
    lines.append(f"  - Total configs testées:      {details['n_total_configs']}")
    lines.append(f"  - Configs saines:             {details['n_healthy_configs']} ({majority:.1f}%)")
    lines.append(f"  - D testés:                   {details['n_total_d']}")
    lines.append(f"  - D viables (≥1 config saine): {details['n_viable_d']} ({robustness:.1f}%)")
    lines.append("")
    
    # Tests critiques
    critical_tests = details['critical_tests']
    if critical_tests:
        lines.append(f"Tests critiques (score ≥ 0.80): {len(critical_tests)}")
        lines.append("")
        
        for i, test in enumerate(critical_tests[:10], 1):  # Top 10
            lines.append(f"  [{i}] exec_id={test['exec_id']}, test={test['test_name']}")
            lines.append(f"      score: {test['test_score']:.3f}")
            if test['pathology_flags']:
                lines.append(f"      flags: {test['pathology_flags']}")
            if test['critical_metrics']:
                lines.append(f"      metrics critiques: {test['critical_metrics']}")
            lines.append("")
        
        if len(critical_tests) > 10:
            lines.append(f"  ... et {len(critical_tests) - 10} autres tests critiques")
            lines.append("")
    else:
        lines.append("Tests critiques: Aucun")
        lines.append("")
    
    # Décomposition par D
    lines.append("Décomposition par D:")
    lines.append("")
    
    d_breakdown = details['d_breakdown']
    for d_id, stats in sorted(d_breakdown.items()):
        health_pct = (stats['n_healthy'] / stats['n_configs']) * 100 if stats['n_configs'] > 0 else 0
        lines.append(
            f"  {d_id}: {stats['n_configs']} configs, "
            f"{stats['n_healthy']} saines ({health_pct:.0f}%), "
            f"score moyen={stats['mean_score']:.3f}"
        )
    lines.append("")
    
    # Conclusion
    lines.append("=" * 80)
    lines.append("CONCLUSION")
    lines.append("=" * 80)
    lines.append("")
    
    if verdict == "SURVIVES[R0]":
        lines.append(f"Γ={verdict_dict['gamma_id']} non éliminé en R0 (SURVIVES).")
        lines.append("")
        lines.append("Pathologies identifiées:")
        # TODO: Extraire patterns pathologies depuis critical_tests
        lines.append("  (Analyse détaillée pathologies nécessite inspection manuelle)")
        lines.append("")
        lines.append("Recommandations exploration R1:")
        lines.append("  - Investiguer tests critiques identifiés")
        lines.append("  - Affiner seuils scoring si nécessaire")
        lines.append("  - Tester compositions avec autres mécanismes")
    
    elif verdict == "REJECTED[R0]":
        lines.append(f"Γ={verdict_dict['gamma_id']} rejeté en R0 (REJECTED).")
        lines.append("")
        lines.append("Pathologies systématiques détectées:")
        lines.append("  - Échec sur tous critères (score, robustesse, majorité)")
        lines.append("  - Mécanisme non viable en l'état")
    
    else:
        lines.append(f"Γ={verdict_dict['gamma_id']} statut indécis en R0 (WIP).")
        lines.append("")
        lines.append("Actions recommandées:")
        lines.append("  - Collecter plus de données")
        lines.append("  - Tester configs alternatives (strict/lenient)")
        lines.append("  - Investigation manuelle requise")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)