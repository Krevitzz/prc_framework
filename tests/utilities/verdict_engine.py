# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.4 - Analyse patterns cross-runs.

Principe : Verdicts basés sur patterns détectés, pas agrégation naïve.
"""

from typing import Dict, List, Tuple
import numpy as np
import sqlite3
import json
from datetime import datetime
from pathlib import Path


# =============================================================================
# CHARGEMENT SCORES AVEC CONTEXTE
# =============================================================================

def load_scores_with_context(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str,
    db_path: str = 'prc_database/prc_r0_results.db'
) -> List[dict]:
    """
    Charge TOUS les scores avec contexte depuis jointure.
    
    Args:
        gamma_id: ID gamma à analyser
        params_config_id: Config params utilisée
        scoring_config_id: Config scoring utilisée
        db_path: Chemin db_results
    
    Returns:
        List[dict]: Tous scores avec contexte complet
    
    Raises:
        ValueError: Si aucun score trouvé
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            e.id as exec_id,
            e.run_id,
            e.gamma_id,
            e.d_base_id,
            e.modifier_id,
            e.seed,
            ts.test_name,
            ts.test_score,
            ts.metric_scores,
            ts.pathology_flags,
            ts.critical_metrics,
            ts.aggregation_mode
        FROM TestScores ts
        JOIN Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ?
          AND ts.params_config_id = ?
          AND ts.scoring_config_id = ?
    """, (gamma_id, params_config_id, scoring_config_id))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(
            f"Aucun score trouvé pour gamma={gamma_id}, "
            f"params={params_config_id}, scoring={scoring_config_id}"
        )
    
    # Conversion + parse JSON
    results = []
    for row in rows:
        results.append({
            'exec_id': row['exec_id'],
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_base_id': row['d_base_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed'],
            'test_name': row['test_name'],
            'test_score': row['test_score'],
            'metric_scores': json.loads(row['metric_scores']),
            'pathology_flags': json.loads(row['pathology_flags']),
            'critical_metrics': json.loads(row['critical_metrics']),
            'aggregation_mode': row['aggregation_mode']
        })
    
    return results


# =============================================================================
# ANALYSE PATTERNS
# =============================================================================

def analyze_metric_patterns(scores_data: List[dict]) -> dict:
    """
    Détecte 7 types de patterns dans scores.
    
    Args:
        scores_data: Liste scores avec contexte
    
    Returns:
        {
            'non_discriminant': list[str],
            'over_discriminant': list[str],
            'd_correlated': dict[str, dict],
            'modifier_correlated': dict[str, dict],
            'seed_unstable': list[str],
            'systematic_failures': list[str],
            'contextual_behaviors': list[str]
        }
    """
    patterns = {
        'non_discriminant': [],
        'over_discriminant': [],
        'd_correlated': {},
        'modifier_correlated': {},
        'seed_unstable': [],
        'systematic_failures': [],
        'contextual_behaviors': []
    }
    
    # Grouper par (test_name, metric_key)
    by_metric = {}
    for row in scores_data:
        test_name = row['test_name']
        for metric_key, metric_data in row['metric_scores'].items():
            key = f"{test_name}.{metric_key}"
            if key not in by_metric:
                by_metric[key] = []
            
            by_metric[key].append({
                'score': metric_data['score'],
                'd_base_id': row['d_base_id'],
                'modifier_id': row['modifier_id'],
                'seed': row['seed'],
                'flag': metric_data['flag']
            })
    
    # Analyse par métrique
    for metric_full_key, metric_data in by_metric.items():
        scores = [d['score'] for d in metric_data]
        
        # 1. NON_DISCRIMINANT (tous < 0.1)
        if max(scores) < 0.1:
            patterns['non_discriminant'].append(metric_full_key)
            continue  # Skip autres analyses
        
        # 2. OVER_DISCRIMINANT (tous > 0.9)
        if min(scores) > 0.9:
            patterns['over_discriminant'].append(metric_full_key)
            continue
        
        # === MÉTRIQUES VALIDES (discriminantes) ===
        
        # 3. D_CORRELATED
        d_groups = {}
        for d in metric_data:
            d_id = d['d_base_id']
            if d_id not in d_groups:
                d_groups[d_id] = []
            d_groups[d_id].append(d['score'])
        
        if len(d_groups) > 1:
            d_means = [np.mean(scores) for scores in d_groups.values()]
            variance_inter_d = np.var(d_means)
            
            if variance_inter_d > 0.3:
                d_pattern = {
                    d_id: {
                        'mean': float(np.mean(d_scores)),
                        'std': float(np.std(d_scores)),
                        'count': len(d_scores)
                    }
                    for d_id, d_scores in d_groups.items()
                }
                patterns['d_correlated'][metric_full_key] = d_pattern
        
        # 4. MODIFIER_CORRELATED
        mod_groups = {}
        for d in metric_data:
            mod_id = d['modifier_id']
            if mod_id not in mod_groups:
                mod_groups[mod_id] = []
            mod_groups[mod_id].append(d['score'])
        
        if len(mod_groups) > 1:
            mod_means = [np.mean(scores) for scores in mod_groups.values()]
            variance_inter_mod = np.var(mod_means)
            
            if variance_inter_mod > 0.3:
                mod_pattern = {
                    mod_id: {
                        'mean': float(np.mean(mod_scores)),
                        'std': float(np.std(mod_scores)),
                        'count': len(mod_scores)
                    }
                    for mod_id, mod_scores in mod_groups.items()
                }
                patterns['modifier_correlated'][metric_full_key] = mod_pattern
        
        # 5. SEED_UNSTABLE
        config_groups = {}
        for d in metric_data:
            config_key = (d['d_base_id'], d['modifier_id'])
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(d['score'])
        
        max_intra_variance = 0.0
        for scores in config_groups.values():
            if len(scores) > 1:
                var = np.var(scores)
                max_intra_variance = max(max_intra_variance, var)
        
        if max_intra_variance > 0.3:
            patterns['seed_unstable'].append(metric_full_key)
        
        # 6. SYSTEMATIC_FAILURE (≥80% scores > 0.7)
        high_scores = sum(1 for s in scores if s > 0.7)
        if high_scores / len(scores) >= 0.8:
            patterns['systematic_failures'].append(metric_full_key)
    
    # 7. CONTEXTUAL_BEHAVIOR
    has_d = bool(patterns['d_correlated'])
    has_mod = bool(patterns['modifier_correlated'])
    no_systematic = not patterns['systematic_failures']
    
    if (has_d or has_mod) and no_systematic:
        affected = list(patterns['d_correlated'].keys()) + \
                  list(patterns['modifier_correlated'].keys())
        patterns['contextual_behaviors'] = list(set(affected))
    
    return patterns


def compute_metric_quality(scores_data: List[dict]) -> dict:
    """
    Évalue qualité métriques (discriminante, calibrée).
    
    Args:
        scores_data: Liste scores avec contexte
    
    Returns:
        {
            'TEST.metric': {
                'valid': bool,
                'discriminant': bool,
                'calibrated': bool,
                'variance': float,
                'mean_score': float
            }
        }
    """
    quality = {}
    
    # Grouper par métrique
    by_metric = {}
    for row in scores_data:
        test_name = row['test_name']
        for metric_key, metric_data in row['metric_scores'].items():
            key = f"{test_name}.{metric_key}"
            if key not in by_metric:
                by_metric[key] = []
            by_metric[key].append(metric_data['score'])
    
    # Évaluation
    for metric_key, scores in by_metric.items():
        mean_score = float(np.mean(scores))
        variance = float(np.var(scores))
        
        discriminant = max(scores) >= 0.1  # Pas toujours 0
        calibrated = min(scores) <= 0.9   # Pas toujours 1
        valid = discriminant and calibrated
        
        quality[metric_key] = {
            'valid': valid,
            'discriminant': discriminant,
            'calibrated': calibrated,
            'variance': variance,
            'mean_score': mean_score
        }
    
    return quality


# =============================================================================
# DÉCISION VERDICT
# =============================================================================

def decide_verdict_from_patterns(
    patterns: dict,
    metric_quality: dict
) -> Tuple[str, str]:
    """
    Décide verdict depuis patterns détectés.
    
    Args:
        patterns: Patterns détectés
        metric_quality: Qualité métriques
    
    Returns:
        (verdict: str, reason: str)
    """
    # Métriques valides
    valid_metrics = [k for k, v in metric_quality.items() if v['valid']]
    
    # REJECTED : Pathologies systématiques sur métriques valides
    systematic = patterns.get('systematic_failures', [])
    systematic_valid = [m for m in systematic if m in valid_metrics]
    
    if len(systematic_valid) >= 3:
        return (
            "REJECTED[R0]",
            f"Pathologies systématiques sur {len(systematic_valid)} métriques valides : "
            f"{', '.join(systematic_valid[:3])}"
        )
    
    # WIP : Comportements contextuels
    has_d_context = bool(patterns.get('d_correlated'))
    has_mod_context = bool(patterns.get('modifier_correlated'))
    
    if has_d_context or has_mod_context:
        affected_count = len(patterns.get('d_correlated', {})) + \
                        len(patterns.get('modifier_correlated', {}))
        return (
            "WIP[R0-open]",
            f"Comportement contextuel détecté sur {affected_count} métriques. "
            f"Nécessite investigation ciblée."
        )
    
    # WIP : Instabilité stochastique
    if len(patterns.get('seed_unstable', [])) >= 2:
        return (
            "WIP[R0-open]",
            f"Instabilité stochastique sur {len(patterns['seed_unstable'])} métriques. "
            f"Augmenter nombre seeds pour confirmer."
        )
    
    # SURVIVES : Pas de pathologie systématique
    return (
        "SURVIVES[R0]",
        "Aucune pathologie systématique détectée. "
        "Mécanisme non absurde sur espace testé."
    )


def generate_actionable_insights(
    patterns: dict,
    metric_quality: dict
) -> List[str]:
    """
    Génère insights actionnables depuis patterns.
    
    Args:
        patterns: Patterns détectés
        metric_quality: Qualité métriques
    
    Returns:
        List[str]: Insights formatés
    """
    insights = []
    
    # Non discriminantes
    if patterns.get('non_discriminant'):
        metrics = patterns['non_discriminant']
        insights.append(
            f"⚠ Recalibrer métriques non discriminantes ({len(metrics)}) : "
            f"{', '.join(metrics[:3])}. "
            f"Action : Augmenter seuils ou supprimer métriques."
        )
    
    # Over discriminantes
    if patterns.get('over_discriminant'):
        metrics = patterns['over_discriminant']
        insights.append(
            f"⚠ Recalibrer métriques over-discriminantes ({len(metrics)}) : "
            f"{', '.join(metrics[:3])}. "
            f"Action : Augmenter seuils critiques."
        )
    
    # Corrélations D
    if patterns.get('d_correlated'):
        for metric, d_pattern in list(patterns['d_correlated'].items())[:3]:
            failing_d = [d for d, stats in d_pattern.items() if stats['mean'] > 0.7]
            if failing_d:
                insights.append(
                    f"📊 {metric} échoue sur {', '.join(failing_d)}. "
                    f"Action : Γ spécialisé, investiguer amplification sur ces D."
                )
    
    # Corrélations modifier
    if patterns.get('modifier_correlated'):
        metrics = list(patterns['modifier_correlated'].keys())
        insights.append(
            f"📊 Sensibilité au bruit détectée ({len(metrics)} métriques) : "
            f"{', '.join(metrics[:3])}. "
            f"Action : Tester régularisation ou pré-filtrage."
        )
    
    # Instabilité seeds
    if patterns.get('seed_unstable'):
        metrics = patterns['seed_unstable']
        insights.append(
            f"⚠ Instabilité stochastique ({len(metrics)} métriques) : "
            f"{', '.join(metrics[:3])}. "
            f"Action : Augmenter seeds (×3) ou investiguer sensibilité initiale."
        )
    
    return insights


# =============================================================================
# GÉNÉRATION RAPPORTS
# =============================================================================

def generate_human_report(verdict_data: dict, output_path: str):
    """
    Génère rapport humain (txt).
    
    Args:
        verdict_data: Données verdict complètes
        output_path: Chemin fichier sortie
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VERDICT REPORT: {verdict_data['gamma_id']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Params:  {verdict_data['params_config_id']}\n")
        f.write(f"  Scoring: {verdict_data['scoring_config_id']}\n")
        f.write(f"  Runs:    {verdict_data['num_runs_analyzed']}\n")
        f.write(f"  Tests:   {verdict_data['num_tests_analyzed']}\n\n")
        
        f.write("VERDICT:\n")
        f.write(f"  {verdict_data['verdict']}\n\n")
        
        f.write("REASON:\n")
        f.write(f"  {verdict_data['verdict_reason']}\n\n")
        
        patterns = verdict_data['patterns_summary']['patterns_detected']
        f.write("Patterns clés:\n")
        for pattern_type, pattern_data in patterns.items():
            if pattern_data and pattern_type not in ['contextual_behaviors']:
                f.write(f"  - {pattern_type}: {len(pattern_data)} occurrences\n")
        
        insights = verdict_data['patterns_summary']['actionable_insights']
        if insights:
            f.write("\nInsights actionnables:\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"  {i}. {insight}\n")
        
        critical = verdict_data['patterns_summary'].get('critical_metrics', [])
        if critical:
            f.write("\nMétriques critiques:\n")
            for metric in critical[:5]:
                f.write(f"  - {metric}\n")
        
        f.write("\n" + "="*80 + "\n")


def generate_llm_report(verdict_data: dict, output_path: str):
    """
    Génère rapport LLM (JSON complet).
    
    Args:
        verdict_data: Données verdict complètes
        output_path: Chemin fichier sortie
    """
    with open(output_path, 'w') as f:
        json.dump(verdict_data, f, indent=2)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def compute_gamma_verdict(
    gamma_id: str,
    params_config_id: str,
    scoring_config_id: str
) -> dict:
    """
    Calcule verdict global pour Γ via analyse patterns.
    
    Args:
        gamma_id: ID gamma à évaluer
        params_config_id: Config params utilisée
        scoring_config_id: Config scoring utilisée
    
    Returns:
        {
            'gamma_id': str,
            'params_config_id': str,
            'scoring_config_id': str,
            'verdict': str,
            'verdict_reason': str,
            'patterns_summary': dict,
            'num_runs_analyzed': int,
            'num_tests_analyzed': int,
            'computed_at': str
        }
    
    Raises:
        ValueError: Si aucun score trouvé
    """
    print(f"\n{'='*70}")
    print(f"CALCUL VERDICT: {gamma_id}")
    print(f"{'='*70}\n")
    
    # 1. Charger scores avec contexte
    print("1. Chargement scores...")
    scores_data = load_scores_with_context(
        gamma_id, params_config_id, scoring_config_id
    )
    print(f"   ✓ {len(scores_data)} scores chargés")
    
    # 2. Analyse patterns
    print("2. Analyse patterns...")
    patterns = analyze_metric_patterns(scores_data)
    metric_quality = compute_metric_quality(scores_data)
    
    # Affichage patterns détectés
    for pattern_type, pattern_data in patterns.items():
        if pattern_data:
            count = len(pattern_data) if isinstance(pattern_data, (list, dict)) else 0
            print(f"   - {pattern_type}: {count}")
    
    # 3. Décision verdict
    print("3. Décision verdict...")
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    print(f"   → {verdict}")
    
    # 4. Insights
    print("4. Génération insights...")
    insights = generate_actionable_insights(patterns, metric_quality)
    print(f"   ✓ {len(insights)} insights générés")
    
    # 5. Extraction infos additionnelles
    all_exec_ids = list(set(row['exec_id'] for row in scores_data))
    
    failing_d = []
    if patterns.get('d_correlated'):
        for metric, d_pattern in patterns['d_correlated'].items():
            failing_d.extend([
                d for d, stats in d_pattern.items()
                if stats['mean'] > 0.7
            ])
    failing_d = list(set(failing_d))
    
    failing_modifiers = []
    if patterns.get('modifier_correlated'):
        for metric, mod_pattern in patterns['modifier_correlated'].items():
            failing_modifiers.extend([
                mod for mod, stats in mod_pattern.items()
                if stats['mean'] > 0.7
            ])
    failing_modifiers = list(set(failing_modifiers))
    
    critical_metrics = []
    for row in scores_data:
        critical_metrics.extend(row['critical_metrics'])
    critical_metrics = list(set(critical_metrics))
    
    num_tests = len(set(row['test_name'] for row in scores_data))
    
    # 6. Construction résultat
    verdict_data = {
        'gamma_id': gamma_id,
        'params_config_id': params_config_id,
        'scoring_config_id': scoring_config_id,
        
        'verdict': verdict,
        'verdict_reason': reason,
        
        'patterns_summary': {
            'patterns_detected': patterns,
            'metric_quality': metric_quality,
            'actionable_insights': insights,
            'all_exec_ids': all_exec_ids,
            'failing_d': failing_d,
            'failing_modifiers': failing_modifiers,
            'critical_metrics': critical_metrics,
            'scores_by_run': scores_data  # Pour rapport LLM
        },
        
        'num_runs_analyzed': len(all_exec_ids),
        'num_tests_analyzed': num_tests,
        'computed_at': datetime.now().isoformat()
    }
    
    return verdict_data