# tests/utilities/score_pattern_analysis.py
"""
Analyse patterns dans scores métriques R0.

Architecture Charter 5.4 - Section 12.9 (patterns)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from collections import defaultdict


def analyze_metric_patterns(scores_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyse patterns dans scores métriques cross-runs.
    
    Args:
        scores_data: Liste de dicts avec structure:
            {
                'test_name': str,
                'metric_scores': dict,  # JSON métriques
                'gamma_id': str,
                'd_base_id': str,
                'modifier_id': str,
                'seed': int,
                'run_id': str
            }
    
    Returns:
        {
            'non_discriminant': list[str],
            'over_discriminant': list[str],
            'modifier_correlated': dict[str, dict],
            'd_correlated': dict[str, dict],
            'seed_unstable': list[str],
            'systematic_failures': list[str],
            'metric_quality': dict[str, dict]
        }
    """
    if not scores_data:
        return _empty_patterns()
    
    # Convertir en format plat pour analyse
    flat_scores = _flatten_scores(scores_data)
    
    patterns = {
        'non_discriminant': [],
        'over_discriminant': [],
        'modifier_correlated': {},
        'd_correlated': {},
        'seed_unstable': [],
        'systematic_failures': []
    }
    
    # Grouper par métrique
    by_metric = _group_by_metric(flat_scores)
    
    for metric_key, metric_data in by_metric.items():
        # 1. Non discriminant (toujours ~0)
        if _is_non_discriminant(metric_data):
            patterns['non_discriminant'].append(metric_key)
        
        # 2. Over discriminant (toujours ~1)
        if _is_over_discriminant(metric_data):
            patterns['over_discriminant'].append(metric_key)
        
        # 3. Corrélation modifier
        modifier_pattern = _analyze_modifier_correlation(metric_data)
        if modifier_pattern:
            patterns['modifier_correlated'][metric_key] = modifier_pattern
        
        # 4. Corrélation type D
        d_pattern = _analyze_d_correlation(metric_data)
        if d_pattern:
            patterns['d_correlated'][metric_key] = d_pattern
        
        # 5. Instabilité seeds
        if _is_seed_unstable(metric_data):
            patterns['seed_unstable'].append(metric_key)
        
        # 6. Échec systématique
        if _is_systematic_failure(metric_data):
            patterns['systematic_failures'].append(metric_key)
    
    # Évaluer qualité métriques
    patterns['metric_quality'] = _evaluate_metric_quality(by_metric, patterns)
    
    return patterns


def _empty_patterns() -> Dict[str, Any]:
    """Retourne patterns vides."""
    return {
        'non_discriminant': [],
        'over_discriminant': [],
        'modifier_correlated': {},
        'd_correlated': {},
        'seed_unstable': [],
        'systematic_failures': [],
        'metric_quality': {}
    }


def _flatten_scores(scores_data: List[Dict]) -> List[Dict]:
    """
    Convertit scores imbriqués en format plat.
    
    Input:
        [{'test_name': 'UNIV-001', 'metric_scores': {...}, ...}]
    
    Output:
        [{'metric_key': 'frobenius_norm', 'score': 0.85, 'test_name': 'UNIV-001', ...}]
    """
    flat = []
    
    for record in scores_data:
        import json
        metric_scores = json.loads(record['metric_scores']) if isinstance(record['metric_scores'], str) else record['metric_scores']
        
        for metric_key, metric_data in metric_scores.items():
            flat.append({
                'metric_key': metric_key,
                'test_name': record['test_name'],
                'score': metric_data['score'],
                'flag': metric_data['flag'],
                'value': metric_data['value'],
                'gamma_id': record['gamma_id'],
                'd_base_id': record['d_base_id'],
                'modifier_id': record['modifier_id'],
                'seed': record['seed'],
                'run_id': record['run_id']
            })
    
    return flat


def _group_by_metric(flat_scores: List[Dict]) -> Dict[str, List[Dict]]:
    """Groupe scores par métrique."""
    by_metric = defaultdict(list)
    for record in flat_scores:
        metric_key = f"{record['test_name']}.{record['metric_key']}"
        by_metric[metric_key].append(record)
    return dict(by_metric)


def _is_non_discriminant(metric_data: List[Dict]) -> bool:
    """Détecte métrique non discriminante (toujours ~0)."""
    scores = [d['score'] for d in metric_data]
    return np.max(scores) < 0.1


def _is_over_discriminant(metric_data: List[Dict]) -> bool:
    """Détecte métrique over-discriminante (toujours ~1)."""
    scores = [d['score'] for d in metric_data]
    return np.min(scores) > 0.9


def _analyze_modifier_correlation(metric_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyse corrélation métrique ↔ modifier.
    
    Returns:
        None si pas de corrélation significative
        Dict avec pattern si corrélation détectée
    """
    by_modifier = defaultdict(list)
    for record in metric_data:
        by_modifier[record['modifier_id']].append(record['score'])
    
    # Calculer moyennes par modifier
    modifier_means = {
        mod: np.mean(scores) 
        for mod, scores in by_modifier.items()
    }
    
    # Vérifier variance inter-modifier
    means_array = np.array(list(modifier_means.values()))
    if len(means_array) < 2:
        return None
    
    variance = np.var(means_array)
    
    # Seuil arbitraire : variance > 0.15 indique corrélation
    if variance > 0.15:
        return {
            'variance': float(variance),
            'by_modifier': {
                mod: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n': len(scores)
                }
                for mod, scores in by_modifier.items()
            },
            'correlation_strength': 'high' if variance > 0.3 else 'medium'
        }
    
    return None


def _analyze_d_correlation(metric_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyse corrélation métrique ↔ type D.
    
    Returns:
        None si pas de corrélation
        Dict avec pattern par D si corrélation détectée
    """
    by_d = defaultdict(list)
    for record in metric_data:
        by_d[record['d_base_id']].append(record['score'])
    
    # Calculer stats par D
    d_stats = {
        d: {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'n': len(scores)
        }
        for d, scores in by_d.items()
    }
    
    # Vérifier variance inter-D
    means = [stats['mean'] for stats in d_stats.values()]
    if len(means) < 2:
        return None
    
    variance = np.var(means)
    
    # Seuil : variance > 0.15
    if variance > 0.15:
        return {
            'variance': float(variance),
            'by_d': d_stats,
            'correlation_strength': 'high' if variance > 0.3 else 'medium',
            'failing_d': [d for d, stats in d_stats.items() if stats['mean'] > 0.7],
            'stable_d': [d for d, stats in d_stats.items() if stats['mean'] < 0.3]
        }
    
    return None


def _is_seed_unstable(metric_data: List[Dict]) -> bool:
    """
    Détecte instabilité inter-seeds.
    
    Groupe par (d_base, modifier), calcule variance inter-seeds.
    """
    by_config = defaultdict(list)
    for record in metric_data:
        key = (record['d_base_id'], record['modifier_id'])
        by_config[key].append(record['score'])
    
    variances = []
    for config, scores in by_config.items():
        if len(scores) >= 2:  # Au moins 2 seeds
            variances.append(np.var(scores))
    
    if not variances:
        return False
    
    mean_variance = np.mean(variances)
    
    # Seuil : variance moyenne > 0.2
    return mean_variance > 0.2


def _is_systematic_failure(metric_data: List[Dict]) -> bool:
    """Détecte échec systématique (score élevé partout)."""
    scores = [d['score'] for d in metric_data]
    mean_score = np.mean(scores)
    
    # Seuil : moyenne > 0.8
    return mean_score > 0.8


def _evaluate_metric_quality(
    by_metric: Dict[str, List[Dict]],
    patterns: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Évalue qualité de chaque métrique.
    
    Returns:
        {
            'metric_key': {
                'valid': bool,
                'discriminant': bool,
                'calibrated': bool,
                'issues': list[str]
            }
        }
    """
    quality = {}
    
    for metric_key, metric_data in by_metric.items():
        scores = [d['score'] for d in metric_data]
        
        issues = []
        
        # Non discriminant
        if metric_key in patterns['non_discriminant']:
            issues.append('non_discriminant')
        
        # Over discriminant
        if metric_key in patterns['over_discriminant']:
            issues.append('over_discriminant')
        
        # Instabilité seeds
        if metric_key in patterns['seed_unstable']:
            issues.append('seed_unstable')
        
        discriminant = metric_key not in patterns['non_discriminant'] and \
                      metric_key not in patterns['over_discriminant']
        
        calibrated = len(issues) == 0
        
        valid = discriminant and calibrated
        
        quality[metric_key] = {
            'valid': valid,
            'discriminant': discriminant,
            'calibrated': calibrated,
            'issues': issues,
            'mean_score': float(np.mean(scores)),
            'variance': float(np.var(scores)),
            'n_samples': len(scores)
        }
    
    return quality


def generate_actionable_insights(patterns: Dict[str, Any]) -> List[str]:
    """
    Génère insights actionnables depuis patterns détectés.
    
    Returns:
        Liste de strings lisibles par humains
    """
    insights = []
    
    # Métriques non discriminantes
    if patterns.get('non_discriminant'):
        metrics = ', '.join(patterns['non_discriminant'])
        insights.append(
            f"⚠ Métriques non discriminantes (toujours ~0) : {metrics}. "
            f"Action : Revoir seuils ou supprimer ces métriques."
        )
    
    # Métriques over-discriminantes
    if patterns.get('over_discriminant'):
        metrics = ', '.join(patterns['over_discriminant'])
        insights.append(
            f"⚠ Métriques trop discriminantes (toujours ~1) : {metrics}. "
            f"Action : Augmenter seuils ou recalibrer."
        )
    
    # Corrélation avec modifier (proxy pour noise)
    if patterns.get('modifier_correlated'):
        for metric, pattern in patterns['modifier_correlated'].items():
            modifiers = pattern['by_modifier']
            high_noise = [mod for mod, stats in modifiers.items() 
                         if stats['mean'] > 0.7]
            if high_noise:
                insights.append(
                    f"📊 {metric} corrélé avec modifier : "
                    f"échec sur {', '.join(high_noise)}. "
                    f"Action : Γ sensible au bruit, investiguer robustesse."
                )
    
    # Corrélation avec type D
    if patterns.get('d_correlated'):
        for metric, pattern in patterns['d_correlated'].items():
            failing_d = pattern.get('failing_d', [])
            stable_d = pattern.get('stable_d', [])
            if failing_d:
                insights.append(
                    f"📊 {metric} échoue sur {', '.join(failing_d)} "
                    f"mais stable sur {', '.join(stable_d)}. "
                    f"Action : Γ spécialisé, investiguer mécanisme sur types D problématiques."
                )
    
    # Instabilité seeds
    if patterns.get('seed_unstable'):
        metrics = ', '.join(patterns['seed_unstable'])
        insights.append(
            f"⚠ Instabilité stochastique : {metrics}. "
            f"Action : Augmenter nombre seeds pour confirmer patterns."
        )
    
    # Échecs systématiques
    if patterns.get('systematic_failures'):
        metrics = ', '.join(patterns['systematic_failures'])
        insights.append(
            f"🔴 Échecs systématiques : {metrics}. "
            f"Action : Pathologie confirmée, investiguer mécanisme Γ."
        )
    
    return insights