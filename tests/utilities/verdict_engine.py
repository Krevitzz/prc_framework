# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.5 - Analyse patterns sur observations brutes.
"""

from typing import Dict, List
import numpy as np
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import percentileofscore

from .config_loader import get_loader


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize_values(
    values: List[float],
    method: str = "robust",
    params: dict = None
) -> np.ndarray:
    """
    Normalise valeurs selon méthode spécifiée.
    
    Args:
        values: Valeurs brutes
        method: "robust" | "z_score" | "percentile"
        params: Paramètres normalisation (outlier_threshold, etc.)
    
    Returns:
        Valeurs normalisées (np.ndarray)
    
    Examples:
        >>> values = [10, 20, 15, 18, 12]
        >>> normalized = normalize_values(values, method="robust")
        >>> # (x - median) / IQR
    """
    values = np.array(values, dtype=float)
    
    if method == "robust":
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr < 1e-10:  # Protection division par zéro
            return np.zeros_like(values)
        
        return (values - median) / iqr
    
    elif method == "z_score":
        mean = np.mean(values)
        std = np.std(values)
        
        if std < 1e-10:
            return np.zeros_like(values)
        
        return (values - mean) / std
    
    elif method == "percentile":
        return np.array([percentileofscore(values, v) / 100.0 for v in values])
    
    else:
        raise ValueError(f"Méthode normalisation inconnue : {method}")


# =============================================================================
# CHARGEMENT OBSERVATIONS
# =============================================================================

def load_all_observations(
    params_config_id: str,
    db_path: str = 'prc_database/prc_r0_results.db'
) -> List[dict]:
    """
    Charge TOUTES les observations pour params_config donné.
    
    Args:
        params_config_id: Config params utilisée
        db_path: Chemin db_results
    
    Returns:
        List[dict]: Observations complètes avec métadonnées
    
    Raises:
        ValueError: Si aucune observation trouvée
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            o.observation_id,
            o.exec_id,
            o.test_name,
            o.params_config_id,
            o.status,
            o.observation_data,
            o.computed_at,
            e.run_id,
            e.gamma_id,
            e.d_encoding_id,
            e.modifier_id,
            e.seed
        FROM TestObservations o
        JOIN Executions e ON o.exec_id = e.id
        WHERE o.params_config_id = ?
          AND o.status = 'SUCCESS'
    """, (params_config_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(
            f"Aucune observation SUCCESS trouvée pour params={params_config_id}"
        )
    
    # Parse JSON observations
    observations = []
    for row in rows:
        obs_data = json.loads(row['observation_data'])
        
        observations.append({
            'observation_id': row['observation_id'],
            'exec_id': row['exec_id'],
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_encoding_id': row['d_encoding_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed'],
            'test_name': row['test_name'],
            'params_config_id': row['params_config_id'],
            'observation_data': obs_data,
            'computed_at': row['computed_at']
        })
    
    return observations


# =============================================================================
# DÉTECTION PATTERNS
# =============================================================================

def detect_non_discriminant(
    observations: List[dict],
    test_name: str,
    metric_name: str,
    config: dict
) -> dict | None:
    """
    Détecte pattern NON_DISCRIMINANT.
    
    Toutes observations normalisées < normalized_threshold.
    """
    pattern_config = config['patterns']['non_discriminant']
    method = pattern_config.get('method', config['normalization']['default_method'])
    threshold = pattern_config['normalized_threshold']
    min_obs = pattern_config['min_observations']
    
    # Extraire valeurs
    values = []
    for obs in observations:
        if obs['test_name'] != test_name:
            continue
        
        stats = obs['observation_data']['statistics']
        if metric_name not in stats:
            continue
        
        values.append(stats[metric_name]['final'])
    
    if len(values) < min_obs:
        return None
    
    # Normaliser
    normalized = normalize_values(values, method=method)
    
    # Vérifier pattern
    if np.all(normalized < threshold):
        return {
            'pattern_type': 'NON_DISCRIMINANT',
            'test_name': test_name,
            'metric_name': metric_name,
            'method': method,
            'threshold': threshold,
            'n_observations': len(values),
            'max_normalized': float(np.max(normalized)),
            'raw_values_range': [float(np.min(values)), float(np.max(values))]
        }
    
    return None


def detect_d_correlated(
    observations: List[dict],
    test_name: str,
    metric_name: str,
    config: dict
) -> dict | None:
    """
    Détecte pattern D_CORRELATED.
    
    Variance inter-d_encoding_id des valeurs normalisées > variance_threshold.
    """
    pattern_config = config['patterns']['d_correlated']
    variance_threshold = pattern_config['variance_threshold']
    min_d_groups = pattern_config['min_d_groups']
    method = config['normalization']['default_method']
    
    # Grouper par d_encoding_id
    by_d = {}
    for obs in observations:
        if obs['test_name'] != test_name:
            continue
        
        stats = obs['observation_data']['statistics']
        if metric_name not in stats:
            continue
        
        d_id = obs['d_encoding_id']
        if d_id not in by_d:
            by_d[d_id] = []
        
        by_d[d_id].append(stats[metric_name]['final'])
    
    if len(by_d) < min_d_groups:
        return None
    
    # Normaliser moyennes par groupe
    d_means = [np.mean(vals) for vals in by_d.values()]
    d_means_normalized = normalize_values(d_means, method=method)
    
    variance = float(np.var(d_means_normalized))
    
    if variance > variance_threshold:
        return {
            'pattern_type': 'D_CORRELATED',
            'test_name': test_name,
            'metric_name': metric_name,
            'method': method,
            'variance': variance,
            'variance_threshold': variance_threshold,
            'n_d_groups': len(by_d),
            'by_d_encoding': {
                d_id: {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'n': len(vals)
                }
                for d_id, vals in by_d.items()
            }
        }
    
    return None


def detect_seed_unstable(
    observations: List[dict],
    test_name: str,
    metric_name: str,
    config: dict
) -> dict | None:
    """
    Détecte pattern SEED_UNSTABLE.
    
    Variance inter-seeds > variance_threshold (à D et modifier fixés).
    """
    pattern_config = config['patterns']['seed_unstable']
    variance_threshold = pattern_config['variance_threshold']
    min_seeds = pattern_config['min_seeds']
    method = config['normalization']['default_method']
    
    # Grouper par (d_encoding_id, modifier_id), puis seeds
    config_groups = {}
    for obs in observations:
        if obs['test_name'] != test_name:
            continue
        
        stats = obs['observation_data']['statistics']
        if metric_name not in stats:
            continue
        
        config_key = (obs['d_encoding_id'], obs['modifier_id'])
        if config_key not in config_groups:
            config_groups[config_key] = []
        
        config_groups[config_key].append(stats[metric_name]['final'])
    
    # Calculer variance max intra-config
    max_variance = 0.0
    unstable_config = None
    
    for config_key, values in config_groups.items():
        if len(values) < min_seeds:
            continue
        
        normalized = normalize_values(values, method=method)
        variance = float(np.var(normalized))
        
        if variance > max_variance:
            max_variance = variance
            unstable_config = config_key
    
    if max_variance > variance_threshold and unstable_config:
        return {
            'pattern_type': 'SEED_UNSTABLE',
            'test_name': test_name,
            'metric_name': metric_name,
            'method': method,
            'max_variance': max_variance,
            'variance_threshold': variance_threshold,
            'unstable_config': {
                'd_encoding_id': unstable_config[0],
                'modifier_id': unstable_config[1]
            }
        }
    
    return None


def detect_systematic_anomaly(
    observations: List[dict],
    test_name: str,
    metric_name: str,
    config: dict
) -> dict | None:
    """
    Détecte pattern SYSTEMATIC_ANOMALY.
    
    Fraction observations |normalized| > threshold dépasse ratio_threshold.
    """
    pattern_config = config['patterns']['systematic_anomaly']
    normalized_threshold = pattern_config['normalized_threshold']
    ratio_threshold = pattern_config['ratio_threshold']
    min_obs = pattern_config['min_observations']
    method = pattern_config.get('method', config['normalization']['default_method'])
    
    # Extraire valeurs
    values = []
    for obs in observations:
        if obs['test_name'] != test_name:
            continue
        
        stats = obs['observation_data']['statistics']
        if metric_name not in stats:
            continue
        
        values.append(stats[metric_name]['final'])
    
    if len(values) < min_obs:
        return None
    
    # Normaliser
    normalized = normalize_values(values, method=method)
    
    # Compter extrêmes
    extreme_count = np.sum(np.abs(normalized) > normalized_threshold)
    ratio = extreme_count / len(normalized)
    
    if ratio > ratio_threshold:
        return {
            'pattern_type': 'SYSTEMATIC_ANOMALY',
            'test_name': test_name,
            'metric_name': metric_name,
            'method': method,
            'normalized_threshold': normalized_threshold,
            'ratio_threshold': ratio_threshold,
            'ratio_observed': float(ratio),
            'n_extreme': int(extreme_count),
            'n_total': len(values)
        }
    
    return None


def analyze_all_patterns(
    observations: List[dict],
    verdict_config: dict
) -> dict:
    """
    Analyse tous patterns sur toutes observations.
    
    Args:
        observations: Liste observations
        verdict_config: Config verdict chargée
    
    Returns:
        {
            'non_discriminant': [...],
            'd_correlated': [...],
            'seed_unstable': [...],
            'systematic_anomaly': [...]
        }
    """
    patterns = {
        'non_discriminant': [],
        'd_correlated': [],
        'seed_unstable': [],
        'systematic_anomaly': []
    }
    
    # Extraire tous (test, metric) uniques
    test_metrics = set()
    for obs in observations:
        test_name = obs['test_name']
        for metric_name in obs['observation_data']['statistics'].keys():
            test_metrics.add((test_name, metric_name))
    
    # Analyser chaque (test, metric)
    for test_name, metric_name in test_metrics:
        # NON_DISCRIMINANT
        pattern = detect_non_discriminant(observations, test_name, metric_name, verdict_config)
        if pattern:
            patterns['non_discriminant'].append(pattern)
        
        # D_CORRELATED
        pattern = detect_d_correlated(observations, test_name, metric_name, verdict_config)
        if pattern:
            patterns['d_correlated'].append(pattern)
        
        # SEED_UNSTABLE
        pattern = detect_seed_unstable(observations, test_name, metric_name, verdict_config)
        if pattern:
            patterns['seed_unstable'].append(pattern)
        
        # SYSTEMATIC_ANOMALY
        pattern = detect_systematic_anomaly(observations, test_name, metric_name, verdict_config)
        if pattern:
            patterns['systematic_anomaly'].append(pattern)
    
    return patterns


# =============================================================================
# DÉCISION VERDICT
# =============================================================================

def decide_verdict(patterns: dict, verdict_config: dict) -> tuple:
    """
    Décide verdict depuis patterns détectés.
    
    Args:
        patterns: Patterns détectés
        verdict_config: Config verdict
    
    Returns:
        (verdict: str, reason: str)
    """
    rules = verdict_config['verdict_rules']
    
    # REJECTED
    if 'REJECTED' in rules:
        required = rules['REJECTED']['required_patterns']
        min_tests = rules['REJECTED'].get('min_tests_affected', 1)
        
        for pattern_type in required:
            if len(patterns.get(pattern_type, [])) >= min_tests:
                affected = patterns[pattern_type]
                return (
                    "REJECTED[R0]",
                    f"Pattern {pattern_type} détecté sur {len(affected)} tests/métriques : "
                    f"{', '.join(set(p['test_name'] for p in affected[:3]))}"
                )
    
    # WIP
    if 'WIP' in rules:
        required = rules['WIP']['required_patterns']
        min_patterns = rules['WIP'].get('min_patterns', 1)
        
        detected_count = sum(1 for pt in required if patterns.get(pt))
        
        if detected_count >= min_patterns:
            detected_types = [pt for pt in required if patterns.get(pt)]
            return (
                "WIP[R0-open]",
                f"Patterns contextuels détectés : {', '.join(detected_types)}. "
                f"Investigation ciblée nécessaire."
            )
    
    # SURVIVES
    total_patterns = sum(len(v) for v in patterns.values())
    if 'SURVIVES' in rules:
        max_patterns = rules['SURVIVES'].get('max_patterns_detected', 1)
        
        if total_patterns <= max_patterns:
            return (
                "SURVIVES[R0]",
                "Aucune pathologie systématique détectée. "
                "Mécanisme non absurde sur espace testé."
            )
    
    # Défaut WIP
    return (
        "WIP[R0-open]",
        f"{total_patterns} patterns détectés, nécessite analyse approfondie."
    )


# =============================================================================
# GÉNÉRATION RAPPORTS
# =============================================================================

def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    observations: List[dict],
    patterns: dict,
    verdict: str,
    verdict_reason: str,
    output_dir: str = "reports/verdicts"
) -> None:
    """
    Génère rapports verdict (humain + JSON).
    
    Args:
        params_config_id: Config params utilisée
        verdict_config_id: Config verdict utilisée
        observations: Observations analysées
        patterns: Patterns détectés
        verdict: Verdict final
        verdict_reason: Raison verdict
        output_dir: Dossier sortie
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_analysis_full"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'n_observations': len(observations),
        'n_tests': len(set(o['test_name'] for o in observations)),
        'n_runs': len(set(o['exec_id'] for o in observations))
    }
    
    with open(report_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Rapport humain
    with open(report_dir / 'summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VERDICT ANALYSIS - {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Params:  {params_config_id}\n")
        f.write(f"  Verdict: {verdict_config_id}\n")
        f.write(f"  Runs:    {metadata['n_runs']}\n")
        f.write(f"  Tests:   {metadata['n_tests']}\n\n")
        
        f.write("VERDICT:\n")
        f.write(f"  {verdict}\n\n")
        
        f.write("REASON:\n")
        f.write(f"  {verdict_reason}\n\n")
        
        f.write("PATTERNS DETECTED:\n")
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                f.write(f"  {pattern_type}: {len(pattern_list)} occurrences\n")
        
        f.write("\n" + "="*80 + "\n")
    
    # Rapport JSON complet
    report_json = {
        'metadata': metadata,
        'verdict': verdict,
        'verdict_reason': verdict_reason,
        'patterns': patterns,
        'observations_summary': [
            {
                'observation_id': o['observation_id'],
                'run_id': o['run_id'],
                'test_name': o['test_name'],
                'gamma_id': o['gamma_id'],
                'd_encoding_id': o['d_encoding_id']
            }
            for o in observations
        ]
    }
    
    with open(report_dir / 'analysis_complete.json', 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"\n✓ Rapports générés : {report_dir}")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None:
    """
    Pipeline complet verdict.
    
    Args:
        params_config_id: Config params utilisée
        verdict_config_id: Config verdict utilisée
    """
    print(f"\n{'='*70}")
    print(f"VERDICT ANALYSIS")
    print(f"{'='*70}\n")
    
    # 1. Charger config verdict
    print("1. Chargement config verdict...")
    loader = get_loader()
    verdict_config = loader.load('verdict', verdict_config_id)
    print(f"   ✓ Config {verdict_config_id} chargée")
    
    # 2. Charger observations
    print("2. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations chargées")
    
    # 3. Analyser patterns
    print("3. Analyse patterns...")
    patterns = analyze_all_patterns(observations, verdict_config)
    
    total_patterns = sum(len(v) for v in patterns.values())
    print(f"   ✓ {total_patterns} patterns détectés")
    for pattern_type, pattern_list in patterns.items():
        if pattern_list:
            print(f"     - {pattern_type}: {len(pattern_list)}")
    
    # 4. Décider verdict
    print("4. Décision verdict...")
    verdict, reason = decide_verdict(patterns, verdict_config)
    print(f"   → {verdict}")
    
    # 5. Générer rapports
    print("5. Génération rapports...")
    generate_verdict_report(
        params_config_id,
        verdict_config_id,
        observations,
        patterns,
        verdict,
        reason
    )
    
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}\n")