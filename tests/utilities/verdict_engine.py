# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.5 - Analyse exploratoire générique CORRIGÉE.

CORRECTIONS CRITIQUES (post-review) :
1. Paires ORIENTÉES : permutations() pas combinations()
2. Interaction = changement d'effet (VR conditionnel >> VR marginal)
3. Filtrage testabilité explicite (min_samples, min_groups)
4. params_config_id exclu interactions (trop corrélé)
5. Drill-down gamma recalculé sur sous-ensemble

ARCHITECTURE :
- FACTORS = liste exhaustive dimensions
- Variance marginale (chaque factor isolé)
- Interactions ORIENTÉES complètes (A|B distinct de B|A)
- Extension triplets ciblée
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, spearmanr
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import warnings

# Imports modules refactorés
from .config_loader import get_loader
from .data_loading import load_all_observations, observations_to_dataframe
from .statistical_utils import (
    compute_eta_squared,
    filter_numeric_artifacts,
    generate_degeneracy_report,
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report,
    kruskal_wallis_test
)
from .regime_utils import stratify_by_regime

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Factors expérimentaux analysés
FACTORS = [
    'gamma_id',
    'd_encoding_id',
    'modifier_id',
    'seed',
    'test_name'
    # params_config_id EXCLU : trop corrélé test_name
]

# Projections numériques
PROJECTIONS = [
    'value_final',
    'value_mean',
    'slope',
    'volatility',
    'relative_change'
]

# Seuils testabilité
MIN_SAMPLES_PER_GROUP = 2
MIN_GROUPS = 2
MIN_TOTAL_SAMPLES = 10

# =============================================================================
# FILTRAGE ARTEFACTS NUMÉRIQUES (R0 minimal)
# =============================================================================

def is_numeric_valid(obs):
    """
    Détecte artefacts numériques (inf/nan) dans TOUTES les données exploitées.
    
    Vérifie :
    - statistics : initial, final, mean, std, min, max
    - evolution : slope, volatility, relative_change
    """
    # Vérifier statistics
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    
    for metric_stats in statistics.values():
        values_to_check = [
            metric_stats.get('initial'),
            metric_stats.get('final'),
            metric_stats.get('mean'),
            metric_stats.get('std'),      # ← AJOUT
            metric_stats.get('min'),      # ← AJOUT
            metric_stats.get('max'),      # ← AJOUT
        ]
        
        for v in values_to_check:
            if v is not None:
                if np.isinf(v) or np.isnan(v):
                    return False
    
    # Vérifier evolution (projections analysées)  # ← AJOUT COMPLET
    evolution = obs.get('evolution', {})
    
    for metric_evol in evolution.values():
        values_to_check = [
            metric_evol.get('slope'),
            metric_evol.get('volatility'),
            metric_evol.get('relative_change'),
        ]
        
        for v in values_to_check:
            if v is not None:
                if np.isinf(v) or np.isnan(v):
                    return False
    
    return True


def filter_numeric_artifacts(observations):
    """
    Filtre observations avec artefacts numériques.
    
    Log rejets pour traçabilité.
    
    Args:
        observations: liste observations
    
    Returns:
        (valid_obs, rejection_stats)
    """
    valid = []
    rejected_by_test = {}
    
    for obs in observations:
        if is_numeric_valid(obs):
            valid.append(obs)
        else:
            test_name = obs.get('test_name', 'UNKNOWN')
            rejected_by_test[test_name] = rejected_by_test.get(test_name, 0) + 1
    
    total_rejected = len(observations) - len(valid)
    
    stats = {
        'total_observations': len(observations),
        'valid_observations': len(valid),
        'rejected_observations': total_rejected,
        'rejection_rate': total_rejected / len(observations) if observations else 0,
        'rejected_by_test': rejected_by_test,
    }
    
    return valid, stats

def diagnose_numeric_degeneracy(obs):
    """
    Détecte dégénérescences numériques sur PROJECTIONS exploitées par verdict.
    
    Inspecte : value_final, value_mean, slope, volatility, relative_change
    (pas statistics brutes)
    
    Flags diagnostics (non exclusifs) :
    - NEAR_ZERO_VARIANCE : projection quasi constante
    - DEGENERATE_SCALE : ratios extrêmes
    - NUMERIC_COLLAPSE : distribution écrasée
    - EXTREME_MAGNITUDE : valeurs très grandes
    
    Args:
        obs: dict observation
    
    Returns:
        list[str]: flags détectés (peut être vide)
    """
    flags = []
    
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    for metric_name in statistics.keys():
        
        # Récupérer les 5 projections utilisées par verdict
        stat = statistics.get(metric_name, {})
        evol = evolution.get(metric_name, {})
        
        projections = {
            'value_final': stat.get('final'),
            'value_mean': stat.get('mean'),
            'slope': evol.get('slope'),
            'volatility': evol.get('volatility'),
            'relative_change': evol.get('relative_change'),
        }
        
        # Inspecter chaque projection
        for proj_name, value in projections.items():
            
            if value is None:
                continue
            
            # Skip si inf/nan (déjà géré par filtre précédent)
            if np.isinf(value):
                flags.append(f"{metric_name}:{proj_name}:INFINITE_PROJECTION")
                continue  # Skip les autres checks pour cette projection
    
            if np.isnan(value):
                flags.append(f"{metric_name}:{proj_name}:NAN_PROJECTION")
                continue
            
            abs_value = abs(value)
            
            # Flag 1: EXTREME_MAGNITUDE
            # Valeurs très grandes (> 1e50 mais < inf)
            if abs_value > 1e50:
                flags.append(f"{metric_name}:{proj_name}:EXTREME_MAGNITUDE")
        
        # Flags nécessitant comparaison entre projections d'une même métrique
        
        # Collecter valeurs finies de value_final sur toutes métriques (pour variance globale)
        # Note : on ne peut pas détecter variance sans plusieurs observations
        # Donc on se concentre sur les flags par valeur individuelle ci-dessus
    
    return flags


def generate_degeneracy_report(observations):
    """
    Génère rapport diagnostique dégénérescences sur PROJECTIONS.
    
    Args:
        observations: liste observations (non filtrées)
    
    Returns:
        dict: statistiques diagnostiques
    """
    flag_counts = defaultdict(int)
    obs_with_flags = 0
    flags_by_test = defaultdict(lambda: defaultdict(int))
    flags_by_projection = defaultdict(int)
    
    for obs in observations:
        flags = diagnose_numeric_degeneracy(obs)
        
        if flags:
            obs_with_flags += 1
            test_name = obs.get('test_name', 'UNKNOWN')
            
            for flag in flags:
                flag_counts[flag] += 1
                
                # Parser flag: metric:projection:flag_type
                parts = flag.split(':')
                if len(parts) >= 3:
                    projection = parts[1]
                    flag_type = parts[2]
                    flags_by_projection[projection] += 1
                    flags_by_test[test_name][flag_type] += 1
    
    report = {
        'total_observations': len(observations),
        'observations_with_flags': obs_with_flags,
        'flag_rate': obs_with_flags / len(observations) if observations else 0,
        'flag_counts': dict(flag_counts),
        'flags_by_test': {k: dict(v) for k, v in flags_by_test.items()},
        'flags_by_projection': dict(flags_by_projection),
    }
    
    return report


def print_degeneracy_report(report):
    """Affiche rapport diagnostique."""
    
    total = report['total_observations']
    flagged = report['observations_with_flags']
    rate = report['flag_rate']
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC DÉGÉNÉRESCENCES NUMÉRIQUES (projections exploitées)")
    print("=" * 80)
    print(f"Total observations:        {total}")
    print(f"Observations flaggées:     {flagged} ({rate*100:.1f}%)")
    print()
    
    if flagged == 0:
        print("✓ Aucune dégénérescence détectée\n")
        return
    
    # Flags par projection
    print("Dégénérescences par projection (variables analysées):")
    print("-" * 80)
    
    projections = report['flags_by_projection']
    if projections:
        for proj_name in ['value_final', 'value_mean', 'slope', 'volatility', 'relative_change']:
            count = projections.get(proj_name, 0)
            if count > 0:
                percentage = (count / total) * 100
                print(f"  {proj_name:20s} : {count:5d} occurrences ({percentage:5.1f}%)")
    
    print()
    
    # Top flags globaux
    print("Flags les plus fréquents:")
    print("-" * 80)
    
    # Agréger par type de flag
    flag_type_counts = defaultdict(int)
    for flag, count in report['flag_counts'].items():
        parts = flag.split(':')
        flag_type = parts[-1] if parts else flag
        flag_type_counts[flag_type] += count
    
    for flag_type, count in sorted(flag_type_counts.items(), key=lambda x: -x[1])[:5]:
        percentage = (count / total) * 100
        print(f"  {flag_type:25s} : {count:5d} occurrences ({percentage:5.1f}%)")
    
    print()
    
    # Détail par test
    print("Dégénérescences par test:")
    print("-" * 80)
    
    for test_name in sorted(report['flags_by_test'].keys()):
        flags = report['flags_by_test'][test_name]
        total_flags = sum(flags.values())
        print(f"\n{test_name}: {total_flags} flags")
        for flag_type, count in sorted(flags.items(), key=lambda x: -x[1]):
            print(f"  {flag_type:25s} : {count:5d}")
    
    print("\n" + "=" * 80 + "\n")
    
def diagnose_scale_outliers(observations):
    """
    Détecte ruptures d'échelle relatives par test/métrique (en log10).
    
    Critère : valeur > P90 + 5 décades (facteur 1e5)
    Raisonnement relatif, pas seuil absolu.
    
    Args:
        observations: liste observations
    
    Returns:
        dict: rapport ruptures d'échelle
    """
    # Projections à analyser
    projections = ['value_final', 'value_mean', 'slope', 'relative_change']
    
    # Collecter valeurs par (test, metric, projection)
    by_context = defaultdict(list)
    obs_indices = defaultdict(list)
    
    for i, obs in enumerate(observations):
        test_name = obs.get('test_name')
        obs_data = obs.get('observation_data', {})
        statistics = obs_data.get('statistics', {})
        evolution = obs_data.get('evolution', {})
        
        for metric_name in statistics.keys():
            stat = statistics.get(metric_name, {})
            evol = evolution.get(metric_name, {})
            
            values = {
                'value_final': stat.get('final'),
                'value_mean': stat.get('mean'),
                'slope': evol.get('slope'),
                'relative_change': evol.get('relative_change'),
            }
            
            for proj_name, value in values.items():
                if value and np.isfinite(value) and abs(value) > 1e-10:
                    key = (test_name, metric_name, proj_name)
                    log_val = np.log10(abs(value))
                    by_context[key].append(log_val)
                    obs_indices[key].append((i, value))
    
    # Calculer P90 par contexte (min 10 observations)
    thresholds = {}
    for key, log_values in by_context.items():
        if len(log_values) >= 10:
            p90 = np.percentile(log_values, 90)
            thresholds[key] = p90
    
    # Détecter outliers (+5 décades au-dessus P90)
    outliers = defaultdict(list)
    outlier_obs_ids = set()
    
    for key, threshold in thresholds.items():
        test_name, metric_name, proj_name = key
        
        for obs_idx, value in obs_indices[key]:
            log_val = np.log10(abs(value))
            gap = log_val - threshold
            
            if gap > 5.0:  # 5 décades = facteur 1e5
                outliers[key].append({
                    'obs_idx': obs_idx,
                    'value': value,
                    'log_value': log_val,
                    'gap_decades': gap,
                })
                outlier_obs_ids.add(obs_idx)
    
    # Compiler rapport
    report = {
        'total_observations': len(observations),
        'observations_with_outliers': len(outlier_obs_ids),
        'outlier_rate': len(outlier_obs_ids) / len(observations) if observations else 0,
        'contexts_analyzed': len(thresholds),
        'contexts_with_outliers': len(outliers),
        'outliers_by_context': dict(outliers),
        'thresholds': thresholds,
    }
    
    return report


def print_scale_outliers_report(report):
    """Affiche rapport ruptures d'échelle."""
    
    total = report['total_observations']
    flagged = report['observations_with_outliers']
    rate = report['outlier_rate']
    contexts_analyzed = report['contexts_analyzed']
    contexts_with_outliers = report['contexts_with_outliers']
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC RUPTURES D'ÉCHELLE RELATIVES")
    print("=" * 80)
    print(f"Total observations:              {total}")
    print(f"Contextes analysés (test×métrique×proj): {contexts_analyzed}")
    print(f"Observations avec outliers:      {flagged} ({rate*100:.1f}%)")
    print(f"Contextes ayant outliers:        {contexts_with_outliers}")
    print()
    
    if flagged == 0:
        print("✓ Aucune rupture d'échelle détectée\n")
        return
    
    # Top contextes avec le plus d'outliers
    print("Contextes avec ruptures d'échelle (>P90 + 5 décades):")
    print("-" * 80)
    
    outliers_by_context = report['outliers_by_context']
    
    # Trier par nombre d'outliers
    sorted_contexts = sorted(
        outliers_by_context.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for key, outlier_list in sorted_contexts[:10]:  # Top 10
        test_name, metric_name, proj_name = key
        count = len(outlier_list)
        percentage = (count / total) * 100
        
        # Stats sur les gaps
        gaps = [o['gap_decades'] for o in outlier_list]
        max_gap = max(gaps)
        mean_gap = np.mean(gaps)
        
        print(f"\n{test_name}/{metric_name} [{proj_name}]:")
        print(f"  {count:4d} outliers ({percentage:5.2f}%)")
        print(f"  Gap max: +{max_gap:.1f} décades, moyen: +{mean_gap:.1f} décades")
        
        # Exemples (3 pires cas)
        worst_cases = sorted(outlier_list, key=lambda x: x['gap_decades'], reverse=True)[:3]
        if worst_cases:
            print(f"  Pires cas:")
            for case in worst_cases:
                print(f"    obs#{case['obs_idx']:5d}: {case['value']:.2e} "
                      f"(+{case['gap_decades']:.1f} décades)")
    
    print("\n" + "=" * 80 + "\n")
    
    
    
def stratify_by_regime(
    observations: List[dict],
    threshold: float = 1e50
) -> Tuple[List[dict], List[dict]]:
    """
    Stratifie observations en régimes stable/explosif.
    
    Critère : présence valeurs >threshold dans projections exploitées.
    Conserve TOUTES observations (aucun filtrage).
    
    Args:
        observations: Liste observations complètes
        threshold: Seuil magnitude extrême (défaut 1e50)
    
    Returns:
        (obs_stable, obs_explosif)
    """
    stable = []
    explosif = []
    
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {})
        evol = obs_data.get('evolution', {})
        
        has_extreme = False
        
        # Vérifier toutes projections exploitées
        for metric_stats in stats.values():
            for key in ['initial', 'final', 'mean', 'max']:
                val = metric_stats.get(key)
                if val is not None and abs(val) > threshold:
                    has_extreme = True
                    break
            if has_extreme:
                break
        
        if not has_extreme:
            for metric_evol in evol.values():
                for key in ['slope', 'relative_change']:
                    val = metric_evol.get(key)
                    if val is not None and abs(val) > threshold:
                        has_extreme = True
                        break
                if has_extreme:
                    break
        
        if has_extreme:
            explosif.append(obs)
        else:
            stable.append(obs)
    
    return stable, explosif


def analyze_regime(
    observations: List[dict],
    regime_name: str,
    params_config_id: str,
    verdict_config_id: str
) -> dict:
    """
    Pipeline analyse complet sur une strate.
    
    Identique à pipeline global, retourne structure complète.
    """
    df = observations_to_dataframe(observations)
    
    if len(df) < MIN_TOTAL_SAMPLES:
        return {
            'regime': regime_name,
            'n_observations': len(observations),
            'status': 'INSUFFICIENT_DATA',
            'message': f'Moins de {MIN_TOTAL_SAMPLES} observations'
        }
    
    # Analyses identiques pipeline global
    marginal_variance = analyze_marginal_variance(df, FACTORS, PROJECTIONS)
    oriented_interactions = analyze_oriented_interactions(
        df, FACTORS, PROJECTIONS, marginal_variance
    )
    discrimination = analyze_metric_discrimination(df, PROJECTIONS)
    correlations = analyze_metric_correlations(df)
    
    patterns_global, patterns_by_gamma = interpret_patterns(
        df, marginal_variance, oriented_interactions,
        discrimination, correlations
    )
    
    verdict, reason, verdicts_by_gamma = decide_verdict(
        patterns_global, patterns_by_gamma
    )
    
    return {
        'regime': regime_name,
        'n_observations': len(observations),
        'n_rows_df': len(df),
        'status': 'SUCCESS',
        'marginal_variance': marginal_variance,
        'oriented_interactions': oriented_interactions,
        'discrimination': discrimination,
        'correlations': correlations,
        'patterns_global': patterns_global,
        'patterns_by_gamma': patterns_by_gamma,
        'verdict': verdict,
        'reason': reason,
        'verdicts_by_gamma': verdicts_by_gamma
    }
    
    
def generate_stratified_report(
    params_config_id: str,
    verdict_config_id: str,
    results_global: dict,
    results_stable: dict,
    results_explosif: dict,
    output_dir: str = "reports/verdicts"
) -> None:
    """Génère rapport structuré avec 3 strates."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_stratified_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata commune
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'architecture': 'stratified_parallel_regimes',
        'stratification_threshold': 1e50,
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'regimes': {
            'GLOBAL': {
                'n_observations': results_global['n_observations'],
                'description': 'Baseline complète (toutes observations)'
            },
            'STABLE': {
                'n_observations': results_stable['n_observations'],
                'description': 'Régime non-extrême (|projections| < 1e50)'
            },
            'EXPLOSIF': {
                'n_observations': results_explosif['n_observations'],
                'description': 'Régime magnitude extrême (|projections| >= 1e50)'
            }
        }
    }
    
    with open(report_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Rapport humain structuré
    with open(report_dir / 'summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("VERDICT ANALYSIS - STRATIFICATION PARALLÈLE\n")
        f.write(f"{timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write("  3 analyses parallèles (pipeline identique)\n")
        f.write("  Stratification : |projection| >= 1e50\n")
        f.write("  Aucune donnée filtrée (conservation intégrale)\n\n")
        
        # Résumé par régime
        for regime_name, results in [
            ('GLOBAL', results_global),
            ('STABLE', results_stable),
            ('EXPLOSIF', results_explosif)
        ]:
            f.write("="*80 + "\n")
            f.write(f"RÉGIME {regime_name}\n")
            f.write("="*80 + "\n")
            f.write(f"Observations: {results['n_observations']}\n")
            
            if results['status'] != 'SUCCESS':
                f.write(f"Status: {results['status']}\n")
                f.write(f"Message: {results.get('message', 'N/A')}\n\n")
                continue
            
            f.write(f"Verdict: {results['verdict']}\n")
            f.write(f"Raison:  {results['reason']}\n\n")
            
            f.write("PATTERNS DÉTECTÉS:\n")
            for pattern_type, pattern_list in results['patterns_global'].items():
                if pattern_list:
                    f.write(f"  {pattern_type}: {len(pattern_list)} occurrences\n")
            f.write("\n")
    
    # JSON complet par régime
    for regime_name, results in [
        ('global', results_global),
        ('stable', results_stable),
        ('explosif', results_explosif)
    ]:
        if results['status'] == 'SUCCESS':
            report_json = {
                'regime': results['regime'],
                'n_observations': results['n_observations'],
                'verdict': results['verdict'],
                'reason': results['reason'],
                'patterns_global': results['patterns_global'],
                'patterns_by_gamma': results['patterns_by_gamma']
            }
            
            with open(report_dir / f'analysis_{regime_name}.json', 'w') as f:
                json.dump(report_json, f, indent=2)
    
    # CSVs par régime (marginal_variance, interactions, etc.)
    for regime_name, results in [
        ('global', results_global),
        ('stable', results_stable),
        ('explosif', results_explosif)
    ]:
        if results['status'] == 'SUCCESS':
            results['marginal_variance'].to_csv(
                report_dir / f'marginal_variance_{regime_name}.csv',
                index=False
            )
            if not results['oriented_interactions'].empty:
                results['oriented_interactions'].to_csv(
                    report_dir / f'oriented_interactions_{regime_name}.csv',
                    index=False
                )
    
    print(f"\n✓ Rapports stratifiés générés : {report_dir}")
# =============================================================================
# STRUCTURATION
# =============================================================================

# tests/utilities/verdict_engine.py
# SECTION : CHARGEMENT (CORRIGÉ pour double DB)

def load_all_observations(
    params_config_id: str,
    db_results_path: str = './prc_automation/prc_database/prc_r0_results.db',
    db_raw_path: str = './prc_automation/prc_database/prc_r0_raw.db'
) -> List[dict]:
    """
    Charge observations SUCCESS avec métadonnées runs.
    
    ⚠️ DOUBLE CONNEXION : TestObservations (db_results) + Executions (db_raw)
    """
    # 1. Charger observations depuis db_results
    conn_results = sqlite3.connect(db_results_path)
    conn_results.row_factory = sqlite3.Row
    cursor_results = conn_results.cursor()
    
    cursor_results.execute("""
        SELECT 
            observation_id,
            exec_id,
            test_name,
            params_config_id,
            status,
            observation_data,
            computed_at
        FROM TestObservations
        WHERE params_config_id = ?
          AND status = 'SUCCESS'
    """, (params_config_id,))
    
    obs_rows = cursor_results.fetchall()
    conn_results.close()
    
    if not obs_rows:
        raise ValueError(
            f"Aucune observation SUCCESS pour params={params_config_id}"
        )
    
    # 2. Extraire exec_ids uniques
    exec_ids = list(set(row['exec_id'] for row in obs_rows))
    
    # 3. Charger métadonnées Executions depuis db_raw
    conn_raw = sqlite3.connect(db_raw_path)
    conn_raw.row_factory = sqlite3.Row
    cursor_raw = conn_raw.cursor()
    
    # Requête avec placeholders pour IN clause
    placeholders = ','.join('?' * len(exec_ids))
    cursor_raw.execute(f"""
        SELECT 
            id,
            run_id,
            gamma_id,
            d_encoding_id,
            modifier_id,
            seed
        FROM Executions
        WHERE id IN ({placeholders})
    """, exec_ids)
    
    exec_rows = cursor_raw.fetchall()
    conn_raw.close()
    
    # 4. Index executions par id
    executions_by_id = {
        row['id']: {
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_encoding_id': row['d_encoding_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed']
        }
        for row in exec_rows
    }
    
    # 5. Fusionner observations + métadonnées
    observations = []
    for row in obs_rows:
        exec_id = row['exec_id']
        
        if exec_id not in executions_by_id:
            print(f"⚠️ Skip observation {row['observation_id']}: exec_id={exec_id} introuvable dans db_raw")
            continue
        
        exec_meta = executions_by_id[exec_id]
        
        try:
            obs_data = json.loads(row['observation_data'])
            
            observations.append({
                'observation_id': row['observation_id'],
                'exec_id': exec_id,
                'run_id': exec_meta['run_id'],
                'gamma_id': exec_meta['gamma_id'],
                'd_encoding_id': exec_meta['d_encoding_id'],
                'modifier_id': exec_meta['modifier_id'],
                'seed': exec_meta['seed'],
                'test_name': row['test_name'],
                'params_config_id': row['params_config_id'],
                'observation_data': obs_data,
                'computed_at': row['computed_at']
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Skip observation {row['observation_id']}: {e}")
            continue
    
    return observations


def observations_to_dataframe(observations: List[dict]) -> pd.DataFrame:
    """Convertit observations → DataFrame normalisé."""
    rows = []
    
    for obs in observations:
        gamma_id = obs['gamma_id']
        d_encoding_id = obs['d_encoding_id']
        modifier_id = obs['modifier_id']
        seed = obs['seed']
        test_name = obs['test_name']
        params_config_id = obs['params_config_id']
        
        obs_data = obs['observation_data']
        
        if 'statistics' not in obs_data or 'evolution' not in obs_data:
            continue
        
        stats = obs_data['statistics']
        evolution = obs_data['evolution']
        
        for metric_name in stats.keys():
            if metric_name not in evolution:
                continue
            
            metric_stats = stats[metric_name]
            metric_evol = evolution[metric_name]
            
            rows.append({
                # Identifiants
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id,
                'seed': seed,
                'test_name': test_name,
                'params_config_id': params_config_id,
                'metric_name': metric_name,
                
                # Projections numériques
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                
                # Catégorielles
                'transition': metric_evol.get('transition', 'unknown'),
                'trend': metric_evol.get('trend', 'unknown'),
            })
    
    df = pd.DataFrame(rows)
    
    # Nettoyer NaN
    numeric_cols = [
        'value_final', 'value_initial', 'value_mean', 'value_std',
        'slope', 'volatility', 'relative_change'
    ]
    df = df.dropna(subset=numeric_cols, how='all')
    
    return df

# =============================================================================
# UTILITAIRE : CALCUL ETA-SQUARED (η²)
# =============================================================================

def compute_eta_squared(groups: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Calcule eta-squared (η²) : proportion de variance expliquée par groupes.
    
    η² = SSB / SST où :
    - SSB (Sum of Squares Between) : variance entre groupes
    - SSW (Sum of Squares Within) : variance intra-groupes
    - SST (Sum of Squares Total) : SSB + SSW
    
    Args:
        groups: Liste de tableaux numpy (un par groupe)
    
    Returns:
        (eta2, ssb, ssw)
        - eta2 : proportion variance expliquée [0, 1]
        - ssb : somme carrés entre groupes
        - ssw : somme carrés intra-groupes
    
    Examples:
        >>> g1 = np.array([1, 2, 3])
        >>> g2 = np.array([4, 5, 6])
        >>> eta2, ssb, ssw = compute_eta_squared([g1, g2])
        >>> # eta2 proche de 1.0 (groupes bien séparés)
    
    Notes:
        - Retourne (0.0, 0.0, 0.0) si données insuffisantes
        - Protection division par zéro (sst < 1e-10)
        - Filtre groupes vides automatiquement
    """
    # Filtrer groupes vides
    groups_valid = [g for g in groups if len(g) > 0]
    
    if len(groups_valid) < 2:
        return 0.0, 0.0, 0.0
    
    # Concaténer toutes valeurs
    all_values = np.concatenate(groups_valid)
    
    if len(all_values) < 2:
        return 0.0, 0.0, 0.0
    
    # Grand mean (moyenne totale)
    grand_mean = np.mean(all_values)
    
    # SSB : variance expliquée par appartenance groupe
    ssb = sum(
        len(g) * (np.mean(g) - grand_mean)**2
        for g in groups_valid
    )
    
    # SSW : variance résiduelle intra-groupes
    ssw = sum(
        np.sum((g - np.mean(g))**2)
        for g in groups_valid
    )
    
    # SST : variance totale
    sst = ssb + ssw
    
    # η² : proportion variance expliquée
    if sst > 1e-10:
        eta2 = ssb / sst
    else:
        eta2 = 0.0
    
    return eta2, ssb, ssw

# =============================================================================
# ANALYSIS 1 : VARIANCE MARGINALE
# =============================================================================

def analyze_marginal_variance(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str]
) -> pd.DataFrame:
    """
    Analyse variance marginale : chaque facteur pris isolément.
    
    Utilise η² (eta-squared) pour mesurer proportion variance expliquée :
    - η² = SSB / (SSB + SSW)
    - 0 ≤ η² ≤ 1
    - η² proche 1 → facteur très discriminant
    - η² proche 0 → facteur peu informatif
    
    ⚠️ PHASE 2.1 : Correction critique du calcul variance ratio
    Ancien calcul (biaisé) :
        var_between = np.var([mean_1, mean_2, ...])
        var_total = np.var([all_values])
        ratio = var_between / var_total  # ← FAUX
    
    Nouveau calcul (correct) :
        η² = SSB / SST via compute_eta_squared()
    
    Args:
        df: DataFrame observations
        factors: Liste facteurs à analyser
        projections: Liste projections numériques
    
    Returns:
        DataFrame avec colonnes :
        - test_name, metric_name, projection, factor
        - variance_ratio (η²)
        - p_value (Kruskal-Wallis)
        - n_groups, significant
    
    Filtrage testabilité :
        - Contexte global : ≥ MIN_TOTAL_SAMPLES observations
        - Par groupe : ≥ MIN_SAMPLES_PER_GROUP observations
        - Facteur : ≥ MIN_GROUPS niveaux distincts
    """
    results = []
    skipped_testability = 0
    
    for projection in projections:
        for (test_name, metric_name), group in df.groupby(['test_name', 'metric_name']):
            
            # Filtrage testabilité globale
            if len(group) < MIN_TOTAL_SAMPLES:
                skipped_testability += 1
                continue
            
            for factor in factors:
                # Filtrage cardinalité factor
                if group[factor].nunique() < MIN_GROUPS:
                    continue
                
                # Grouper par factor
                factor_groups = [
                    g[projection].dropna().values 
                    for name, g in group.groupby(factor)
                    if len(g.dropna()) >= MIN_SAMPLES_PER_GROUP
                ]
                
                if len(factor_groups) < MIN_GROUPS:
                    continue
                
                # Test Kruskal-Wallis
                try:
                    statistic, p_value = kruskal(*factor_groups)
                except (ValueError, Exception):
                    continue
                
                
                # Calcul η² via fonction utilitaire
                variance_ratio, ssb, ssw = compute_eta_squared(factor_groups)
                
                results.append({
                    'test_name': test_name,
                    'metric_name': metric_name,
                    'projection': projection,
                    'factor': factor,
                    'variance_ratio': variance_ratio,
                    'p_value': p_value,
                    'n_groups': len(factor_groups),
                    'significant': p_value < 0.05
                })
    
    print(f"   ⊘ Skipped {skipped_testability} groups (min_samples)")
    
    return pd.DataFrame(results).sort_values('variance_ratio', ascending=False)


# =============================================================================
# ANALYSIS 2 : INTERACTIONS ORIENTÉES
# =============================================================================

def analyze_oriented_interactions(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str],
    marginal_variance: pd.DataFrame,
    min_interaction_strength: float = 2.0
) -> pd.DataFrame:
    """
     Détecte interactions ORIENTÉES : A|B distinct de B|A.
    
    Interaction vraie détectée si :
    1. VR(A|B=b) significatif (η² > 0.3, p < 0.05)
    2. VR(A|B=b) >> VR(A) marginal (ratio > min_interaction_strength)
    3. Testabilité robuste (≥3 niveaux, ≥5 obs/groupe)
    
    ⚠️ PHASE 2.2 : Critères testabilité renforcés
    - vr_marginal ≥ 0.1 (marginal substantiel)
    - n_groups ≥ 3 (interaction nécessite ≥3 niveaux)
    - min_group_size ≥ 5 (robustesse statistique)
    
    Différence vs combinations() :
    - Paires orientées : (A, B) ET (B, A) testées séparément
    - Total : len(factors) × (len(factors) - 1) paires
    - Interaction orientée : effet de A change selon contexte B
    
    Args:
        df: DataFrame observations
        factors: Liste facteurs
        projections: Projections numériques
        marginal_variance: Résultats analyze_marginal_variance()
        min_interaction_strength: Seuil ratio VR_cond / VR_marg
    
    Returns:
        DataFrame interactions détectées avec colonnes :
        - factor_varying, factor_context, context_value
        - test_name, metric_name, projection
        - vr_conditional, vr_marginal, interaction_strength
        - p_value, n_groups
    
    Examples:
        Interaction détectée :
        factor_varying = "seed"
        factor_context = "modifier"
        context_value = "M3"
        vr_conditional = 0.62
        vr_marginal = 0.15
        interaction_strength = 4.13
        
        Interprétation : 
        "L'effet de seed sur la métrique devient 4× plus fort 
         quand modifier=M3 (62% variance vs 15% normalement)"
    """
    results = []
    skipped_testability = 0
    
    # Index variance marginale pour lookup rapide
    marginal_index = {}
    for _, row in marginal_variance.iterrows():
        key = (row['test_name'], row['metric_name'], row['projection'], row['factor'])
        marginal_index[key] = row['variance_ratio']
    
    # PAIRES ORIENTÉES (pas combinations)
    print(f"   Génération {len(factors) * (len(factors) - 1)} paires orientées...")
    
    for factor_varying in factors:
        if factor_varying == 'test_name':
            continue  # test_name n'est jamais un facteur "actif"

        for factor_context in factors:
            if factor_context == factor_varying:
                continue

            for (test_name, metric_name), tm_group in df.groupby(['test_name', 'metric_name']):

                if len(tm_group) < MIN_TOTAL_SAMPLES:
                    continue

                for projection in projections:

                    for context_value, context_group in tm_group.groupby(factor_context):

                        if len(context_group) < MIN_TOTAL_SAMPLES:
                            continue


                        
                        # Variance factor_varying dans ce contexte                        
                        varying_groups = [
                            g[projection].dropna().values
                            for name, g in context_group.groupby(factor_varying)
                            if len(g.dropna()) >= MIN_SAMPLES_PER_GROUP
                        ]
                        
                        if len(varying_groups) < MIN_GROUPS:
                            continue
                        
                        # Test significativité
                        try:
                            statistic, p_value = kruskal(*varying_groups)
                        except:
                            continue
                        
                        # Calcul η² conditionnel via fonction utilitaire
                        vr_conditional, ssb, ssw = compute_eta_squared(varying_groups)
                        
                        # Récupérer variance marginale pour comparaison
                        marginal_key = (test_name, metric_name, projection, factor_varying)
                        vr_marginal = marginal_index.get(marginal_key, 0.0)
                        
                        # PHASE 2.2 : Marginal doit être substantiel (≥10%)
                        if vr_marginal < 0.1:
                            continue
                        
                        # Calcul force interaction
                        interaction_strength = vr_conditional / vr_marginal
                        
                        # Interaction = effet change selon contexte
                        # Critères testabilité renforcés (Phase 2.2)
                        min_group_size = min(len(g) for g in varying_groups)
                        
                        if (vr_conditional > 0.3 and 
                            vr_marginal > 0.1 and  # ← Déjà filtré avant, mais explicit
                            p_value < 0.05 and 
                            interaction_strength > min_interaction_strength and
                            len(varying_groups) >= 3 and  # ← Au moins 3 niveaux factor_varying
                            min_group_size >= 5):  # ← Robustesse statistique
                            
                            results.append({
                                'factor_varying': factor_varying,
                                'factor_context': factor_context,
                                'context_value': str(context_value),
                                'test_name': test_name,
                                'metric_name': metric_name,
                                'projection': projection,
                                'vr_conditional': vr_conditional,
                                'vr_marginal': vr_marginal,
                                'interaction_strength': interaction_strength,
                                'p_value': p_value,
                                'n_groups': len(varying_groups)
                            })
    
    print(f"   ⊘ Skipped {skipped_testability} contexts (testability)")
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 3 : DISCRIMINANCE
# =============================================================================

def analyze_metric_discrimination(
    df: pd.DataFrame,
    projections: List[str]
) -> pd.DataFrame:
    """Détecte métriques non discriminantes (CV < 0.1)."""
    results = []
    
    for projection in projections:
        for (test_name, metric_name), group in df.groupby(['test_name', 'metric_name']):
            
            values = group[projection].dropna().values
            
            if len(values) < 5:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if abs(mean_val) < 1e-10:
                cv = np.nan
            else:
                cv = std_val / abs(mean_val)
            
            results.append({
                'test_name': test_name,
                'metric_name': metric_name,
                'projection': projection,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'n_observations': len(values),
                'non_discriminant': cv < 0.1 if not np.isnan(cv) else False
            })
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 4 : CORRÉLATIONS
# =============================================================================

def analyze_metric_correlations(
    df: pd.DataFrame,
    projection: str = 'value_final',
    threshold: float = 0.8
) -> pd.DataFrame:
    """Détecte corrélations fortes entre métriques."""
    results = []
    
    for test_name, test_group in df.groupby('test_name'):
        
        # Pivot
        pivot = test_group.pivot_table(
            index=['gamma_id', 'd_encoding_id', 'modifier_id', 'seed'],
            columns='metric_name',
            values=projection
        )
        
        metrics = pivot.columns.tolist()
        
        if len(metrics) < 2:
            continue
        
        # Corrélations pairwise
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                
                valid = pivot[[metric1, metric2]].dropna()
                
                if len(valid) < 5:
                    continue
                
                try:
                    corr, p_value = spearmanr(valid[metric1], valid[metric2])
                except:
                    continue
                
                if abs(corr) > threshold:
                    results.append({
                        'test_name': test_name,
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_observations': len(valid)
                    })
    
    return pd.DataFrame(results)


# =============================================================================
# INTERPRETATION
# =============================================================================

def interpret_patterns(
    df: pd.DataFrame,
    marginal_variance: pd.DataFrame,
    oriented_interactions: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame
) -> Tuple[dict, dict]:
    """
    Synthèse patterns : GLOBAL + PAR GAMMA.
    
    Returns:
        (patterns_global, patterns_by_gamma)
    """
    patterns_global = {
        'marginal_dominant': [],
        'oriented_interactions': [],
        'non_discriminant': [],
        'redundant': []
    }
    
    # 1. Variance marginale dominante
    dominant = marginal_variance[
        (marginal_variance['variance_ratio'] > 0.5) &
        (marginal_variance['significant'])
    ]
    
    if not dominant.empty:
        for factor in dominant['factor'].unique():
            subset = dominant[dominant['factor'] == factor]
            patterns_global['marginal_dominant'].append({
                'factor': factor,
                'n_metrics': len(subset),
                'projections': subset['projection'].unique().tolist(),
                'max_variance_ratio': float(subset['variance_ratio'].max())
            })
    
    # 2. Interactions orientées (VRAIES)
    if not oriented_interactions.empty:
        # Grouper par paire orientée
        for (fv, fc), group in oriented_interactions.groupby(['factor_varying', 'factor_context']):
            patterns_global['oriented_interactions'].append({
                'interaction': f"{fv} | {fc}",  # Notation orientée
                'n_cases': len(group),
                'contexts_affected': group['context_value'].unique().tolist()[:5],
                'max_strength': float(group['interaction_strength'].max()),
                'examples': group.nlargest(3, 'interaction_strength')[
                    ['test_name', 'metric_name', 'projection', 'context_value', 'interaction_strength']
                ].to_dict('records')
            })
    
    # 3. Non discriminant
    non_disc = discrimination[discrimination['non_discriminant']]
    if not non_disc.empty:
        patterns_global['non_discriminant'].append({
            'n_metrics': len(non_disc),
            'projections': non_disc['projection'].unique().tolist()
        })
    
    # 4. Redondant
    if not correlations.empty:
        patterns_global['redundant'].append({
            'n_pairs': len(correlations)
        })
    
    # =========================================================================
    # PATTERNS PAR GAMMA (drill-down STRICT)
    # =========================================================================
    # TODO Phase 3 : Remplacer par gamma_profiling.profile_all_gammas()
    # Le drill-down actuel filtre les patterns globaux (pas recalcul réel)
    # Désactivé en attendant implémentation Phase 3 (profiling gamma)
    
    patterns_by_gamma = {}

    for gamma_id in df['gamma_id'].unique():
    
        # Sous-ensemble STRICT
        gamma_df = df[df['gamma_id'] == gamma_id]
    
        # RECALCUL analyses sur ce gamma uniquement
        gamma_marginal = analyze_marginal_variance(
            gamma_df,
            [f for f in FACTORS if f != 'gamma_id'],  # Exclure gamma
            PROJECTIONS
        )
    
        patterns_gamma = {
            'marginal_dominant': [],
            'oriented_interactions': []
        }
        # Dominant dans gamma
        dominant_gamma = gamma_marginal[
            (gamma_marginal['variance_ratio'] > 0.5) &
            (gamma_marginal['significant'])
        ]
        
        if not dominant_gamma.empty:
            for factor in dominant_gamma['factor'].unique():
                subset = dominant_gamma[dominant_gamma['factor'] == factor]
                patterns_gamma['marginal_dominant'].append({
                    'factor': factor,
                    'n_metrics': len(subset)
                })
        
        # Interactions dans gamma (simplifié pour drill-down)
        # Protection contre oriented_interactions vide
        if not oriented_interactions.empty:
            gamma_interactions = oriented_interactions[
                oriented_interactions['test_name'].isin(gamma_df['test_name'].unique())
            ]
    
            if not gamma_interactions.empty:
                for (fv, fc), group in gamma_interactions.groupby(['factor_varying', 'factor_context']):
                    if fv != 'gamma_id' and fc != 'gamma_id':
                        patterns_gamma['oriented_interactions'].append({
                            'interaction': f"{fv} | {fc}",
                            'n_cases': len(group)
                        })  
        
        patterns_by_gamma[gamma_id] = patterns_gamma
        
    # NOTE : Les analyses par gamma seront dans gamma_profiling.py
    # qui recalculera TOUT (marginal + interactions) par gamma  
    return patterns_global, patterns_by_gamma


# =============================================================================
# VERDICT
# =============================================================================

def decide_verdict(
    patterns_global: dict,
    patterns_by_gamma: dict
) -> Tuple[str, str, dict]:
    """Décision verdict : GLOBAL + PAR GAMMA."""
    
    has_critical = any(len(v) > 0 for v in patterns_global.values())

    
    if not has_critical:
        verdict_global = "SURVIVES[R0]"
        reason_global = "Aucun pattern pathologique systématique détecté."
    else:
        reasons = []
        
        if patterns_global['marginal_dominant']:
            reasons.append(f"{len(patterns_global['marginal_dominant'])} factors dominants")
        
        if patterns_global['oriented_interactions']:
            reasons.append(f"{len(patterns_global['oriented_interactions'])} interactions vraies")
        
        if patterns_global['non_discriminant']:
            p = patterns_global['non_discriminant'][0]
            reasons.append(f"{p['n_metrics']} métriques non discriminantes")
        
        if patterns_global['redundant']:
            p = patterns_global['redundant'][0]
            reasons.append(f"{p['n_pairs']} paires redondantes")
        
        reason_global = " | ".join(reasons)
        verdict_global = "WIP[R0-open]"
    
    # Verdicts par gamma
    # TODO Phase 3 : Verdicts par gamma via gamma_profiling
    verdicts_by_gamma = {}
    
    if patterns_by_gamma:  # Si non vide (future Phase 3)
        for gamma_id, patterns_gamma in patterns_by_gamma.items():
            for gamma_id, patterns_gamma in patterns_by_gamma.items():
                critical_gamma = sum(len(v) for v in patterns_gamma.values() if v)
        
                if critical_gamma == 0:
                    verdict_gamma = "SURVIVES[R0]"
                    reason_gamma = "Aucun pattern spécifique."
                else:
                    reasons_gamma = []
                    if patterns_gamma['marginal_dominant']:
                        reasons_gamma.append(f"{len(patterns_gamma['marginal_dominant'])} factors dominants")
                    if patterns_gamma['oriented_interactions']:
                        reasons_gamma.append(f"{len(patterns_gamma['oriented_interactions'])} interactions")
            
                    reason_gamma = " | ".join(reasons_gamma)
                    verdict_gamma = "WIP[R0-open]"
            
                verdicts_by_gamma[gamma_id] = {
                    'verdict': verdict_gamma,
                    'reason': reason_gamma,
                    'patterns': patterns_gamma
                }
    else:
        # Phase 3 en attente : pas de verdicts individuels
        pass
        
    return verdict_global, reason_global, verdicts_by_gamma


# =============================================================================
# RAPPORTS
# =============================================================================

def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    df: pd.DataFrame,
    marginal_variance: pd.DataFrame,
    oriented_interactions: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame,
    patterns_global: dict,
    patterns_by_gamma: dict,
    verdict_global: str,
    reason_global: str,
    verdicts_by_gamma: dict,
    output_dir: str = "reports/verdicts"
) -> None:
    """Génère rapports complets."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_analysis_full"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    n_oriented_pairs = len(FACTORS) * (len(FACTORS) - 1)
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'architecture': 'oriented_interactions',
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'factors_analyzed': FACTORS,
        'projections_analyzed': PROJECTIONS,
        'n_observations': len(df),
        'n_factors': len(FACTORS),
        'n_oriented_pairs': n_oriented_pairs,
        'testability_thresholds': {
            'min_samples_per_group': MIN_SAMPLES_PER_GROUP,
            'min_groups': MIN_GROUPS,
            'min_total_samples': MIN_TOTAL_SAMPLES
        },
        'limitations': {
            'variance_ratio': 'Heuristique R0',
            'p_values': 'Non corrigées multiplicité',
            'interaction_strength': 'Ratio VR_conditionnel/VR_marginal (seuil=2.0)'
        }
    }
    
    with open(report_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Rapport humain
    with open(report_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"VERDICT ANALYSIS - INTERACTIONS ORIENTÉES\n")
        f.write(f"{timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write(f"  Factors analysés : {', '.join(FACTORS)}\n")
        f.write(f"  Paires orientées : {n_oriented_pairs}\n")
        f.write(f"  Projections : {', '.join(PROJECTIONS)}\n")
        f.write(f"  Testabilité : min_samples={MIN_SAMPLES_PER_GROUP}, min_groups={MIN_GROUPS}\n\n")
        
        f.write("CORRECTIONS CRITIQUES APPLIQUÉES:\n")
        f.write("  ✓ Paires orientées (permutations pas combinations)\n")
        f.write("  ✓ Interaction = VR_conditionnel >> VR_marginal\n")
        f.write("  ✓ Filtrage testabilité explicite\n")
        f.write("  ✓ params_config_id exclu (corrélation)\n")
        f.write("  ✓ Drill-down gamma recalculé\n\n")
        
        f.write("="*80 + "\n")
        f.write("VERDICT GLOBAL\n")
        f.write("="*80 + "\n")
        f.write(f"Verdict: {verdict_global}\n")
        f.write(f"Raison:  {reason_global}\n\n")
        
        f.write("PATTERNS GLOBAUX:\n")
        for pattern_type, pattern_list in patterns_global.items():
            if pattern_list:
                f.write(f"  {pattern_type}: {len(pattern_list)} occurrences\n")
        f.write("\n")
        
        if patterns_by_gamma:
            f.write("="*80 + "\n")
            f.write("VERDICTS PAR GAMMA (drill-down recalculé)\n")
            f.write("="*80 + "\n\n")
            
            for gamma_id, verdict_info in verdicts_by_gamma.items():
                f.write(f"--- {gamma_id} ---\n")
                f.write(f"Verdict: {verdict_info['verdict']}\n")
                f.write(f"Raison:  {verdict_info['reason']}\n\n")
        
        else:
            f.write("\n")
            f.write("NOTE : Profiling par gamma disponible en Phase 3\n")
            f.write("  Module gamma_profiling.py en développement\n")        
        
        f.write("="*80 + "\n")
    
    # Rapport JSON
    report_json = {
        'metadata': metadata,
        'verdict_global': {
            'verdict': verdict_global,
            'reason': reason_global,
            'patterns': patterns_global
        },
        'verdicts_by_gamma': verdicts_by_gamma
    }
    
    with open(report_dir / 'analysis_complete.json', 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
    
    # CSV
    marginal_variance.to_csv(report_dir / 'marginal_variance.csv', index=False)
    if not oriented_interactions.empty:
        oriented_interactions.to_csv(report_dir / 'oriented_interactions.csv', index=False)
    discrimination.to_csv(report_dir / 'discrimination.csv', index=False)
    if not correlations.empty:
        correlations.to_csv(report_dir / 'correlations.csv', index=False)
    
    print(f"\n✓ Rapports générés : {report_dir}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None:
    """Pipeline complet verdict exploratoire CORRIGÉ."""
    
    print(f"\n{'='*70}")
    print(f"VERDICT ANALYSIS - INTERACTIONS ORIENTÉES")
    print(f"{'='*70}\n")
    
    n_oriented_pairs = len(FACTORS) * (len(FACTORS) - 1)
    
    print(f"Factors analysés : {', '.join(FACTORS)}")
    print(f"Paires orientées : {n_oriented_pairs}")
    print(f"Projections : {', '.join(PROJECTIONS)}")
    print(f"Testabilité : min_samples={MIN_SAMPLES_PER_GROUP}, min_groups={MIN_GROUPS}\n")
    
    # 1. Config
    print("1. Chargement config...")
    loader = get_loader()
    verdict_config = loader.load('verdict', verdict_config_id)
    
    # 2. Observations
    print("2. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations")
    
    # Filtrage artefacts numériques (inf/nan)
    observations, rejection_stats = filter_numeric_artifacts(observations)
     
    # Log rejets
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        print(f"      Détail par test :")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"        {test}: {count} invalides")
    print()
    
    # ← AJOUTER ICI (avant le "3. Structuration DataFrame...")
    # Diagnostic dégénérescences numériques (informatif uniquement)
    degeneracy_report = generate_degeneracy_report(observations)
    print_degeneracy_report(degeneracy_report)
    
    # Diagnostic ruptures d'échelle relatives (contextuel)
    scale_report = diagnose_scale_outliers(observations)
    print_scale_outliers_report(scale_report)
     
    
    # NOUVEAU : Stratification
    print("4. Stratification régimes...")
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE    : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF  : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    # Analyses parallèles
    print("\n5. Analyses statistiques...")
    
    print("   [GLOBAL] Baseline complète...")
    results_global = analyze_regime(
        observations, 'GLOBAL', 
        params_config_id, verdict_config_id
    )
    
    print("   [STABLE] Régime non-extrême...")
    results_stable = analyze_regime(
        obs_stable, 'STABLE',
        params_config_id, verdict_config_id
    )
    
    print("   [EXPLOSIF] Régime magnitude extrême...")
    results_explosif = analyze_regime(
        obs_explosif, 'EXPLOSIF',
        params_config_id, verdict_config_id
    )
    
    # Génération rapports stratifiés
    print("\n6. Génération rapports stratifiés...")
    generate_stratified_report(
        params_config_id, verdict_config_id,
        results_global, results_stable, results_explosif
    )

    
    # 3. DataFrame
    print("3. Structuration DataFrame...")
    #df = observations_to_dataframe(observations)
    #print(f"   ✓ {len(df)} lignes")
    
    # 4. Analyses
    print("4. Analyses statistiques...")
    
    # 4a. Variance marginale
    #marginal_variance = analyze_marginal_variance(df, FACTORS, PROJECTIONS)
    #print(f"   ✓ Variance marginale: {len(marginal_variance)} résultats")
    
    # 4b. Interactions orientées (CORRIGÉ)
    #oriented_interactions = analyze_oriented_interactions(
    #    df, FACTORS, PROJECTIONS, marginal_variance
    #)
    #print(f"   ✓ Interactions orientées: {len(oriented_interactions)} détectées")
    
    # 4c. Discrimination
    #discrimination = analyze_metric_discrimination(df, PROJECTIONS)
    #print(f"   ✓ Discrimination: {len(discrimination)} métriques")
    
    # 4d. Corrélations
    #correlations = analyze_metric_correlations(df)
    #print(f"   ✓ Corrélations: {len(correlations)} paires")
    
    # 5. Interpretation
    print("5. Interprétation patterns...")
    #patterns_global, patterns_by_gamma = interpret_patterns(
    #    df,
     #   marginal_variance,
     #   oriented_interactions,
     #   discrimination,
     #   correlations
    #)
    
    # 6. Verdict
    print("6. Décision verdict...")
    #verdict_global, reason_global, verdicts_by_gamma = decide_verdict(
    #    patterns_global,
    #    patterns_by_gamma
    #)
    #print(f"   → Global: {verdict_global}")
    
    # 7. Rapports
    print("7. Génération rapports...")
    #generate_verdict_report(
    #    params_config_id,
    #    verdict_config_id,
    #    df,
    #    marginal_variance,
    #    oriented_interactions,
    #    discrimination,
    #    correlations,
     #   patterns_global,
     #   patterns_by_gamma,
    #    verdict_global,
    #    reason_global,
    #    verdicts_by_gamma
    #)
    
    print(f"\n{'='*70}")
    #print(f"VERDICT: {verdict_global}")
    print(f"{'='*70}\n")