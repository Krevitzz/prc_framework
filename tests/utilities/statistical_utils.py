# tests/utilities/statistical_utils.py
"""
Statistical Utilities - Outils statistiques réutilisables.

RESPONSABILITÉS :
- Calculs variance (η², SSB/SSW)
- Filtrage artefacts numériques (inf/nan)
- Diagnostics dégénérescences projections
- Diagnostics ruptures échelle relatives
- Tests statistiques standards (Kruskal-Wallis)

UTILISATEURS :
- verdict_engine.py (analyses globales)
- gamma_profiling.py (agrégations)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# CALCULS VARIANCE (η²)
# =============================================================================

def compute_eta_squared(groups: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Calcule eta-squared (η²) : proportion variance expliquée par groupes.
    
    FORMULE :
    η² = SSB / (SSB + SSW)
    - SSB (Sum of Squares Between) : variance entre groupes
    - SSW (Sum of Squares Within) : variance intra-groupes
    - SST (Sum of Squares Total) : SSB + SSW
    
    Args:
        groups: Liste tableaux numpy (un par groupe)
    
    Returns:
        (eta2, ssb, ssw)
        - eta2 : proportion variance expliquée [0, 1]
        - ssb : somme carrés entre groupes
        - ssw : somme carrés intra-groupes
    
    Examples:
        >>> g1 = np.array([1, 2, 3])
        >>> g2 = np.array([4, 5, 6])
        >>> eta2, ssb, ssw = compute_eta_squared([g1, g2])
        >>> eta2  # Proche de 1.0 (groupes bien séparés)
        0.95
    
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


def kruskal_wallis_test(groups: List[np.ndarray]) -> Tuple[float, float]:
    """
    Test Kruskal-Wallis avec gestion erreurs.
    
    Args:
        groups: Liste tableaux numpy (un par groupe)
    
    Returns:
        (statistic, p_value)
    
    Raises:
        ValueError: Si moins de 2 groupes ou données insuffisantes
    
    Examples:
        >>> g1 = np.array([1, 2, 3])
        >>> g2 = np.array([4, 5, 6])
        >>> stat, pval = kruskal_wallis_test([g1, g2])
        >>> pval < 0.05  # Différence significative
        True
    """
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis nécessite au moins 2 groupes")
    
    # Filtrer groupes vides
    groups_valid = [g for g in groups if len(g) > 0]
    
    if len(groups_valid) < 2:
        raise ValueError("Moins de 2 groupes non vides")
    
    try:
        statistic, p_value = kruskal(*groups_valid)
        return float(statistic), float(p_value)
    except (ValueError, Exception) as e:
        raise ValueError(f"Erreur Kruskal-Wallis: {e}")


# =============================================================================
# FILTRAGE ARTEFACTS NUMÉRIQUES
# =============================================================================

def is_numeric_valid(obs: dict) -> bool:
    """
    Détecte artefacts numériques (inf/nan) dans projections exploitées.
    
    VÉRIFIE :
    - statistics : initial, final, mean, std, min, max
    - evolution : slope, volatility, relative_change
    
    Args:
        obs: Observation dict avec observation_data
    
    Returns:
        True si aucun artefact détecté
    
    Notes:
        - Fonction privée (usage interne filter_numeric_artifacts)
        - Vérifie TOUTES projections utilisées par verdict_engine
    """
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    # Vérifier statistics
    for metric_stats in statistics.values():
        values_to_check = [
            metric_stats.get('initial'),
            metric_stats.get('final'),
            metric_stats.get('mean'),
            metric_stats.get('std'),
            metric_stats.get('min'),
            metric_stats.get('max'),
        ]
        
        for v in values_to_check:
            if v is not None:
                if np.isinf(v) or np.isnan(v):
                    return False
    
    # Vérifier evolution
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


def filter_numeric_artifacts(observations: List[dict]) -> Tuple[List[dict], dict]:
    """
    Filtre observations avec artefacts numériques.
    
    Log rejets pour traçabilité (par test).
    
    Args:
        observations: Liste observations
    
    Returns:
        (valid_obs, rejection_stats)
        
        rejection_stats :
        {
            'total_observations': int,
            'valid_observations': int,
            'rejected_observations': int,
            'rejection_rate': float,
            'rejected_by_test': dict
        }
    
    Examples:
        >>> valid, stats = filter_numeric_artifacts(observations)
        >>> stats['rejection_rate']
        0.05  # 5% observations rejetées
        >>> stats['rejected_by_test']
        {'TOP-001': 12, 'SPE-001': 3}
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


# =============================================================================
# DIAGNOSTICS DÉGÉNÉRESCENCES
# =============================================================================

def diagnose_numeric_degeneracy(obs: dict) -> List[str]:
    """
    Détecte dégénérescences numériques sur projections exploitées.
    
    INSPECTE :
    - value_final, value_mean, slope, volatility, relative_change
    
    FLAGS DÉTECTÉS (non exclusifs) :
    - INFINITE_PROJECTION : inf détecté
    - NAN_PROJECTION : nan détecté
    - EXTREME_MAGNITUDE : |valeur| > 1e50
    
    Args:
        obs: Observation dict
    
    Returns:
        Liste flags format "metric:projection:flag_type"
        Exemple : ['asymmetry:value_final:EXTREME_MAGNITUDE']
    
    Notes:
        - Flags par projection (pas global)
        - inf/nan normalement filtrés avant (filter_numeric_artifacts)
        - EXTREME_MAGNITUDE : valeurs très grandes mais finies
    """
    flags = []
    
    obs_data = obs.get('observation_data', {})
    statistics = obs_data.get('statistics', {})
    evolution = obs_data.get('evolution', {})
    
    for metric_name in statistics.keys():
        # Récupérer projections exploitées
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
            
            # Flags artefacts
            if np.isinf(value):
                flags.append(f"{metric_name}:{proj_name}:INFINITE_PROJECTION")
                continue
            
            if np.isnan(value):
                flags.append(f"{metric_name}:{proj_name}:NAN_PROJECTION")
                continue
            
            # Flag magnitude extrême (> 1e50 mais < inf)
            if abs(value) > 1e50:
                flags.append(f"{metric_name}:{proj_name}:EXTREME_MAGNITUDE")
    
    return flags


def generate_degeneracy_report(observations: List[dict]) -> dict:
    """
    Génère rapport diagnostique dégénérescences.
    
    AGRÉGATIONS :
    - Comptage flags par type
    - Comptage observations flaggées
    - Répartition par test
    - Répartition par projection
    
    Args:
        observations: Liste observations (non filtrées)
    
    Returns:
        {
            'total_observations': int,
            'observations_with_flags': int,
            'flag_rate': float,
            'flag_counts': dict,
            'flags_by_test': dict,
            'flags_by_projection': dict
        }
    
    Examples:
        >>> report = generate_degeneracy_report(observations)
        >>> report['flag_rate']
        0.12  # 12% observations avec flags
        >>> report['flags_by_projection']['value_final']
        45  # 45 occurrences flags sur value_final
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


def print_degeneracy_report(report: dict) -> None:
    """
    Affiche rapport diagnostique dégénérescences (stdout).
    
    FORMAT :
    - Header avec totaux
    - Dégénérescences par projection (5 projections analysées)
    - Top 5 flags les plus fréquents
    - Détail par test
    
    Args:
        report: Retour generate_degeneracy_report()
    """
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


# =============================================================================
# DIAGNOSTICS RUPTURES ÉCHELLE
# =============================================================================

def diagnose_scale_outliers(observations: List[dict]) -> dict:
    """
    Détecte ruptures d'échelle relatives par contexte (test×metric×projection).
    
    CRITÈRE :
    - Valeur > P90 + 5 décades (facteur 1e5)
    - Raisonnement relatif (pas seuil absolu)
    - Contextuel (P90 calculé par test/métrique/projection)
    
    PROJECTIONS ANALYSÉES :
    - value_final, value_mean, slope, relative_change
    
    Args:
        observations: Liste observations
    
    Returns:
        {
            'total_observations': int,
            'observations_with_outliers': int,
            'outlier_rate': float,
            'contexts_analyzed': int,
            'contexts_with_outliers': int,
            'outliers_by_context': dict,
            'thresholds': dict
        }
    
    Notes:
        - Min 10 observations par contexte pour calculer P90
        - Outliers stockent gap en décades (log10)
        - Clés contexte : (test_name, metric_name, proj_name)
    """
    projections = ['value_final', 'value_mean', 'slope', 'relative_change']
    
    # Collecter valeurs par contexte
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
        for obs_idx, value in obs_indices[key]:
            log_val = np.log10(abs(value))
            gap = log_val - threshold
            
            if gap > 5.0:
                outliers[key].append({
                    'obs_idx': obs_idx,
                    'value': value,
                    'log_value': log_val,
                    'gap_decades': gap,
                })
                outlier_obs_ids.add(obs_idx)
    
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


def print_scale_outliers_report(report: dict) -> None:
    """
    Affiche rapport ruptures d'échelle (stdout).
    
    FORMAT :
    - Header avec totaux
    - Top 10 contextes avec le plus d'outliers
    - Pour chaque contexte : stats gaps + pires cas
    
    Args:
        report: Retour diagnose_scale_outliers()
    """
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
    
    # Top contextes
    print("Contextes avec ruptures d'échelle (>P90 + 5 décades):")
    print("-" * 80)
    
    outliers_by_context = report['outliers_by_context']
    
    # Trier par nombre d'outliers
    sorted_contexts = sorted(
        outliers_by_context.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for key, outlier_list in sorted_contexts[:10]:
        test_name, metric_name, proj_name = key
        count = len(outlier_list)
        percentage = (count / total) * 100
        
        # Stats gaps
        gaps = [o['gap_decades'] for o in outlier_list]
        max_gap = max(gaps)
        mean_gap = np.mean(gaps)
        
        print(f"\n{test_name}/{metric_name} [{proj_name}]:")
        print(f"  {count:4d} outliers ({percentage:5.2f}%)")
        print(f"  Gap max: +{max_gap:.1f} décades, moyen: +{mean_gap:.1f} décades")
        
        # Pires cas (3 exemples)
        worst_cases = sorted(outlier_list, key=lambda x: x['gap_decades'], reverse=True)[:3]
        if worst_cases:
            print(f"  Pires cas:")
            for case in worst_cases:
                print(f"    obs#{case['obs_idx']:5d}: {case['value']:.2e} "
                      f"(+{case['gap_decades']:.1f} décades)")
    
    print("\n" + "=" * 80 + "\n")