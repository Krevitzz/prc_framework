# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.5 - Analyse exploratoire statistique.

MÉTHODOLOGIE :
- Tests non-paramétriques (Kruskal-Wallis, Spearman)
- Heuristiques variance_ratio R0 (documentées, non standards)
- Analyses multi-projections (final, mean, slope, volatility)
- Tests statistiques NON CORRIGÉS pour multiplicité (exploration R0)

ARCHITECTURE :
1. Structuration : observations → DataFrame normalisé
2. Analysis : variance par TOUS factors (gamma, D, modifier, seed, test)
3. Interpretation : patterns globaux + drill-down par gamma
4. Verdict : global + par gamma
5. Rapports : résultats globaux + sections par gamma

FACTORS ANALYSÉS :
- gamma_id : Mécanisme testé
- d_encoding_id : Encodage dissymétrie
- modifier_id : Perturbations
- seed : Initialisation stochastique
- test_name : Protocole observation

LIMITATIONS CONNUES :
- Variance ratio : heuristique, pas décomposition ANOVA
- Seuils fixes (0.3, 0.1, 0.8) : arbitraires, non adaptatifs
- P-values non corrigées : faux positifs attendus
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, spearmanr
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import warnings

from .config_loader import get_loader

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# STRUCTURATION : OBSERVATIONS → DATAFRAME
# =============================================================================

def load_all_observations(
    params_config_id: str,
    db_path: str = 'prc_database/prc_r0_results.db'
) -> List[dict]:
    """
    Charge observations SUCCESS uniquement.
    
    Args:
        params_config_id: Config params utilisée
        db_path: Chemin db_results
    
    Returns:
        List[dict]: Observations SUCCESS avec métadonnées complètes
    
    Raises:
        ValueError: Si aucune observation SUCCESS trouvée
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
            f"Aucune observation SUCCESS pour params={params_config_id}"
        )
    
    observations = []
    for row in rows:
        try:
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
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Skip observation {row['observation_id']}: {e}")
            continue
    
    return observations


def observations_to_dataframe(observations: List[dict]) -> pd.DataFrame:
    """
    Convertit observations → DataFrame avec TOUTES projections.
    
    Chaque ligne = (gamma, d, modifier, seed, test, metric, projections).
    
    Args:
        observations: Liste observations SUCCESS
    
    Returns:
        DataFrame normalisé avec toutes projections
    """
    rows = []
    
    for obs in observations:
        gamma_id = obs['gamma_id']
        d_encoding_id = obs['d_encoding_id']
        modifier_id = obs['modifier_id']
        seed = obs['seed']
        test_name = obs['test_name']
        
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
                'metric_name': metric_name,
                
                # Projections numériques - statistiques
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                
                # Projections numériques - évolution
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                
                # Projections catégorielles
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
# ANALYSIS : VARIANCE PAR FACTOR (TOUS FACTORS ÉGAUX)
# =============================================================================

def analyze_variance_by_factor(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str]
) -> pd.DataFrame:
    """
    Pour chaque (test, metric, projection), calcule variance expliquée par factor.
    
    ⚠️  TOUS factors traités égalitairement : gamma, D, modifier, seed, test.
    
    Méthodologie :
    - Kruskal-Wallis : significativité (non-paramétrique)
    - Variance ratio : importance relative (heuristique R0)
    
    Variance ratio = Var(moyennes_groupes) / Var(totale)
    ⚠️  Heuristique R0, pas décomposition ANOVA standard.
    ⚠️  P-values NON CORRIGÉES pour multiplicité.
    
    Args:
        df: DataFrame observations
        factors: TOUS factors ['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', 'test_name']
        projections: Projections à analyser
    
    Returns:
        DataFrame résultats avec [test, metric, projection, factor, variance_ratio, p_value]
    """
    results = []
    
    for projection in projections:
        # Grouper par (test, metric)
        for (test_name, metric_name), group in df.groupby(['test_name', 'metric_name']):
            
            for factor in factors:
                # Grouper par factor
                factor_groups = [
                    g[projection].dropna().values 
                    for name, g in group.groupby(factor)
                    if len(g.dropna()) >= 2
                ]
                
                if len(factor_groups) < 2:
                    continue
                
                # Test Kruskal-Wallis
                try:
                    statistic, p_value = kruskal(*factor_groups)
                except (ValueError, Exception):
                    continue
                
                # Variance ratio (heuristique)
                group_means = [np.mean(g) for g in factor_groups]
                var_between = np.var(group_means)
                var_total = np.var(group[projection].dropna())
                
                if var_total < 1e-10:
                    variance_ratio = 0.0
                else:
                    variance_ratio = var_between / var_total
                
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
    
    return pd.DataFrame(results).sort_values('variance_ratio', ascending=False)


# =============================================================================
# ANALYSIS : DISCRIMINANCE MÉTRIQUES
# =============================================================================

def analyze_metric_discrimination(
    df: pd.DataFrame,
    projections: List[str]
) -> pd.DataFrame:
    """
    Détecte métriques avec variance très faible (non discriminantes).
    
    Coefficient variation (CV) = std / |mean|
    CV < 0.1 → métrique peu discriminante (seuil arbitraire R0)
    
    Args:
        df: DataFrame observations
        projections: Projections à analyser
    
    Returns:
        DataFrame avec [test, metric, projection, cv, non_discriminant]
    """
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
# ANALYSIS : CORRÉLATIONS MÉTRIQUES
# =============================================================================

def analyze_metric_correlations(
    df: pd.DataFrame,
    projection: str = 'value_final',
    threshold: float = 0.8
) -> pd.DataFrame:
    """
    Détecte corrélations fortes entre métriques (redondance).
    
    Spearman rank correlation (non-paramétrique).
    
    Args:
        df: DataFrame observations
        projection: Projection à analyser
        threshold: Seuil corrélation (> threshold → redondant)
    
    Returns:
        DataFrame avec [test, metric1, metric2, correlation, p_value]
    """
    results = []
    
    for test_name, test_group in df.groupby('test_name'):
        
        # Pivot : obs × metrics
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
# ANALYSIS : INTERACTIONS CONDITIONNELLES
# =============================================================================

def analyze_conditional_interactions(
    df: pd.DataFrame,
    projections: List[str]
) -> pd.DataFrame:
    """
    Détecte interactions conditionnelles entre factors.
    
    Exemples :
    - Variance modifier DANS chaque D
    - Variance seed DANS chaque modifier
    - Variance D DANS chaque gamma
    
    Args:
        df: DataFrame observations
        projections: Projections à analyser
    
    Returns:
        DataFrame avec interactions conditionnelles détectées
    """
    results = []
    
    # Définir paires (factor_in, factor_conditioned)
    interaction_pairs = [
        ('modifier_id', 'd_encoding_id'),      # Modifier DANS D
        ('seed', 'modifier_id'),               # Seed DANS modifier
        ('d_encoding_id', 'gamma_id'),         # D DANS gamma
        ('modifier_id', 'gamma_id'),           # Modifier DANS gamma
        ('seed', 'd_encoding_id'),             # Seed DANS D
    ]
    
    for projection in projections:
        for factor_in, factor_conditioned in interaction_pairs:
            
            # Pour chaque valeur du factor conditionné
            for conditioned_value, conditioned_group in df.groupby(factor_conditioned):
                
                # Grouper par (test, metric) dans ce contexte
                for (test_name, metric_name), group in conditioned_group.groupby(['test_name', 'metric_name']):
                    
                    # Variance du factor_in
                    factor_groups = [
                        g[projection].dropna().values
                        for name, g in group.groupby(factor_in)
                        if len(g.dropna()) >= 2
                    ]
                    
                    if len(factor_groups) < 2:
                        continue
                    
                    # Test significativité
                    try:
                        statistic, p_value = kruskal(*factor_groups)
                    except:
                        continue
                    
                    # Variance ratio
                    group_means = [np.mean(g) for g in factor_groups]
                    var_between = np.var(group_means)
                    var_total = np.var(group[projection].dropna())
                    
                    if var_total < 1e-10:
                        variance_ratio = 0.0
                    else:
                        variance_ratio = var_between / var_total
                    
                    # Seuil plus strict pour interactions
                    if variance_ratio > 0.3 and p_value < 0.05:
                        results.append({
                            'interaction_type': f'{factor_in}_in_{factor_conditioned}',
                            'test_name': test_name,
                            'metric_name': metric_name,
                            'projection': projection,
                            'conditioned_value': conditioned_value,  # Valeur D, gamma, etc.
                            'factor_varying': factor_in,
                            'variance_ratio': variance_ratio,
                            'p_value': p_value,
                            'n_groups': len(factor_groups)
                        })
    
    return pd.DataFrame(results)


# =============================================================================
# INTERPRETATION : PATTERNS GLOBAUX + PAR GAMMA
# =============================================================================

def interpret_patterns(
    df: pd.DataFrame,
    variance_analysis: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame,
    interactions: pd.DataFrame,
    verdict_config: dict
) -> Tuple[dict, dict]:
    """
    Synthèse patterns : GLOBAUX + PAR GAMMA.
    
    Architecture à 2 niveaux :
    1. Patterns globaux (tous gammas confondus)
    2. Patterns par gamma (drill-down)
    
    Args:
        df: DataFrame complet
        variance_analysis: Résultats variance (tous factors)
        discrimination: Résultats discriminance
        correlations: Résultats corrélations
        interactions: Résultats interactions
        verdict_config: Config verdict
    
    Returns:
        (patterns_global, patterns_by_gamma)
    """
    # =========================================================================
    # PATTERNS GLOBAUX
    # =========================================================================
    
    patterns_global = {
        'factor_dominant': [],
        'non_discriminant': [],
        'redundant': [],
        'conditional': []
    }
    
    # 1. Factor dominant (TOUS factors)
    dominant = variance_analysis[
        (variance_analysis['variance_ratio'] > 0.5) &
        (variance_analysis['significant'])
    ]
    
    if not dominant.empty:
        for factor in dominant['factor'].unique():
            subset = dominant[dominant['factor'] == factor]
            
            patterns_global['factor_dominant'].append({
                'factor': factor,
                'n_metrics': len(subset),
                'projections': subset['projection'].unique().tolist(),
                'max_variance_ratio': float(subset['variance_ratio'].max()),
                'examples': subset.head(3)[['test_name', 'metric_name', 'projection', 'variance_ratio']].to_dict('records')
            })
    
    # 2. Non discriminant
    non_disc = discrimination[discrimination['non_discriminant']]
    
    if not non_disc.empty:
        patterns_global['non_discriminant'].append({
            'n_metrics': len(non_disc),
            'projections': non_disc['projection'].unique().tolist(),
            'examples': non_disc[['test_name', 'metric_name', 'projection', 'cv']].to_dict('records')
        })
    
    # 3. Redondant
    if not correlations.empty:
        patterns_global['redundant'].append({
            'n_pairs': len(correlations),
            'examples': correlations.nlargest(5, 'correlation', keep='all').to_dict('records')
        })
    
    # 4. Conditional
    if not interactions.empty:
        # Grouper par type interaction
        interaction_types = {}
        for int_type in interactions['interaction_type'].unique():
            subset = interactions[interactions['interaction_type'] == int_type]
            interaction_types[int_type] = {
                'n_cases': len(subset),
                'examples': subset.head(3).to_dict('records')
            }
        
        patterns_global['conditional'].append(interaction_types)
    
    # =========================================================================
    # PATTERNS PAR GAMMA (DRILL-DOWN)
    # =========================================================================
    
    patterns_by_gamma = {}
    
    for gamma_id in df['gamma_id'].unique():
        
        gamma_df = df[df['gamma_id'] == gamma_id]
        
        # Variance analysis restreint à ce gamma (factors sauf gamma)
        gamma_variance = variance_analysis[
            variance_analysis['test_name'].isin(gamma_df['test_name'].unique()) &
            (variance_analysis['factor'] != 'gamma_id')  # Exclure variance gamma dans drill-down
        ]
        
        # Discrimination restreinte
        gamma_discrimination = discrimination[
            discrimination['test_name'].isin(gamma_df['test_name'].unique())
        ]
        
        # Interactions restreintes
        gamma_interactions = interactions[
            interactions['test_name'].isin(gamma_df['test_name'].unique())
        ]
        
        patterns_gamma = {
            'factor_dominant': [],
            'non_discriminant': [],
            'conditional': []
        }
        
        # Factor dominant dans ce gamma
        dominant_gamma = gamma_variance[
            (gamma_variance['variance_ratio'] > 0.5) &
            (gamma_variance['significant'])
        ]
        
        if not dominant_gamma.empty:
            for factor in dominant_gamma['factor'].unique():
                subset = dominant_gamma[dominant_gamma['factor'] == factor]
                
                patterns_gamma['factor_dominant'].append({
                    'factor': factor,
                    'n_metrics': len(subset),
                    'projections': subset['projection'].unique().tolist(),
                    'max_variance_ratio': float(subset['variance_ratio'].max())
                })
        
        # Non discriminant dans ce gamma
        non_disc_gamma = gamma_discrimination[gamma_discrimination['non_discriminant']]
        
        if not non_disc_gamma.empty:
            patterns_gamma['non_discriminant'].append({
                'n_metrics': len(non_disc_gamma),
                'projections': non_disc_gamma['projection'].unique().tolist()
            })
        
        # Interactions conditionnelles dans ce gamma
        if not gamma_interactions.empty:
            interaction_types_gamma = {}
            for int_type in gamma_interactions['interaction_type'].unique():
                subset = gamma_interactions[gamma_interactions['interaction_type'] == int_type]
                interaction_types_gamma[int_type] = {
                    'n_cases': len(subset)
                }
            
            patterns_gamma['conditional'].append(interaction_types_gamma)
        
        patterns_by_gamma[gamma_id] = patterns_gamma
    
    return patterns_global, patterns_by_gamma


# =============================================================================
# VERDICT : GLOBAL + PAR GAMMA
# =============================================================================

def decide_verdict(
    patterns_global: dict,
    patterns_by_gamma: dict,
    verdict_config: dict
) -> Tuple[str, str, dict]:
    """
    Décision verdict : GLOBAL + PAR GAMMA.
    
    Returns:
        (verdict_global, reason_global, verdicts_by_gamma)
    """
    # =========================================================================
    # VERDICT GLOBAL
    # =========================================================================
    
    critical_count = sum(len(v) for v in patterns_global.values() if v)
    
    if critical_count == 0:
        verdict_global = "SURVIVES[R0]"
        reason_global = "Aucun pattern pathologique systématique détecté."
    else:
        reasons = []
        
        if patterns_global['factor_dominant']:
            for p in patterns_global['factor_dominant']:
                reasons.append(f"{p['factor']} explique >{p['max_variance_ratio']:.0%} variance ({p['n_metrics']} métriques)")
        
        if patterns_global['non_discriminant']:
            p = patterns_global['non_discriminant'][0]
            reasons.append(f"{p['n_metrics']} métriques non discriminantes (CV<0.1)")
        
        if patterns_global['redundant']:
            p = patterns_global['redundant'][0]
            reasons.append(f"{p['n_pairs']} paires métriques redondantes (|r|>0.8)")
        
        if patterns_global['conditional']:
            p = patterns_global['conditional'][0]
            n_interactions = sum(v['n_cases'] for v in p.values())
            reasons.append(f"{n_interactions} interactions conditionnelles détectées")
        
        reason_global = " | ".join(reasons)
        verdict_global = "WIP[R0-open]"
    
    # =========================================================================
    # VERDICTS PAR GAMMA
    # =========================================================================
    
    verdicts_by_gamma = {}
    
    for gamma_id, patterns_gamma in patterns_by_gamma.items():
        
        critical_count_gamma = sum(len(v) for v in patterns_gamma.values() if v)
        
        if critical_count_gamma == 0:
            verdict_gamma = "SURVIVES[R0]"
            reason_gamma = "Aucun pattern spécifique détecté pour ce gamma."
        else:
            reasons_gamma = []
            
            if patterns_gamma['factor_dominant']:
                for p in patterns_gamma['factor_dominant']:
                    reasons_gamma.append(f"{p['factor']} domine ({p['n_metrics']} métriques)")
            
            if patterns_gamma['non_discriminant']:
                p = patterns_gamma['non_discriminant'][0]
                reasons_gamma.append(f"{p['n_metrics']} métriques non discriminantes")
            
            if patterns_gamma['conditional']:
                p = patterns_gamma['conditional'][0]
                n_int = sum(v['n_cases'] for v in p.values())
                reasons_gamma.append(f"{n_int} interactions")
            
            reason_gamma = " | ".join(reasons_gamma)
            verdict_gamma = "WIP[R0-open]"
        
        verdicts_by_gamma[gamma_id] = {
            'verdict': verdict_gamma,
            'reason': reason_gamma,
            'patterns': patterns_gamma
        }
    
    return verdict_global, reason_global, verdicts_by_gamma


# =============================================================================
# RAPPORTS
# =============================================================================

def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    df: pd.DataFrame,
    variance_analysis: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame,
    interactions: pd.DataFrame,
    patterns_global: dict,
    patterns_by_gamma: dict,
    verdict_global: str,
    reason_global: str,
    verdicts_by_gamma: dict,
    projections_analyzed: List[str],
    output_dir: str = "reports/verdicts"
) -> None:
    """
    Génère rapports : GLOBAL + PAR GAMMA.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_analysis_full"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'factors_analyzed': ['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', 'test_name'],
        'projections_analyzed': projections_analyzed,
        'projections_ignored': ['transition', 'trend'],
        'n_observations': len(df),
        'n_gammas': len(df['gamma_id'].unique()),
        'n_tests': len(df['test_name'].unique()),
        'n_metrics': len(df.groupby(['test_name', 'metric_name'])),
        'limitations': {
            'variance_ratio': 'Heuristique R0, pas décomposition ANOVA standard',
            'p_values': 'Non corrigées pour multiplicité (faux positifs attendus)',
            'seuils': 'Arbitraires fixes (0.3, 0.1, 0.8)',
            'interactions': 'Conditionnelles simples uniquement'
        }
    }
    
    with open(report_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Rapport humain
    with open(report_dir / 'summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VERDICT ANALYSIS - EXPLORATOIRE R0\n")
        f.write(f"{timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("⚠️  LIMITATIONS MÉTHODOLOGIQUES :\n")
        f.write("  - Tests statistiques NON CORRIGÉS pour multiplicité\n")
        f.write("  - Variance ratio = heuristique R0 (pas ANOVA standard)\n")
        f.write("  - Seuils fixes arbitraires (0.3, 0.1, 0.8)\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Params:  {params_config_id}\n")
        f.write(f"  Verdict: {verdict_config_id}\n")
        f.write(f"  Observations: {metadata['n_observations']}\n")
        f.write(f"  Gammas: {metadata['n_gammas']}\n")
        f.write(f"  Tests: {metadata['n_tests']}\n")
        f.write(f"  Métriques: {metadata['n_metrics']}\n\n")
        
        f.write("FACTORS ANALYSÉS (tous égalitairement):\n")
        f.write(f"  {', '.join(metadata['factors_analyzed'])}\n\n")
        
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
        
        f.write("="*80 + "\n")
        f.write("VERDICTS PAR GAMMA (drill-down)\n")
        f.write("="*80 + "\n\n")
        
        for gamma_id, verdict_info in verdicts_by_gamma.items():
            f.write(f"--- {gamma_id} ---\n")
            f.write(f"Verdict: {verdict_info['verdict']}\n")
            f.write(f"Raison:  {verdict_info['reason']}\n")
            
            patterns_gamma = verdict_info['patterns']
            critical_gamma = sum(len(v) for v in patterns_gamma.values() if v)
            if critical_gamma > 0:
                f.write(f"Patterns: ")
                for pt, pl in patterns_gamma.items():
                    if pl:
                        f.write(f"{pt}({len(pl)}) ")
                f.write("\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("NOTE : Résultats exploratoires nécessitant validation humaine.\n")
        f.write("="*80 + "\n")
    
    # Rapport JSON complet
    report_json = {
        'metadata': metadata,
        'verdict_global': {
            'verdict': verdict_global,
            'reason': reason_global,
            'patterns': patterns_global
        },
        'verdicts_by_gamma': verdicts_by_gamma
    }
    
    with open(report_dir / 'analysis_complete.json', 'w') as f:
        json.dump(report_json, f, indent=2)
    
    # CSV analyses
    variance_analysis.to_csv(report_dir / 'variance_by_factor.csv', index=False)
    discrimination.to_csv(report_dir / 'discrimination.csv', index=False)
    if not correlations.empty:
        correlations.to_csv(report_dir / 'correlations.csv', index=False)
    if not interactions.empty:
        interactions.to_csv(report_dir / 'conditional_interactions.csv', index=False)
    
    print(f"\n✓ Rapports générés : {report_dir}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None:
    """
    Pipeline complet verdict exploratoire.
    
    Architecture à 2 niveaux :
    1. Analyse globale (tous factors égalitairement)
    2. Drill-down par gamma (présentation)
    """
    print(f"\n{'='*70}")
    print(f"VERDICT ANALYSIS - EXPLORATOIRE R0")
    print(f"{'='*70}\n")
    
    # Projections analysées
    projections_analyzed = [
        'value_final',
        'value_mean', 
        'slope',
        'volatility',
        'relative_change'
    ]
    
    print(f"Projections analysées : {', '.join(projections_analyzed)}")
    print(f"Projections ignorées : transition, trend (catégorielles)\n")
    
    # 1. Config
    print("1. Chargement config verdict...")
    loader = get_loader()
    verdict_config = loader.load('verdict', verdict_config_id)
    print(f"   ✓ Config {verdict_config_id} chargée")
    
    # 2. Observations
    print("2. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations SUCCESS")
    
    # 3. DataFrame
    print("3. Structuration DataFrame...")
    df = observations_to_dataframe(observations)
    print(f"   ✓ {len(df)} lignes")
    print(f"   ✓ {len(df['gamma_id'].unique())} gammas distincts")
    
    # 4. Analyses
    print("4. Analyses statistiques...")
    
    # TOUS factors égalitairement
    factors = ['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', 'test_name']
    print(f"   Factors analysés : {', '.join(factors)}")
    
    variance_analysis = analyze_variance_by_factor(df, factors, projections_analyzed)
    print(f"   ✓ Variance analysis: {len(variance_analysis)} résultats")
    
    discrimination = analyze_metric_discrimination(df, projections_analyzed)
    print(f"   ✓ Discrimination: {len(discrimination)} métriques")
    
    correlations = analyze_metric_correlations(df, projection='value_final')
    print(f"   ✓ Correlations: {len(correlations)} paires")
    
    interactions = analyze_conditional_interactions(df, projections_analyzed)
    print(f"   ✓ Conditional interactions: {len(interactions)} détectées")
    
    # 5. Interpretation (2 niveaux)
    print("5. Interprétation patterns...")
    patterns_global, patterns_by_gamma = interpret_patterns(
        df,
        variance_analysis,
        discrimination,
        correlations,
        interactions,
        verdict_config
    )
    
    total_patterns_global = sum(len(v) for v in patterns_global.values() if v)
    print(f"   ✓ Patterns globaux : {total_patterns_global}")
    
    for gamma_id, patterns_gamma in patterns_by_gamma.items():
        total_gamma = sum(len(v) for v in patterns_gamma.values() if v)
        if total_gamma > 0:
            print(f"     - {gamma_id}: {total_gamma} patterns")
    
    # 6. Verdict (2 niveaux)
    print("6. Décision verdict...")
    verdict_global, reason_global, verdicts_by_gamma = decide_verdict(
        patterns_global,
        patterns_by_gamma,
        verdict_config
    )
    print(f"   → Global: {verdict_global}")
    for gamma_id, v in verdicts_by_gamma.items():
        print(f"     - {gamma_id}: {v['verdict']}")
    
    # 7. Rapports
    print("7. Génération rapports...")
    generate_verdict_report(
        params_config_id,
        verdict_config_id,
        df,
        variance_analysis,
        discrimination,
        correlations,
        interactions,
        patterns_global,
        patterns_by_gamma,
        verdict_global,
        reason_global,
        verdicts_by_gamma,
        projections_analyzed
    )
    
    print(f"\n{'='*70}")
    print(f"VERDICT GLOBAL: {verdict_global}")
    print(f"{'='*70}\n")