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
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import warnings

from .config_loader import get_loader

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
# ANALYSIS 1 : VARIANCE MARGINALE
# =============================================================================

def analyze_marginal_variance(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str]
) -> pd.DataFrame:
    """
    Variance marginale : chaque factor pris isolément.
    
    ⚠️  Filtrage testabilité : min_samples, min_groups.
    
    Returns:
        DataFrame avec [test, metric, projection, factor, variance_ratio, p_value]
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
                
                # Variance ratio
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
    Interactions ORIENTÉES : A|B distinct de B|A.
    
    Interaction détectée si :
    - VR(A|B=b) significatif
    - VR(A|B=b) >> VR(A) marginal (ratio > min_interaction_strength)
    
    ⚠️  Correction critique : permutations() pas combinations().
    ⚠️  Interaction = changement d'effet, pas juste effet conditionnel.
    
    Args:
        df: DataFrame observations
        factors: Liste factors
        projections: Projections
        marginal_variance: Variance marginale (pour comparaison)
        min_interaction_strength: Ratio VR_conditionnel / VR_marginal
    
    Returns:
        DataFrame interactions VRAIES détectées
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
                        
                        # Variance ratio conditionnel
                        group_means = [np.mean(g) for g in varying_groups]
                        var_between = np.var(group_means)
                        var_total = np.var(context_group[projection].dropna())
                        
                        if var_total < 1e-10:
                            vr_conditional = 0.0
                        else:
                            vr_conditional = var_between / var_total
                        
                        # Comparer à variance marginale
                        marginal_key = (test_name, metric_name, projection, factor_varying)
                        vr_marginal = marginal_index.get(marginal_key, 0.0)
                        
                        if vr_marginal < 0.05:
                            continue
                            #interaction_strength = np.inf if vr_conditional > 0.3 else 0.0
                        else:
                            interaction_strength = vr_conditional / vr_marginal
                        
                        # Interaction = effet change selon contexte
                        if (vr_conditional > 0.3 and 
                            p_value < 0.05 and 
                            interaction_strength > min_interaction_strength):
                            
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
    verdicts_by_gamma = {}
    
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
    
    # 3. DataFrame
    print("3. Structuration DataFrame...")
    df = observations_to_dataframe(observations)
    print(f"   ✓ {len(df)} lignes")
    
    # 4. Analyses
    print("4. Analyses statistiques...")
    
    # 4a. Variance marginale
    marginal_variance = analyze_marginal_variance(df, FACTORS, PROJECTIONS)
    print(f"   ✓ Variance marginale: {len(marginal_variance)} résultats")
    
    # 4b. Interactions orientées (CORRIGÉ)
    oriented_interactions = analyze_oriented_interactions(
        df, FACTORS, PROJECTIONS, marginal_variance
    )
    print(f"   ✓ Interactions orientées: {len(oriented_interactions)} détectées")
    
    # 4c. Discrimination
    discrimination = analyze_metric_discrimination(df, PROJECTIONS)
    print(f"   ✓ Discrimination: {len(discrimination)} métriques")
    
    # 4d. Corrélations
    correlations = analyze_metric_correlations(df)
    print(f"   ✓ Corrélations: {len(correlations)} paires")
    
    # 5. Interpretation
    print("5. Interprétation patterns...")
    patterns_global, patterns_by_gamma = interpret_patterns(
        df,
        marginal_variance,
        oriented_interactions,
        discrimination,
        correlations
    )
    
    # 6. Verdict
    print("6. Décision verdict...")
    verdict_global, reason_global, verdicts_by_gamma = decide_verdict(
        patterns_global,
        patterns_by_gamma
    )
    print(f"   → Global: {verdict_global}")
    
    # 7. Rapports
    print("7. Génération rapports...")
    generate_verdict_report(
        params_config_id,
        verdict_config_id,
        df,
        marginal_variance,
        oriented_interactions,
        discrimination,
        correlations,
        patterns_global,
        patterns_by_gamma,
        verdict_global,
        reason_global,
        verdicts_by_gamma
    )
    
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict_global}")
    print(f"{'='*70}\n")