# tests/utilities/verdict_engine.py
"""
Verdict Engine Charter 5.5 - Analyse exploratoire générique REFACTORISÉ.

ARCHITECTURE REFACTORISÉE (Phase 2.1) :
- Délégation I/O → data_loading.py
- Délégation filtrage/diagnostics → statistical_utils.py
- Délégation stratification → regime_utils.py
- Cœur métier : analyses statistiques multi-facteurs (variance, interactions)

CORRECTIONS APPLIQUÉES (post-review) :
1. Paires ORIENTÉES : permutations() pas combinations()
2. Interaction = changement d'effet (VR conditionnel >> VR marginal)
3. Filtrage testabilité explicite (min_samples, min_groups)
4. params_config_id exclu interactions (trop corrélé)
5. Drill-down gamma recalculé sur sous-ensemble

RESPONSABILITÉS CONSERVÉES :
- Analyses variance marginale (η²)
- Analyses interactions orientées
- Discrimination métriques
- Corrélations
- Interprétation patterns
- Décision verdict
- Pipeline régime complet
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import warnings
import json

# ============================================================================
# IMPORTS MODULES REFACTORISÉS
# ============================================================================

# I/O et structuration données
from ..utils.data_loading import (
    load_all_observations,
    observations_to_dataframe
)

# Filtrage et diagnostics numériques
from ..utils.statistical_utils import (
    compute_eta_squared,
    kruskal_wallis_test,
    filter_numeric_artifacts,
    generate_degeneracy_report,
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report
)

# Stratification régimes
from ..utils.regime_utils import stratify_by_regime

# Configuration
from ..utils.config_loader import get_loader

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
    
    ⚠️ PHASE 2.1 : Correction critique calcul via compute_eta_squared()
    
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
                    statistic, p_value = kruskal_wallis_test(factor_groups)
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

    # Protection cas limite : données insuffisantes
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print(f"   ⚠️ Aucun groupe testable (tous < {MIN_TOTAL_SAMPLES} observations)")
        return pd.DataFrame()  # Retour vide propre

    return df_results.sort_values('variance_ratio', ascending=False)


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
                            statistic, p_value = kruskal_wallis_test(varying_groups)
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
    if not marginal_variance.empty:  # ← Protection ajoutée
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
    if not discrimination.empty:  # ← Protection ajoutée
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
    # PATTERNS PAR GAMMA (drill-down RECALCULÉ)
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
        if not gamma_marginal.empty:  # ← Protection ajoutée
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
    
    if patterns_by_gamma:
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
# PIPELINE RÉGIME
# =============================================================================

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


# =============================================================================
# RAPPORTS STRATIFIÉS
# =============================================================================

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
    
    # CSVs par régime
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
# PIPELINE PRINCIPAL
# =============================================================================

def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None:
    """Pipeline complet verdict exploratoire REFACTORISÉ."""
    
    print(f"\n{'='*70}")
    print(f"VERDICT ANALYSIS - INTERACTIONS ORIENTÉES (REFACTORISÉ)")
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
    
    # 2. Observations (DÉLÉGUÉ → data_loading)
    print("2. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations")
    
    # Filtrage artefacts numériques (DÉLÉGUÉ → statistical_utils)
    observations, rejection_stats = filter_numeric_artifacts(observations)
    
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        print(f"      Détail par test :")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"        {test}: {count} invalides")
    print()
    
    # Diagnostics (DÉLÉGUÉ → statistical_utils)
    print("3. Diagnostics numériques...")
    degeneracy_report = generate_degeneracy_report(observations)
    print_degeneracy_report(degeneracy_report)
    
    scale_report = diagnose_scale_outliers(observations)
    print_scale_outliers_report(scale_report)
    
    # Stratification (DÉLÉGUÉ → regime_utils)
    print("4. Stratification régimes...")
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE    : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF  : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    # Analyses parallèles (CŒUR MÉTIER)
    print("\n5. Analyses statistiques stratifiées...")
    
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
    
    print(f"\n{'='*70}")
    print(f"VERDICT REFACTORISÉ : Analyses complètes sur 3 strates")
    print(f"{'='*70}\n")