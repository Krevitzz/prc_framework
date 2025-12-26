#!/usr/bin/env python3
"""
prc_automation/query_results.py

Requêtes sur db_results (analyses et verdicts).

Usage:
    python query_results.py --summary
    python query_results.py --gamma GAM-001 --config weights_default
    python query_results.py --verdicts --config weights_default --thresholds thresholds_default
    python query_results.py --compare-configs --gamma GAM-001
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


# =============================================================================
# QUERIES DE BASE
# =============================================================================

def get_summary(conn) -> pd.DataFrame:
    """Résumé global des analyses."""
    query = """
        SELECT 
            (SELECT COUNT(DISTINCT exec_id) FROM TestObservations) as n_runs_tested,
            (SELECT COUNT(DISTINCT config_id) FROM TestScores) as n_configs,
            (SELECT COUNT(*) FROM GammaVerdicts) as n_verdicts,
            (SELECT COUNT(DISTINCT gamma_id) FROM GammaVerdicts) as n_gammas_evaluated
    """
    return pd.read_sql_query(query, conn)


def get_gamma_scores(conn, gamma_id: str, config_id: str) -> pd.DataFrame:
    """Scores pour un Γ et une config."""
    query = """
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            ts.weight,
            AVG(ts.weighted_score) as avg_weighted_score,
            COUNT(*) as n_runs
        FROM TestScores ts
        JOIN TestObservations to2 ON ts.exec_id = to2.exec_id AND ts.test_name = to2.test_name
        WHERE ts.config_id = ?
        GROUP BY ts.test_name, ts.weight
        ORDER BY ts.test_name
    """
    # Note: Besoin de filter gamma_id via db_raw, simplifié ici
    return pd.read_sql_query(query, conn, params=(config_id,))


def get_verdicts(conn, config_id: str, threshold_id: str) -> pd.DataFrame:
    """Tous les verdicts pour une config/threshold."""
    query = """
        SELECT 
            gamma_id,
            verdict,
            score_global,
            majority_pct,
            robustness_pct,
            verdict_reason,
            computed_at
        FROM GammaVerdicts
        WHERE config_id = ? AND threshold_id = ?
        ORDER BY score_global DESC
    """
    return pd.read_sql_query(query, conn, params=(config_id, threshold_id))


def get_gamma_verdict(conn, gamma_id: str, config_id: str, threshold_id: str) -> Optional[Dict]:
    """Verdict d'un Γ spécifique."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM GammaVerdicts
        WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
    """, (gamma_id, config_id, threshold_id))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def compare_configs(conn, gamma_id: str) -> pd.DataFrame:
    """Compare verdicts d'un Γ pour toutes configs."""
    query = """
        SELECT 
            config_id,
            threshold_id,
            verdict,
            score_global,
            majority_pct,
            robustness_pct
        FROM GammaVerdicts
        WHERE gamma_id = ?
        ORDER BY config_id, threshold_id
    """
    return pd.read_sql_query(query, conn, params=(gamma_id,))


def get_test_observations(conn, gamma_id: str, test_name: str) -> pd.DataFrame:
    """Observations d'un test spécifique pour un Γ."""
    query = """
        SELECT 
            exec_id,
            observation_data,
            initial_value,
            final_value,
            transition
        FROM TestObservations
        WHERE test_name = ?
        ORDER BY exec_id
    """
    # Note: Filtrage gamma_id nécessite join avec db_raw, simplifié
    return pd.read_sql_query(query, conn, params=(test_name,))


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Affiche résumé global."""
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL db_results")
    print("="*70)
    
    row = df.iloc[0]
    print(f"\nRuns testés: {row['n_runs_tested']}")
    print(f"Configs scoring: {row['n_configs']}")
    print(f"Verdicts calculés: {row['n_verdicts']}")
    print(f"Γ évalués: {row['n_gammas_evaluated']}")
    
    print("\n" + "="*70 + "\n")


def print_gamma_scores(df: pd.DataFrame, gamma_id: str, config_id: str):
    """Affiche scores d'un Γ."""
    print(f"\n{'='*70}")
    print(f"SCORES: {gamma_id}, config={config_id}")
    print(f"{'='*70}\n")
    
    if df.empty:
        print("Aucun score trouvé")
        return
    
    # Calculer score global
    total_weighted = df['avg_weighted_score'].sum()
    total_weight = df['weight'].sum()
    score_global = (total_weighted / total_weight * 20) if total_weight > 0 else 0
    
    print(f"Score global: {score_global:.2f}/20\n")
    
    print("Détail par test:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}\n")


def print_verdicts(df: pd.DataFrame, config_id: str, threshold_id: str):
    """Affiche tous les verdicts."""
    print(f"\n{'='*70}")
    print(f"VERDICTS: config={config_id}, thresholds={threshold_id}")
    print(f"{'='*70}\n")
    
    if df.empty:
        print("Aucun verdict trouvé")
        return
    
    # Grouper par verdict
    print("Résumé par verdict:")
    for verdict_type in ['SURVIVES[R0]', 'WIP[R0-closed]', 'FLAGGED_FOR_REVIEW']:
        subset = df[df['verdict'] == verdict_type]
        if not subset.empty:
            print(f"\n{verdict_type}: {len(subset)} Γ")
            for _, row in subset.iterrows():
                print(f"  - {row['gamma_id']}: score={row['score_global']:.1f}, "
                      f"maj={row['majority_pct']:.1f}%, rob={row['robustness_pct']:.1f}%")
    
    print(f"\n{'='*70}\n")


def print_gamma_verdict(verdict: Dict):
    """Affiche verdict d'un Γ spécifique."""
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict['gamma_id']}")
    print(f"{'='*70}\n")
    
    print(f"Config: {verdict['config_id']}")
    print(f"Thresholds: {verdict['threshold_id']}")
    print(f"\nVerdict: {verdict['verdict']}")
    print(f"Raison: {verdict['verdict_reason']}")
    
    print(f"\nCritères:")
    print(f"  Score global: {verdict['score_global']:.2f}/20")
    print(f"  Majorité: {verdict['majority_pct']:.1f}%")
    print(f"  Robustesse: {verdict['robustness_pct']:.1f}%")
    
    print(f"\nStatistiques:")
    print(f"  Total configs: {verdict['n_total_configs']}")
    print(f"  Configs PASS: {verdict['n_pass_configs']}")
    print(f"  D viables: {verdict['n_viable_d_bases']}/{verdict['n_total_d_bases']}")
    
    print(f"\nCalculé: {verdict['computed_at']}")
    
    print(f"{'='*70}\n")


def print_config_comparison(df: pd.DataFrame, gamma_id: str):
    """Affiche comparaison configs pour un Γ."""
    print(f"\n{'='*70}")
    print(f"COMPARAISON CONFIGS: {gamma_id}")
    print(f"{'='*70}\n")
    
    if df.empty:
        print("Aucun verdict trouvé")
        return
    
    print("Verdicts selon config/threshold:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}\n")


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_verdict_distribution(conn, config_id: str, threshold_id: str):
    """Analyse distribution des verdicts."""
    df = get_verdicts(conn, config_id, threshold_id)
    
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION VERDICTS")
    print(f"{'='*70}\n")
    
    if df.empty:
        print("Aucun verdict trouvé")
        return
    
    # Distribution
    for verdict_type in ['SURVIVES[R0]', 'WIP[R0-closed]', 'FLAGGED_FOR_REVIEW']:
        count = (df['verdict'] == verdict_type).sum()
        pct = count / len(df) * 100
        print(f"{verdict_type}: {count} ({pct:.1f}%)")
    
    print(f"\nTotal: {len(df)} Γ évalués")
    
    # Statistiques scores
    print(f"\nStatistiques scores globaux:")
    print(f"  Moyenne: {df['score_global'].mean():.2f}")
    print(f"  Médiane: {df['score_global'].median():.2f}")
    print(f"  Min: {df['score_global'].min():.2f}")
    print(f"  Max: {df['score_global'].max():.2f}")
    
    print(f"\n{'='*70}\n")


def analyze_criteria_correlation(conn, config_id: str, threshold_id: str):
    """Analyse corrélation entre les 3 critères."""
    df = get_verdicts(conn, config_id, threshold_id)
    
    if df.empty or len(df) < 3:
        print("Données insuffisantes pour corrélation")
        return
    
    print(f"\n{'='*70}")
    print(f"CORRÉLATION CRITÈRES")
    print(f"{'='*70}\n")
    
    # Corrélation
    corr_matrix = df[['score_global', 'majority_pct', 'robustness_pct']].corr()
    
    print("Matrice de corrélation:")
    print(corr_matrix)
    
    print(f"\n{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Query db_results")
    
    # Modes
    parser.add_argument('--summary', action='store_true',
                       help='Résumé global')
    parser.add_argument('--gamma', type=str,
                       help='Scores pour un Γ spécifique')
    parser.add_argument('--verdicts', action='store_true',
                       help='Liste tous les verdicts')
    parser.add_argument('--compare-configs', action='store_true',
                       help='Compare configs pour un Γ')
    parser.add_argument('--distribution', action='store_true',
                       help='Distribution verdicts')
    
    # Arguments
    parser.add_argument('--config', type=str, default='weights_default',
                       help='Config scoring')
    parser.add_argument('--thresholds', type=str, default='thresholds_default',
                       help='Config seuils')
    
    args = parser.parse_args()
    
    if not DB_RESULTS_PATH.exists():
        print(f"❌ Base non trouvée: {DB_RESULTS_PATH}")
        print("Lancer d'abord: python init_databases.py")
        return
    
    conn = sqlite3.connect(DB_RESULTS_PATH)
    
    try:
        if args.summary:
            df = get_summary(conn)
            print_summary(df)
        
        elif args.gamma and args.compare_configs:
            df = compare_configs(conn, args.gamma)
            print_config_comparison(df, args.gamma)
        
        elif args.gamma:
            df = get_gamma_scores(conn, args.gamma, args.config)
            print_gamma_scores(df, args.gamma, args.config)
            
            # Afficher verdict si existe
            verdict = get_gamma_verdict(conn, args.gamma, args.config, args.thresholds)
            if verdict:
                print_gamma_verdict(verdict)
        
        elif args.verdicts:
            df = get_verdicts(conn, args.config, args.thresholds)
            print_verdicts(df, args.config, args.thresholds)
        
        elif args.distribution:
            analyze_verdict_distribution(conn, args.config, args.thresholds)
            analyze_criteria_correlation(conn, args.config, args.thresholds)
        
        else:
            print("❌ Spécifier une option (--summary, --gamma, --verdicts, etc.)")
            print("\nExemples:")
            print("  python query_results.py --summary")
            print("  python query_results.py --gamma GAM-001")
            print("  python query_results.py --verdicts")
            print("  python query_results.py --distribution")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()