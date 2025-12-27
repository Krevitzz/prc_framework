#!/usr/bin/env python3
"""
prc_automation/visualize_results.py

Visualiseur interactif pour inspecter et valider les résultats.

Usage:
    # Vue d'ensemble complète
    python visualize_results.py --overview
    
    # Inspecter un Γ spécifique
    python visualize_results.py --inspect GAM-001 --config weights_default
    
    # Voir évolution temporelle d'un run
    python visualize_results.py --timeline GAM-001_beta2.0_SYM-001_M0_s1
    
    # Comparer configs
    python visualize_results.py --compare GAM-001
    
    # Rapport complet HTML
    python visualize_results.py --report GAM-001 --output report.html
"""

import argparse
import sqlite3
import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour génération fichiers

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


# =============================================================================
# VUE D'ENSEMBLE
# =============================================================================

def show_overview():
    """Vue d'ensemble complète du pipeline."""
    print("\n" + "="*80)
    print("OVERVIEW - État du pipeline R0")
    print("="*80 + "\n")
    
    # db_raw
    if DB_RAW_PATH.exists():
        conn_raw = sqlite3.connect(DB_RAW_PATH)
        
        cursor = conn_raw.cursor()
        cursor.execute("SELECT COUNT(*) FROM Executions")
        n_runs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Executions WHERE status = 'COMPLETED'")
        n_completed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT gamma_id) FROM Executions")
        n_gammas = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Snapshots")
        n_snapshots = cursor.fetchone()[0]
        
        print(f"db_raw (données factuelles):")
        print(f"  ✓ Runs totaux: {n_runs}")
        print(f"  ✓ Complétés: {n_completed} ({100*n_completed/n_runs:.1f}%)" if n_runs > 0 else "")
        print(f"  ✓ Γ testés: {n_gammas}")
        print(f"  ✓ Snapshots: {n_snapshots}")
        
        conn_raw.close()
    else:
        print("db_raw: ❌ Non initialisée")
    
    print()
    
    # db_results
    if DB_RESULTS_PATH.exists():
        conn_results = sqlite3.connect(DB_RESULTS_PATH)
        
        cursor = conn_results.cursor()
        cursor.execute("SELECT COUNT(DISTINCT exec_id) FROM TestObservations")
        n_tested = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT config_id) FROM TestScores")
        n_configs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM GammaVerdicts")
        n_verdicts = cursor.fetchone()[0]
        
        print(f"db_results (analyses):")
        print(f"  ✓ Runs testés: {n_tested}")
        print(f"  ✓ Configs scoring: {n_configs}")
        print(f"  ✓ Verdicts: {n_verdicts}")
        
        # Distribution verdicts
        if n_verdicts > 0:
            cursor.execute("""
                SELECT verdict, COUNT(*) 
                FROM GammaVerdicts 
                GROUP BY verdict
            """)
            print(f"\n  Distribution verdicts:")
            for verdict, count in cursor.fetchall():
                print(f"    - {verdict}: {count}")
        
        conn_results.close()
    else:
        print("db_results: ❌ Non initialisée")
    
    print("\n" + "="*80 + "\n")


# =============================================================================
# INSPECTION D'UN Γ SPÉCIFIQUE
# =============================================================================

def inspect_gamma(gamma_id: str, config_id: str = "weights_default"):
    """Inspection détaillée d'un Γ."""
    print(f"\n{'='*80}")
    print(f"INSPECTION: {gamma_id} (config={config_id})")
    print(f"{'='*80}\n")
    
    # Connexion avec ATTACH
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # 1. Résumé exécutions
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors,
            AVG(execution_time_seconds) as avg_time
        FROM db_raw.Executions
        WHERE gamma_id = ?
    """, (gamma_id,))
    
    row = cursor.fetchone()
    print(f"Exécutions (db_raw):")
    print(f"  Total: {row[0]}")
    print(f"  Complétés: {row[1]}")
    print(f"  Erreurs: {row[2]}")
    print(f"  Temps moyen: {row[3]:.1f}s" if row[3] else "  Temps moyen: N/A")
    
    # 2. Tests appliqués
    cursor.execute("""
        SELECT COUNT(DISTINCT to2.test_name)
        FROM TestObservations to2
        JOIN db_raw.Executions e ON to2.exec_id = e.id
        WHERE e.gamma_id = ? AND to2.applicable = 1
    """, (gamma_id,))
    n_tests = cursor.fetchone()[0]
    print(f"\nTests (db_results):")
    print(f"  Tests applicables: {n_tests}")
    
    # 3. Scores par test
    cursor.execute("""
        SELECT 
            ts.test_name,
            COUNT(*) as n_runs,
            AVG(ts.score) as avg_score,
            MIN(ts.score) as min_score,
            MAX(ts.score) as max_score,
            ts.weight
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
        GROUP BY ts.test_name, ts.weight
        ORDER BY ts.test_name
    """, (gamma_id, config_id))
    
    print(f"\nScores par test (config={config_id}):")
    print(f"  {'Test':<15} {'N':<5} {'Avg':<6} {'Min':<6} {'Max':<6} {'Weight':<6}")
    print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    
    test_scores = cursor.fetchall()
    for test_name, n_runs, avg_score, min_score, max_score, weight in test_scores:
        print(f"  {test_name:<15} {n_runs:<5} {avg_score:>5.3f}  {min_score:>5.3f}  {max_score:>5.3f}  {weight:>5.1f}")
    
    # 4. Calcul score global
    if test_scores:
        total_weighted = sum(avg * weight for _, _, avg, _, _, weight in test_scores)
        total_weight = sum(weight for _, _, _, _, _, weight in test_scores)
        score_global = (total_weighted / total_weight * 20) if total_weight > 0 else 0
        print(f"\n  Score global: {score_global:.2f}/20")
    
    # 5. Verdict
    cursor.execute("""
        SELECT verdict, verdict_reason, score_global, majority_pct, robustness_pct
        FROM GammaVerdicts
        WHERE gamma_id = ? AND config_id = ?
        LIMIT 1
    """, (gamma_id, config_id))
    
    verdict_row = cursor.fetchone()
    if verdict_row:
        verdict, reason, score, maj, rob = verdict_row
        print(f"\nVerdict:")
        print(f"  → {verdict}")
        print(f"  Raison: {reason}")
        print(f"  Critères: score={score:.2f}/20, majorité={maj:.1f}%, robustesse={rob:.1f}%")
    else:
        print(f"\nVerdict: ❌ Pas de verdict calculé")
    
    # 6. Distribution observations par D
    cursor.execute("""
        SELECT 
            e.d_base_id,
            COUNT(DISTINCT e.id) as n_runs,
            AVG(ts.score) as avg_score
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
        GROUP BY e.d_base_id
        ORDER BY avg_score DESC
    """, (gamma_id, config_id))
    
    print(f"\nPerformance par D:")
    print(f"  {'D':<10} {'N runs':<8} {'Score moyen':<12}")
    print(f"  {'-'*10} {'-'*8} {'-'*12}")
    
    for d_base_id, n_runs, avg_score in cursor.fetchall():
        print(f"  {d_base_id:<10} {n_runs:<8} {avg_score:>11.3f}")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    print(f"\n{'='*80}\n")


# =============================================================================
# TIMELINE D'UN RUN
# =============================================================================

def show_timeline(run_id: str):
    """Affiche évolution temporelle d'un run."""
    print(f"\n{'='*80}")
    print(f"TIMELINE: {run_id}")
    print(f"{'='*80}\n")
    
    conn_raw = sqlite3.connect(DB_RAW_PATH)
    cursor = conn_raw.cursor()
    
    # Récupérer exec_id
    cursor.execute("SELECT id FROM Executions WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if not row:
        print(f"❌ Run non trouvé: {run_id}")
        conn_raw.close()
        return
    
    exec_id = row[0]
    
    # Charger métriques
    cursor.execute("""
        SELECT 
            iteration,
            norm_frobenius,
            norm_spectral,
            std_value,
            min_value,
            max_value,
            distance_to_previous,
            asymmetry_norm
        FROM Metrics
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    metrics = cursor.fetchall()
    
    if not metrics:
        print(f"⚠ Aucune métrique trouvée")
        conn_raw.close()
        return
    
    # Afficher statistiques
    print(f"Métriques temporelles ({len(metrics)} iterations):\n")
    
    # Créer DataFrame pour analyse
    df = pd.DataFrame(metrics, columns=[
        'iteration', 'norm_fro', 'norm_spec', 'std', 
        'min', 'max', 'dist_prev', 'asym'
    ])
    
    print("Statistiques globales:")
    print(df.describe())
    
    print(f"\nÉvolution clés:")
    print(f"  Norme initiale: {df.iloc[0]['norm_fro']:.3e}")
    print(f"  Norme finale: {df.iloc[-1]['norm_fro']:.3e}")
    print(f"  Diversité (std) initiale: {df.iloc[0]['std']:.3e}")
    print(f"  Diversité (std) finale: {df.iloc[-1]['std']:.3e}")
    
    # Détecter convergence
    if df['dist_prev'].iloc[-10:].max() < 1e-6:
        print(f"  → Convergence détectée")
    
    # Détecter explosions
    if df['norm_fro'].max() > 1000:
        print(f"  ⚠ Explosion détectée: max_norm={df['norm_fro'].max():.2e}")
    
    conn_raw.close()
    
    print(f"\n{'='*80}\n")


# =============================================================================
# COMPARAISON CONFIGS
# =============================================================================

def compare_configs(gamma_id: str):
    """Compare verdicts pour différentes configs."""
    print(f"\n{'='*80}")
    print(f"COMPARAISON CONFIGS: {gamma_id}")
    print(f"{'='*80}\n")
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    
    cursor.execute("""
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
    """, (gamma_id,))
    
    verdicts = cursor.fetchall()
    
    if not verdicts:
        print(f"❌ Aucun verdict trouvé pour {gamma_id}")
        conn_results.close()
        return
    
    print(f"Verdicts par config/threshold:\n")
    print(f"  {'Config':<20} {'Threshold':<20} {'Verdict':<20} {'Score':<8} {'Maj%':<8} {'Rob%':<8}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    
    for config_id, threshold_id, verdict, score, maj, rob in verdicts:
        print(f"  {config_id:<20} {threshold_id:<20} {verdict:<20} {score:>7.2f}  {maj:>7.1f}  {rob:>7.1f}")
    
    # Analyse variance
    scores = [s for _, _, _, s, _, _ in verdicts]
    if len(scores) > 1:
        print(f"\nVariance scores:")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")
        print(f"  Écart: {max(scores) - min(scores):.2f}")
        print(f"  Std: {np.std(scores):.2f}")
    
    conn_results.close()
    
    print(f"\n{'='*80}\n")


# =============================================================================
# GÉNÉRATION RAPPORT HTML
# =============================================================================

def generate_html_report(gamma_id: str, config_id: str, output_path: str):
    """Génère rapport HTML complet."""
    print(f"\n{'='*80}")
    print(f"GÉNÉRATION RAPPORT HTML: {gamma_id}")
    print(f"{'='*80}\n")
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    html_parts = []
    
    # Header
    html_parts.append(f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport {gamma_id} - {config_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .verdict {{ font-size: 1.5em; font-weight: bold; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .survives {{ background: #4CAF50; color: white; }}
        .wip {{ background: #FFC107; color: black; }}
        .flagged {{ background: #F44336; color: white; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport d'Analyse: {gamma_id}</h1>
        <p><strong>Config:</strong> {config_id} | <strong>Généré:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    """)
    
    # Verdict
    cursor.execute("""
        SELECT verdict, verdict_reason, score_global, majority_pct, robustness_pct
        FROM GammaVerdicts
        WHERE gamma_id = ? AND config_id = ?
        LIMIT 1
    """, (gamma_id, config_id))
    
    verdict_row = cursor.fetchone()
    if verdict_row:
        verdict, reason, score, maj, rob = verdict_row
        verdict_class = verdict.lower().replace('[', '').replace(']', '').replace('-', '')
        
        html_parts.append(f"""
        <div class="verdict {verdict_class}">
            VERDICT: {verdict}
        </div>
        <p><strong>Raison:</strong> {reason}</p>
        
        <h2>Critères d'Évaluation</h2>
        <div class="metric">
            <span class="metric-label">Score Global:</span>
            <span class="metric-value">{score:.2f}/20</span>
        </div>
        <div class="metric">
            <span class="metric-label">Majorité:</span>
            <span class="metric-value">{maj:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Robustesse:</span>
            <span class="metric-value">{rob:.1f}%</span>
        </div>
        """)
    
    # Scores par test
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            ts.weight,
            COUNT(*) as n_runs
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
        GROUP BY ts.test_name, ts.weight
        ORDER BY ts.test_name
    """, (gamma_id, config_id))
    
    html_parts.append("""
        <h2>Scores par Test</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Score Moyen</th>
                <th>Pondération</th>
                <th>N Runs</th>
            </tr>
    """)
    
    for test_name, avg_score, weight, n_runs in cursor.fetchall():
        html_parts.append(f"""
            <tr>
                <td>{test_name}</td>
                <td>{avg_score:.3f}</td>
                <td>{weight:.1f}</td>
                <td>{n_runs}</td>
            </tr>
        """)
    
    html_parts.append("</table>")
    
    # Performance par D
    cursor.execute("""
        SELECT 
            e.d_base_id,
            COUNT(DISTINCT e.id) as n_runs,
            AVG(ts.score) as avg_score
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
        GROUP BY e.d_base_id
        ORDER BY avg_score DESC
    """, (gamma_id, config_id))
    
    html_parts.append("""
        <h2>Performance par Base D</h2>
        <table>
            <tr>
                <th>D Base</th>
                <th>N Runs</th>
                <th>Score Moyen</th>
            </tr>
    """)
    
    for d_base_id, n_runs, avg_score in cursor.fetchall():
        html_parts.append(f"""
            <tr>
                <td>{d_base_id}</td>
                <td>{n_runs}</td>
                <td>{avg_score:.3f}</td>
            </tr>
        """)
    
    html_parts.append("</table>")
    
    # Footer
    html_parts.append("""
    </div>
</body>
</html>
    """)
    
    # Écrire fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    print(f"✓ Rapport généré: {output_path}")
    print(f"\n{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualiseur résultats R0")
    
    # Modes
    parser.add_argument('--overview', action='store_true',
                       help='Vue d\'ensemble complète')
    parser.add_argument('--inspect', type=str,
                       help='Inspecter un Γ spécifique')
    parser.add_argument('--timeline', type=str,
                       help='Évolution temporelle d\'un run (run_id)')
    parser.add_argument('--compare', type=str,
                       help='Comparer configs pour un Γ')
    parser.add_argument('--report', type=str,
                       help='Générer rapport HTML pour un Γ')
    
    # Arguments
    parser.add_argument('--config', type=str, default='weights_default',
                       help='Config scoring')
    parser.add_argument('--output', type=str, default='report.html',
                       help='Fichier sortie pour rapport HTML')
    
    args = parser.parse_args()
    
    if args.overview:
        show_overview()
    
    elif args.inspect:
        inspect_gamma(args.inspect, args.config)
    
    elif args.timeline:
        show_timeline(args.timeline)
    
    elif args.compare:
        compare_configs(args.compare)
    
    elif args.report:
        generate_html_report(args.report, args.config, args.output)
    
    else:
        print("❌ Spécifier une option (--overview, --inspect, etc.)")
        print("\nExemples:")
        print("  python visualize_results.py --overview")
        print("  python visualize_results.py --inspect GAM-001")
        print("  python visualize_results.py --timeline GAM-001_beta2.0_SYM-001_M0_s1")
        print("  python visualize_results.py --compare GAM-001")
        print("  python visualize_results.py --report GAM-001 --output report.html")


if __name__ == "__main__":
    main()