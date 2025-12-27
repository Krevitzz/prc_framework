#!/usr/bin/env python3
"""
prc_automation/export_for_analysis.py

Exporte les données dans un format compact pour analyse externe.

Usage:
    # Export JSON compact
    python export_for_analysis.py --format json --output data_export.json
    
    # Export CSV pour analyse tableur
    python export_for_analysis.py --format csv --output data_export/
    
    # Export pour Claude (résumé compact)
    python export_for_analysis.py --format claude --output analysis_request.txt
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


# =============================================================================
# EXPORT JSON COMPACT
# =============================================================================

def export_json(output_path: str):
    """Exporte données en JSON structuré compact."""
    print(f"\nExport JSON vers {output_path}...")
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'charter_version': '5.1',
            'phase': 'R0',
        },
        'verdicts': [],
        'gamma_summary': [],
        'test_scores': [],
    }
    
    # Verdicts
    cursor.execute("""
        SELECT 
            gamma_id, config_id, threshold_id, verdict,
            score_global, majority_pct, robustness_pct,
            n_total_configs, n_pass_configs,
            n_total_d_bases, n_viable_d_bases
        FROM GammaVerdicts
        ORDER BY gamma_id, config_id
    """)
    
    for row in cursor.fetchall():
        export_data['verdicts'].append({
            'gamma_id': row[0],
            'config_id': row[1],
            'threshold_id': row[2],
            'verdict': row[3],
            'score_global': row[4],
            'majority_pct': row[5],
            'robustness_pct': row[6],
            'n_total_configs': row[7],
            'n_pass_configs': row[8],
            'n_total_d_bases': row[9],
            'n_viable_d_bases': row[10],
        })
    
    # Résumé par Γ
    cursor.execute("""
        SELECT 
            e.gamma_id,
            COUNT(DISTINCT e.id) as n_runs,
            COUNT(DISTINCT e.d_base_id) as n_d_tested,
            AVG(CASE WHEN e.status = 'COMPLETED' THEN 1.0 ELSE 0.0 END) as completion_rate
        FROM db_raw.Executions e
        GROUP BY e.gamma_id
    """)
    
    for row in cursor.fetchall():
        export_data['gamma_summary'].append({
            'gamma_id': row[0],
            'n_runs': row[1],
            'n_d_tested': row[2],
            'completion_rate': row[3],
        })
    
    # Scores moyens par test et Γ
    cursor.execute("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            AVG(ts.score) as avg_score,
            COUNT(*) as n_runs,
            ts.config_id
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        GROUP BY e.gamma_id, ts.test_name, ts.config_id
    """)
    
    for row in cursor.fetchall():
        export_data['test_scores'].append({
            'gamma_id': row[0],
            'test_name': row[1],
            'avg_score': row[2],
            'n_runs': row[3],
            'config_id': row[4],
        })
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    # Écrire JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exporté {len(export_data['verdicts'])} verdicts")
    print(f"✓ Exporté {len(export_data['gamma_summary'])} Γ")
    print(f"✓ Exporté {len(export_data['test_scores'])} scores de tests")


# =============================================================================
# EXPORT CSV
# =============================================================================

def export_csv(output_dir: str):
    """Exporte données en plusieurs CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nExport CSV vers {output_dir}/...")
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # 1. Verdicts
    df = pd.read_sql_query("""
        SELECT * FROM GammaVerdicts
        ORDER BY gamma_id, config_id
    """, conn_results)
    df.to_csv(output_path / 'verdicts.csv', index=False)
    print(f"✓ verdicts.csv ({len(df)} lignes)")
    
    # 2. Scores par Γ et test
    df = pd.read_sql_query("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            ts.config_id,
            AVG(ts.score) as avg_score,
            MIN(ts.score) as min_score,
            MAX(ts.score) as max_score,
            COUNT(*) as n_runs
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        GROUP BY e.gamma_id, ts.test_name, ts.config_id
    """, conn_results)
    df.to_csv(output_path / 'test_scores_by_gamma.csv', index=False)
    print(f"✓ test_scores_by_gamma.csv ({len(df)} lignes)")
    
    # 3. Performance par Γ et D
    df = pd.read_sql_query("""
        SELECT 
            e.gamma_id,
            e.d_base_id,
            ts.config_id,
            AVG(ts.score) as avg_score,
            COUNT(DISTINCT e.id) as n_runs
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        GROUP BY e.gamma_id, e.d_base_id, ts.config_id
    """, conn_results)
    df.to_csv(output_path / 'performance_by_d.csv', index=False)
    print(f"✓ performance_by_d.csv ({len(df)} lignes)")
    
    # 4. Résumé exécutions
    df = pd.read_sql_query("""
        SELECT 
            gamma_id, d_base_id, modifier_id, seed,
            status, final_iteration, execution_time_seconds,
            converged, convergence_iteration
        FROM db_raw.Executions
    """, conn_results)
    df.to_csv(output_path / 'executions_summary.csv', index=False)
    print(f"✓ executions_summary.csv ({len(df)} lignes)")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()


# =============================================================================
# EXPORT POUR CLAUDE (Format résumé)
# =============================================================================

def export_for_claude(output_path: str):
    """Exporte résumé structuré pour analyse par Claude."""
    print(f"\nExport résumé pour analyse Claude...")
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    lines = []
    
    # Header
    lines.append("="*80)
    lines.append("RÉSUMÉ DONNÉES R0 - DEMANDE D'ANALYSE DE PATTERNS")
    lines.append("="*80)
    lines.append(f"Export: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Vue d'ensemble
    cursor.execute("SELECT COUNT(*) FROM db_raw.Executions")
    n_runs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT gamma_id) FROM db_raw.Executions")
    n_gammas = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM GammaVerdicts")
    n_verdicts = cursor.fetchone()[0]
    
    lines.append("VUE D'ENSEMBLE")
    lines.append("-" * 80)
    lines.append(f"Runs totaux: {n_runs}")
    lines.append(f"Γ testés: {n_gammas}")
    lines.append(f"Verdicts calculés: {n_verdicts}")
    lines.append("")
    
    # Distribution verdicts
    cursor.execute("""
        SELECT verdict, COUNT(*) 
        FROM GammaVerdicts 
        WHERE config_id = 'weights_default' AND threshold_id = 'thresholds_default'
        GROUP BY verdict
    """)
    
    lines.append("DISTRIBUTION VERDICTS (config/threshold par défaut)")
    lines.append("-" * 80)
    for verdict, count in cursor.fetchall():
        lines.append(f"{verdict}: {count}")
    lines.append("")
    
    # Verdicts détaillés par Γ
    cursor.execute("""
        SELECT 
            gamma_id, verdict, score_global, majority_pct, robustness_pct
        FROM GammaVerdicts
        WHERE config_id = 'weights_default' AND threshold_id = 'thresholds_default'
        ORDER BY score_global DESC
    """)
    
    lines.append("VERDICTS PAR Γ")
    lines.append("-" * 80)
    lines.append(f"{'Γ':<12} {'Verdict':<20} {'Score':<8} {'Maj%':<8} {'Rob%':<8}")
    lines.append("-" * 80)
    
    for gamma_id, verdict, score, maj, rob in cursor.fetchall():
        lines.append(f"{gamma_id:<12} {verdict:<20} {score:>7.2f}  {maj:>7.1f}  {rob:>7.1f}")
    lines.append("")
    
    # Scores moyens par test (tous Γ confondus)
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            MIN(ts.score) as min_score,
            MAX(ts.score) as max_score,
            COUNT(DISTINCT e.gamma_id) as n_gammas
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY ts.test_name
        ORDER BY ts.test_name
    """)
    
    lines.append("SCORES MOYENS PAR TEST (tous Γ)")
    lines.append("-" * 80)
    lines.append(f"{'Test':<15} {'Avg':<8} {'Min':<8} {'Max':<8} {'N Γ':<8}")
    lines.append("-" * 80)
    
    for test_name, avg, min_s, max_s, n_g in cursor.fetchall():
        lines.append(f"{test_name:<15} {avg:>7.3f}  {min_s:>7.3f}  {max_s:>7.3f}  {n_g:>7}")
    lines.append("")
    
    # Matrice Γ × Test (scores moyens)
    cursor.execute("""
        SELECT DISTINCT gamma_id FROM db_raw.Executions ORDER BY gamma_id
    """)
    gammas = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("""
        SELECT DISTINCT test_name FROM TestScores ORDER BY test_name
    """)
    tests = [row[0] for row in cursor.fetchall()]
    
    lines.append("MATRICE Γ × TEST (scores moyens)")
    lines.append("-" * 80)
    
    # Header
    header = f"{'Γ':<12}"
    for test in tests[:10]:  # Limiter à 10 tests pour lisibilité
        header += f" {test:<7}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Lignes
    for gamma_id in gammas:
        cursor.execute("""
            SELECT ts.test_name, AVG(ts.score)
            FROM TestScores ts
            JOIN db_raw.Executions e ON ts.exec_id = e.id
            WHERE e.gamma_id = ? AND ts.config_id = 'weights_default'
            GROUP BY ts.test_name
        """, (gamma_id,))
        
        scores_dict = dict(cursor.fetchall())
        
        line = f"{gamma_id:<12}"
        for test in tests[:10]:
            score = scores_dict.get(test, 0.0)
            line += f" {score:>7.3f}"
        lines.append(line)
    lines.append("")
    
    # Questions d'analyse
    lines.append("QUESTIONS POUR ANALYSE")
    lines.append("-" * 80)
    lines.append("1. Y a-t-il des patterns de corrélation entre tests ?")
    lines.append("   (ex: tous les Γ qui échouent TEST-X échouent aussi TEST-Y)")
    lines.append("")
    lines.append("2. Y a-t-il des familles de comportement émergentes ?")
    lines.append("   (ex: Γ markoviens vs non-markoviens)")
    lines.append("")
    lines.append("3. Quels tests sont les plus discriminants ?")
    lines.append("   (variance élevée entre Γ)")
    lines.append("")
    lines.append("4. Y a-t-il des D où tous les Γ échouent ?")
    lines.append("   (ou inversement, où tous réussissent)")
    lines.append("")
    lines.append("5. Peut-on suggérer des contraintes CON-GAM-XXX ?")
    lines.append("   (basées sur patterns d'échec systématiques)")
    lines.append("")
    lines.append("="*80)
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    # Écrire fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Résumé exporté: {output_path}")
    print(f"\nCe fichier peut être copié-collé dans Claude pour analyse de patterns.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export données pour analyse")
    
    parser.add_argument('--format', type=str, required=True,
                       choices=['json', 'csv', 'claude'],
                       help='Format d\'export')
    parser.add_argument('--output', type=str, required=True,
                       help='Fichier ou dossier de sortie')
    
    args = parser.parse_args()
    
    if not DB_RAW_PATH.exists() or not DB_RESULTS_PATH.exists():
        print("❌ Bases de données non trouvées")
        return
    
    if args.format == 'json':
        export_json(args.output)
    
    elif args.format == 'csv':
        export_csv(args.output)
    
    elif args.format == 'claude':
        export_for_claude(args.output)
    
    print("\n✓ Export terminé")


if __name__ == "__main__":
    main()