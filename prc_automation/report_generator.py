#!/usr/bin/env python3
"""
prc_automation/report_generator.py

Génération de rapports d'analyse Phase 1.

Usage:
    python report_generator.py --summary
    python report_generator.py --gamma GAM-001
    python report_generator.py --matrix
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime


DB_PATH = Path("prc_database/prc_r0_results.db")


# ============================================================================
# REQUÊTES ANALYSES
# ============================================================================

def get_gamma_summary(conn, gamma_id: str = None) -> List[Dict]:
    """Résumé par Γ."""
    cursor = conn.cursor()
    
    if gamma_id:
        query = """
            SELECT 
                gamma_id,
                gamma_params,
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as pass_count,
                SUM(CASE WHEN global_verdict = 'POOR' THEN 1 ELSE 0 END) as poor_count,
                SUM(CASE WHEN global_verdict = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
                AVG(execution_time_seconds) as avg_time
            FROM Executions
            WHERE gamma_id = ?
            GROUP BY gamma_id, gamma_params
        """
        cursor.execute(query, (gamma_id,))
    else:
        query = """
            SELECT 
                gamma_id,
                gamma_params,
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as pass_count,
                SUM(CASE WHEN global_verdict = 'POOR' THEN 1 ELSE 0 END) as poor_count,
                SUM(CASE WHEN global_verdict = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
                AVG(execution_time_seconds) as avg_time
            FROM Executions
            GROUP BY gamma_id, gamma_params
            ORDER BY gamma_id
        """
        cursor.execute(query)
    
    rows = cursor.fetchall()
    
    return [
        {
            'gamma_id': row[0],
            'gamma_params': json.loads(row[1]),
            'total_runs': row[2],
            'completed': row[3],
            'errors': row[4],
            'pass_count': row[5],
            'poor_count': row[6],
            'rejected_count': row[7],
            'avg_time': row[8],
        }
        for row in rows
    ]


def get_gamma_x_d_matrix(conn, gamma_id: str = None) -> Dict:
    """Matrice Γ × D (taux de succès)."""
    cursor = conn.cursor()
    
    if gamma_id:
        query = """
            SELECT 
                gamma_id,
                gamma_params,
                d_base_id,
                COUNT(*) as n_runs,
                SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as n_pass
            FROM Executions
            WHERE status = 'COMPLETED' AND gamma_id = ?
            GROUP BY gamma_id, gamma_params, d_base_id
            ORDER BY gamma_id, d_base_id
        """
        cursor.execute(query, (gamma_id,))
    else:
        query = """
            SELECT 
                gamma_id,
                gamma_params,
                d_base_id,
                COUNT(*) as n_runs,
                SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as n_pass
            FROM Executions
            WHERE status = 'COMPLETED'
            GROUP BY gamma_id, gamma_params, d_base_id
            ORDER BY gamma_id, d_base_id
        """
        cursor.execute(query)
    
    rows = cursor.fetchall()
    
    # Construire matrice
    matrix = {}
    for row in rows:
        gamma_key = f"{row[0]}_{json.dumps(row[1])}"
        d_base_id = row[2]
        n_runs = row[3]
        n_pass = row[4]
        success_rate = n_pass / n_runs if n_runs > 0 else 0
        
        if gamma_key not in matrix:
            matrix[gamma_key] = {}
        
        matrix[gamma_key][d_base_id] = {
            'n_runs': n_runs,
            'n_pass': n_pass,
            'success_rate': success_rate
        }
    
    return matrix


def get_failing_tests(conn) -> List[Dict]:
    """Tests échouant le plus fréquemment."""
    cursor = conn.cursor()
    
    query = """
        SELECT 
            test_name,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as fails,
            CAST(SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as fail_rate
        FROM TestResults
        GROUP BY test_name
        HAVING fail_rate > 0.1
        ORDER BY fail_rate DESC
        LIMIT 10
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    return [
        {
            'test_name': row[0],
            'total': row[1],
            'fails': row[2],
            'fail_rate': row[3]
        }
        for row in rows
    ]


# ============================================================================
# AFFICHAGE RAPPORTS
# ============================================================================

def print_summary_report(conn):
    """Affiche résumé global."""
    print("\n" + "="*70)
    print("RAPPORT GLOBAL - PHASE 1")
    print("="*70 + "\n")
    
    # Statistiques totales
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Executions")
    total_runs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE status = 'COMPLETED'")
    completed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE status = 'ERROR'")
    errors = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(execution_time_seconds) FROM Executions WHERE status = 'COMPLETED'")
    avg_time = cursor.fetchone()[0] or 0
    
    print(f"Statistiques globales:")
    print(f"  Total exécutions: {total_runs}")
    print(f"  Complétées: {completed} ({100*completed/total_runs:.1f}%)")
    print(f"  Erreurs: {errors} ({100*errors/total_runs:.1f}%)")
    print(f"  Temps moyen: {avg_time:.1f}s")
    
    # Verdicts
    cursor.execute("SELECT global_verdict, COUNT(*) FROM Executions WHERE status = 'COMPLETED' GROUP BY global_verdict")
    verdicts = cursor.fetchall()
    
    print(f"\nVerdicts:")
    for verdict, count in verdicts:
        print(f"  {verdict}: {count} ({100*count/completed:.1f}%)")
    
    # Résumé par Γ
    print(f"\n{'─'*70}")
    print("RÉSUMÉ PAR Γ")
    print(f"{'─'*70}")
    
    summaries = get_gamma_summary(conn)
    
    print(f"\n{'Γ':<12} {'Runs':<8} {'PASS':<8} {'POOR':<8} {'REJ':<8} {'Score':<8}")
    print("─"*70)
    
    for s in summaries:
        if s['completed'] > 0:
            score = s['pass_count'] / s['completed']
            print(f"{s['gamma_id']:<12} {s['completed']:<8} {s['pass_count']:<8} "
                  f"{s['poor_count']:<8} {s['rejected_count']:<8} {score:.1%}")
    
    print("="*70 + "\n")


def print_gamma_report(conn, gamma_id: str):
    """Rapport détaillé pour un Γ."""
    print(f"\n{'='*70}")
    print(f"RAPPORT DÉTAILLÉ - {gamma_id}")
    print(f"{'='*70}\n")
    
    summaries = get_gamma_summary(conn, gamma_id)
    
    if not summaries:
        print(f"❌ Aucune donnée pour {gamma_id}")
        return
    
    for s in summaries:
        print(f"Configuration: {json.dumps(s['gamma_params'])}")
        print(f"  Total runs: {s['total_runs']}")
        print(f"  Complétés: {s['completed']}")
        print(f"  Erreurs: {s['errors']}")
        print(f"  Temps moyen: {s['avg_time']:.1f}s")
        
        if s['completed'] > 0:
            print(f"\n  Verdicts:")
            print(f"    PASS: {s['pass_count']} ({100*s['pass_count']/s['completed']:.1f}%)")
            print(f"    POOR: {s['poor_count']} ({100*s['poor_count']/s['completed']:.1f}%)")
            print(f"    REJECTED: {s['rejected_count']} ({100*s['rejected_count']/s['completed']:.1f}%)")
            
            # Score global
            score = s['pass_count'] / s['completed']
            print(f"\n  Score global: {score:.1%}")
            
            # Classification
            if score >= 0.6:
                print(f"  ✓ Classification: PROMETTEUR")
            elif score >= 0.4:
                print(f"  ○ Classification: AMBIGU")
            else:
                print(f"  ✗ Classification: FAIBLE")
    
    # Matrice D
    print(f"\n{'─'*70}")
    print(f"MATRICE {gamma_id} × D_base")
    print(f"{'─'*70}\n")
    
    matrix = get_gamma_x_d_matrix(conn, gamma_id)
    
    if matrix:
        gamma_key = list(matrix.keys())[0]
        d_results = matrix[gamma_key]
        
        print(f"{'D_base':<12} {'Runs':<8} {'PASS':<8} {'Taux':<8}")
        print("─"*40)
        
        for d_id in sorted(d_results.keys()):
            result = d_results[d_id]
            print(f"{d_id:<12} {result['n_runs']:<8} {result['n_pass']:<8} "
                  f"{result['success_rate']:.1%}")
    
    print("\n" + "="*70 + "\n")


def print_matrix_report(conn):
    """Affiche matrice complète Γ × D."""
    print(f"\n{'='*70}")
    print("MATRICE COMPLÈTE Γ × D (taux de succès)")
    print(f"{'='*70}\n")
    
    matrix = get_gamma_x_d_matrix(conn)
    
    if not matrix:
        print("❌ Aucune donnée")
        return
    
    # Obtenir tous les D uniques
    all_d = set()
    for gamma_results in matrix.values():
        all_d.update(gamma_results.keys())
    all_d = sorted(all_d)
    
    # Header
    print(f"{'Γ':<15}", end="")
    for d_id in all_d:
        print(f"{d_id:<10}", end="")
    print()
    print("─"*70)
    
    # Lignes
    for gamma_key in sorted(matrix.keys()):
        gamma_id = gamma_key.split('_')[0]
        print(f"{gamma_id:<15}", end="")
        
        for d_id in all_d:
            if d_id in matrix[gamma_key]:
                rate = matrix[gamma_key][d_id]['success_rate']
                print(f"{rate:>8.1%}  ", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    print("\n" + "="*70 + "\n")


def print_failing_tests_report(conn):
    """Rapport sur tests échouant fréquemment."""
    print(f"\n{'='*70}")
    print("TESTS ÉCHOUANT LE PLUS FRÉQUEMMENT")
    print(f"{'='*70}\n")
    
    failing = get_failing_tests(conn)
    
    if not failing:
        print("✓ Aucun test échouant fréquemment (>10%)")
        return
    
    print(f"{'Test':<20} {'Total':<10} {'Échecs':<10} {'Taux':<10}")
    print("─"*50)
    
    for test in failing:
        print(f"{test['test_name']:<20} {test['total']:<10} "
              f"{test['fails']:<10} {test['fail_rate']:.1%}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Générateur de rapports Phase 1")
    parser.add_argument('--summary', action='store_true', help='Résumé global')
    parser.add_argument('--gamma', type=str, help='Rapport pour Γ spécifique')
    parser.add_argument('--matrix', action='store_true', help='Matrice Γ×D')
    parser.add_argument('--failing-tests', action='store_true', help='Tests échouant fréquemment')
    parser.add_argument('--all', action='store_true', help='Tous les rapports')
    
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ Base de données non trouvée: {DB_PATH}")
        print("Exécuter d'abord: python batch_runner.py --init-db")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    if args.all or args.summary:
        print_summary_report(conn)
    
    if args.gamma:
        print_gamma_report(conn, args.gamma)
    
    if args.all or args.matrix:
        print_matrix_report(conn)
    
    if args.all or args.failing_tests:
        print_failing_tests_report(conn)
    
    if not (args.summary or args.gamma or args.matrix or args.failing_tests or args.all):
        print("Usage: python report_generator.py --summary | --gamma GAM-XXX | --matrix | --all")
    
    conn.close()


if __name__ == "__main__":
    main()