#!/usr/bin/env python3
"""
prc_automation/query_raw_data.py

Requêtes basiques pour explorer données brutes.
Pas d'interprétation, juste extraction et visualisation.

Usage:
    python query_raw_data.py --summary
    python query_raw_data.py --gamma GAM-001
    python query_raw_data.py --run <run_id>
    python query_raw_data.py --metrics <run_id>
"""

import argparse
import sqlite3
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DB_PATH = Path("prc_database/prc_r0_raw.db")


# ============================================================================
# QUERIES BASIQUES
# ============================================================================

def get_summary(conn) -> pd.DataFrame:
    """Résumé global : nombre de runs par statut."""
    query = """
        SELECT 
            gamma_id,
            COUNT(*) as total_runs,
            SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors,
            AVG(CASE WHEN status = 'COMPLETED' THEN execution_time_seconds END) as avg_time,
            SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END) as converged_count
        FROM Executions
        GROUP BY gamma_id
        ORDER BY gamma_id
    """
    return pd.read_sql_query(query, conn)


def get_gamma_runs(conn, gamma_id: str) -> pd.DataFrame:
    """Tous les runs pour un Γ donné."""
    query = """
        SELECT 
            run_id, d_base_id, modifier_id, seed,
            alpha, beta, gamma_param, omega, sigma,
            final_iteration, converged, convergence_iteration,
            execution_time_seconds, status
        FROM Executions
        WHERE gamma_id = ?
        ORDER BY d_base_id, modifier_id, seed
    """
    return pd.read_sql_query(query, conn, params=(gamma_id,))


def get_run_details(conn, run_id: str) -> Optional[Dict]:
    """Détails complets d'un run."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM Executions WHERE run_id = ?
    """, (run_id,))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def get_metrics(conn, run_id: str) -> pd.DataFrame:
    """Métriques temporelles pour un run."""
    query = """
        SELECT m.*
        FROM Metrics m
        JOIN Executions e ON m.exec_id = e.id
        WHERE e.run_id = ?
        ORDER BY m.iteration
    """
    return pd.read_sql_query(query, conn, params=(run_id,))


def get_snapshots_metadata(conn, run_id: str) -> pd.DataFrame:
    """Métadonnées snapshots (sans charger les états)."""
    query = """
        SELECT 
            s.iteration, s.norm_frobenius, s.norm_spectral,
            s.min_value, s.max_value, s.mean_value, s.std_value
        FROM Snapshots s
        JOIN Executions e ON s.exec_id = e.id
        WHERE e.run_id = ?
        ORDER BY s.iteration
    """
    return pd.read_sql_query(query, conn, params=(run_id,))


def load_snapshot_state(conn, run_id: str, iteration: int) -> Optional[np.ndarray]:
    """Charge un snapshot d'état spécifique."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.state_blob
        FROM Snapshots s
        JOIN Executions e ON s.exec_id = e.id
        WHERE e.run_id = ? AND s.iteration = ?
    """, (run_id, iteration))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    # Décomprésser
    state_compressed = row[0]
    state_bytes = gzip.decompress(state_compressed)
    state = pickle.loads(state_bytes)
    
    return state


def get_param_grid(conn, gamma_id: str) -> pd.DataFrame:
    """Grille de paramètres testés pour un Γ."""
    query = f"""
        SELECT DISTINCT
            alpha, beta, gamma_param, omega,
            memory_weight, window_size, epsilon,
            sigma, lambda_param, eta, subspace_dim
        FROM Executions
        WHERE gamma_id = ?
    """
    return pd.read_sql_query(query, conn, params=(gamma_id,))


# ============================================================================
# AFFICHAGE
# ============================================================================

def print_summary(df: pd.DataFrame):
    """Affiche résumé global."""
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL")
    print("="*70)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    
    print("\n" + "="*70 + "\n")


def print_gamma_runs(df: pd.DataFrame, gamma_id: str):
    """Affiche runs pour un Γ."""
    print(f"\n{'='*70}")
    print(f"RUNS: {gamma_id}")
    print(f"{'='*70}")
    
    if df.empty:
        print("Aucun run trouvé")
        return
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Complétés: {(df['status'] == 'COMPLETED').sum()}")
    print(f"Erreurs: {(df['status'] == 'ERROR').sum()}")
    print(f"Convergés: {df['converged'].sum()}")
    
    print("\nDétails:")
    pd.set_option('display.max_rows', None)
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}\n")


def print_run_details(details: Dict):
    """Affiche détails d'un run."""
    print(f"\n{'='*70}")
    print(f"DÉTAILS RUN: {details['run_id']}")
    print(f"{'='*70}\n")
    
    # Grouper par catégorie
    print("Configuration Γ:")
    print(f"  gamma_id: {details['gamma_id']}")
    for param in ['alpha', 'beta', 'gamma_param', 'omega', 'memory_weight', 
                  'window_size', 'epsilon', 'sigma', 'lambda_param', 'eta', 
                  'subspace_dim']:
        val = details.get(param)
        if val is not None:
            print(f"  {param}: {val}")
    
    print("\nConfiguration D:")
    print(f"  d_base_id: {details['d_base_id']}")
    print(f"  modifier_id: {details['modifier_id']}")
    print(f"  seed: {details['seed']}")
    
    print("\nExécution:")
    print(f"  status: {details['status']}")
    print(f"  final_iteration: {details['final_iteration']}")
    print(f"  converged: {details['converged']}")
    if details['converged']:
        print(f"  convergence_iteration: {details['convergence_iteration']}")
    print(f"  execution_time: {details['execution_time_seconds']:.2f}s")
    
    if details['error_message']:
        print(f"\nErreur:")
        print(f"  {details['error_message']}")
    
    print(f"\n{'='*70}\n")


def print_metrics(df: pd.DataFrame, run_id: str):
    """Affiche statistiques métriques."""
    print(f"\n{'='*70}")
    print(f"MÉTRIQUES: {run_id}")
    print(f"{'='*70}\n")
    
    if df.empty:
        print("Aucune métrique trouvée")
        return
    
    print(f"Nombre d'itérations: {len(df)}")
    print("\nStatistiques:")
    print(df.describe())
    
    print("\nDernières valeurs:")
    print(df.tail(10).to_string(index=False))
    
    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Query données brutes")
    parser.add_argument('--summary', action='store_true',
                       help='Résumé global')
    parser.add_argument('--gamma', type=str,
                       help='Voir runs pour Γ')
    parser.add_argument('--run', type=str,
                       help='Détails d\'un run (run_id)')
    parser.add_argument('--metrics', type=str,
                       help='Métriques d\'un run (run_id)')
    parser.add_argument('--snapshots', type=str,
                       help='Liste snapshots d\'un run (run_id)')
    parser.add_argument('--param-grid', type=str,
                       help='Grille paramètres testés pour Γ')
    
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ Base de données non trouvée: {DB_PATH}")
        print("Lancer d'abord: python batch_runner_raw.py --init-db")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        if args.summary:
            df = get_summary(conn)
            print_summary(df)
        
        elif args.gamma:
            df = get_gamma_runs(conn, args.gamma)
            print_gamma_runs(df, args.gamma)
        
        elif args.run:
            details = get_run_details(conn, args.run)
            if details:
                print_run_details(details)
            else:
                print(f"❌ Run non trouvé: {args.run}")
        
        elif args.metrics:
            df = get_metrics(conn, args.metrics)
            print_metrics(df, args.metrics)
        
        elif args.snapshots:
            df = get_snapshots_metadata(conn, args.snapshots)
            print(f"\nSnapshots pour {args.snapshots}:")
            print(df.to_string(index=False))
        
        elif args.param_grid:
            df = get_param_grid(conn, args.param_grid)
            print(f"\nGrille paramètres testés pour {args.param_grid}:")
            print(df.to_string(index=False))
        
        else:
            print("❌ Spécifier une option (--summary, --gamma, --run, etc.)")
            print("\nExemples:")
            print("  python query_raw_data.py --summary")
            print("  python query_raw_data.py --gamma GAM-001")
            print("  python query_raw_data.py --run GAM-001_beta2.0_SYM-001_M0_s1")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()