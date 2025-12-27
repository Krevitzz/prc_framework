#!/usr/bin/env python3
"""
prc_automation/validate_data.py

Validation de cohérence des données à tous les niveaux du pipeline.

Usage:
    python validate_data.py --all
    python validate_data.py --raw
    python validate_data.py --results
    python validate_data.py --pipeline GAM-001
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Tuple

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


class ValidationError:
    def __init__(self, level: str, message: str, details: dict = None):
        self.level = level  # "ERROR" | "WARNING" | "INFO"
        self.message = message
        self.details = details or {}
    
    def __repr__(self):
        symbol = "❌" if self.level == "ERROR" else "⚠" if self.level == "WARNING" else "ℹ"
        return f"{symbol} [{self.level}] {self.message}"


# =============================================================================
# VALIDATION DB_RAW
# =============================================================================

def validate_db_raw() -> List[ValidationError]:
    """Valide cohérence db_raw."""
    errors = []
    
    if not DB_RAW_PATH.exists():
        errors.append(ValidationError("ERROR", "db_raw non trouvée"))
        return errors
    
    conn = sqlite3.connect(DB_RAW_PATH)
    cursor = conn.cursor()
    
    # 1. Vérifier tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    required = {'Executions', 'Snapshots', 'Metrics'}
    missing = required - tables
    if missing:
        errors.append(ValidationError("ERROR", f"Tables manquantes: {missing}"))
    
    # 2. Vérifier intégrité Executions
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE gamma_id IS NULL")
    null_gamma = cursor.fetchone()[0]
    if null_gamma > 0:
        errors.append(ValidationError("ERROR", f"{null_gamma} runs avec gamma_id NULL"))
    
    # 3. Vérifier statuts valides
    cursor.execute("SELECT DISTINCT status FROM Executions")
    statuses = {row[0] for row in cursor.fetchall()}
    valid = {'COMPLETED', 'ERROR', 'TIMEOUT'}
    invalid = statuses - valid
    if invalid:
        errors.append(ValidationError("ERROR", f"Statuts invalides: {invalid}"))
    
    # 4. Vérifier snapshots cohérents
    cursor.execute("""
        SELECT COUNT(*) FROM Executions e
        WHERE e.status = 'COMPLETED'
        AND NOT EXISTS (SELECT 1 FROM Snapshots s WHERE s.exec_id = e.id)
    """)
    missing_snaps = cursor.fetchone()[0]
    if missing_snaps > 0:
        errors.append(ValidationError("WARNING", f"{missing_snaps} runs complétés sans snapshots"))
    
    # 5. Vérifier métriques cohérentes
    cursor.execute("""
        SELECT COUNT(*) FROM Metrics
        WHERE min_value > max_value
    """)
    invalid_minmax = cursor.fetchone()[0]
    if invalid_minmax > 0:
        errors.append(ValidationError("ERROR", f"{invalid_minmax} métriques avec min > max"))
    
    # 6. Vérifier pas de NaN/Inf
    cursor.execute("""
        SELECT COUNT(*) FROM Metrics
        WHERE norm_frobenius = 'nan' OR norm_frobenius = 'inf'
           OR min_value = 'nan' OR max_value = 'inf'
    """)
    nan_metrics = cursor.fetchone()[0]
    if nan_metrics > 0:
        errors.append(ValidationError("WARNING", f"{nan_metrics} métriques avec NaN/Inf"))
    
    conn.close()
    
    if not errors:
        errors.append(ValidationError("INFO", "db_raw: ✓ Toutes validations passées"))
    
    return errors


# =============================================================================
# VALIDATION DB_RESULTS
# =============================================================================

def validate_db_results() -> List[ValidationError]:
    """Valide cohérence db_results."""
    errors = []
    
    if not DB_RESULTS_PATH.exists():
        errors.append(ValidationError("ERROR", "db_results non trouvée"))
        return errors
    
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    
    # 1. Vérifier tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    required = {'TestObservations', 'TestScores', 'GammaVerdicts'}
    missing = required - tables
    if missing:
        errors.append(ValidationError("ERROR", f"Tables manquantes: {missing}"))
    
    # 2. Vérifier scores dans [0, 1]
    cursor.execute("""
        SELECT COUNT(*) FROM TestScores
        WHERE score < 0 OR score > 1
    """)
    invalid_scores = cursor.fetchone()[0]
    if invalid_scores > 0:
        errors.append(ValidationError("ERROR", f"{invalid_scores} scores hors [0,1]"))
    
    # 3. Vérifier verdicts valides
    cursor.execute("SELECT DISTINCT verdict FROM GammaVerdicts")
    verdicts = {row[0] for row in cursor.fetchall()}
    valid = {'SURVIVES[R0]', 'WIP[R0-closed]', 'FLAGGED_FOR_REVIEW'}
    invalid = verdicts - valid
    if invalid:
        errors.append(ValidationError("ERROR", f"Verdicts invalides: {invalid}"))
    
    # 4. Vérifier intégrité référentielle (avec db_raw)
    if DB_RAW_PATH.exists():
        cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
        
        cursor.execute("""
            SELECT COUNT(*) FROM TestObservations to2
            WHERE NOT EXISTS (
                SELECT 1 FROM db_raw.Executions e WHERE e.id = to2.exec_id
            )
        """)
        orphan_obs = cursor.fetchone()[0]
        if orphan_obs > 0:
            errors.append(ValidationError("ERROR", f"{orphan_obs} observations orphelines"))
        
        cursor.execute("""
            SELECT COUNT(*) FROM TestScores ts
            WHERE NOT EXISTS (
                SELECT 1 FROM db_raw.Executions e WHERE e.id = ts.exec_id
            )
        """)
        orphan_scores = cursor.fetchone()[0]
        if orphan_scores > 0:
            errors.append(ValidationError("ERROR", f"{orphan_scores} scores orphelins"))
        
        cursor.execute("DETACH DATABASE db_raw")
    
    # 5. Vérifier cohérence TestScores ↔ TestObservations
    cursor.execute("""
        SELECT COUNT(*) FROM TestScores ts
        WHERE NOT EXISTS (
            SELECT 1 FROM TestObservations to2 
            WHERE to2.exec_id = ts.exec_id AND to2.test_name = ts.test_name
        )
    """)
    missing_obs = cursor.fetchone()[0]
    if missing_obs > 0:
        errors.append(ValidationError("WARNING", f"{missing_obs} scores sans observation"))
    
    conn.close()
    
    if not errors:
        errors.append(ValidationError("INFO", "db_results: ✓ Toutes validations passées"))
    
    return errors


# =============================================================================
# VALIDATION PIPELINE COMPLET
# =============================================================================

def validate_pipeline(gamma_id: str) -> List[ValidationError]:
    """Valide cohérence du pipeline pour un Γ."""
    errors = []
    
    if not DB_RAW_PATH.exists() or not DB_RESULTS_PATH.exists():
        errors.append(ValidationError("ERROR", "Bases de données manquantes"))
        return errors
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # 1. Vérifier runs existent
    cursor.execute("""
        SELECT COUNT(*) FROM db_raw.Executions
        WHERE gamma_id = ? AND status = 'COMPLETED'
    """, (gamma_id,))
    n_runs = cursor.fetchone()[0]
    
    if n_runs == 0:
        errors.append(ValidationError("ERROR", f"Aucun run complété pour {gamma_id}"))
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return errors
    
    # 2. Vérifier tests appliqués
    cursor.execute("""
        SELECT COUNT(DISTINCT to2.exec_id)
        FROM TestObservations to2
        JOIN db_raw.Executions e ON to2.exec_id = e.id
        WHERE e.gamma_id = ?
    """, (gamma_id,))
    n_tested = cursor.fetchone()[0]
    
    if n_tested == 0:
        errors.append(ValidationError("WARNING", f"Aucun test appliqué pour {gamma_id}"))
    elif n_tested < n_runs:
        errors.append(ValidationError("WARNING", f"Seulement {n_tested}/{n_runs} runs testés"))
    
    # 3. Vérifier scores calculés
    cursor.execute("""
        SELECT COUNT(DISTINCT ts.exec_id)
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ?
    """, (gamma_id,))
    n_scored = cursor.fetchone()[0]
    
    if n_scored == 0:
        errors.append(ValidationError("WARNING", f"Aucun score calculé pour {gamma_id}"))
    elif n_scored < n_tested:
        errors.append(ValidationError("WARNING", f"Seulement {n_scored}/{n_tested} runs scorés"))
    
    # 4. Vérifier verdict existe
    cursor.execute("""
        SELECT COUNT(*) FROM GammaVerdicts
        WHERE gamma_id = ?
    """, (gamma_id,))
    n_verdicts = cursor.fetchone()[0]
    
    if n_verdicts == 0:
        errors.append(ValidationError("INFO", f"Aucun verdict pour {gamma_id} (normal si pas calculé)"))
    
    # 5. Vérifier cohérence scores
    cursor.execute("""
        SELECT 
            ts.test_name,
            COUNT(*) as n_scores,
            AVG(ts.score) as avg_score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ?
        GROUP BY ts.test_name
    """, (gamma_id,))
    
    test_stats = cursor.fetchall()
    for test_name, n_scores, avg_score in test_stats:
        if n_scores < n_runs * 0.5:
            errors.append(ValidationError(
                "WARNING", 
                f"Test {test_name}: seulement {n_scores}/{n_runs} scores"
            ))
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    if not errors:
        errors.append(ValidationError("INFO", f"{gamma_id}: ✓ Pipeline cohérent"))
    
    return errors


# =============================================================================
# VALIDATION CRITÈRES VERDICT
# =============================================================================

def validate_verdict_criteria(gamma_id: str, config_id: str) -> List[ValidationError]:
    """Valide calcul des 3 critères."""
    errors = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Récupérer verdict
    cursor.execute("""
        SELECT score_global, majority_pct, robustness_pct,
               n_total_configs, n_pass_configs,
               n_total_d_bases, n_viable_d_bases
        FROM GammaVerdicts
        WHERE gamma_id = ? AND config_id = ?
    """, (gamma_id, config_id))
    
    row = cursor.fetchone()
    if not row:
        errors.append(ValidationError("INFO", f"Verdict non calculé pour {gamma_id}/{config_id}"))
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return errors
    
    score_global, maj_pct, rob_pct, n_total, n_pass, n_d_total, n_d_viable = row
    
    # Vérifier bornes
    if not (0 <= score_global <= 20):
        errors.append(ValidationError("ERROR", f"Score global hors [0,20]: {score_global}"))
    
    if not (0 <= maj_pct <= 100):
        errors.append(ValidationError("ERROR", f"Majorité hors [0,100]: {maj_pct}"))
    
    if not (0 <= rob_pct <= 100):
        errors.append(ValidationError("ERROR", f"Robustesse hors [0,100]: {rob_pct}"))
    
    # Vérifier cohérence statistiques
    if n_pass > n_total:
        errors.append(ValidationError("ERROR", f"n_pass ({n_pass}) > n_total ({n_total})"))
    
    if n_d_viable > n_d_total:
        errors.append(ValidationError("ERROR", f"n_d_viable ({n_d_viable}) > n_d_total ({n_d_total})"))
    
    # Recalculer majorité et vérifier
    calc_maj = (n_pass / n_total * 100) if n_total > 0 else 0
    if abs(calc_maj - maj_pct) > 0.1:
        errors.append(ValidationError(
            "WARNING", 
            f"Majorité incohérente: stockée={maj_pct:.1f}, calculée={calc_maj:.1f}"
        ))
    
    # Recalculer robustesse et vérifier
    calc_rob = (n_d_viable / n_d_total * 100) if n_d_total > 0 else 0
    if abs(calc_rob - rob_pct) > 0.1:
        errors.append(ValidationError(
            "WARNING",
            f"Robustesse incohérente: stockée={rob_pct:.1f}, calculée={calc_rob:.1f}"
        ))
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    if not errors:
        errors.append(ValidationError("INFO", f"Critères cohérents pour {gamma_id}/{config_id}"))
    
    return errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validation données R0")
    
    parser.add_argument('--all', action='store_true',
                       help='Valide tout')
    parser.add_argument('--raw', action='store_true',
                       help='Valide db_raw')
    parser.add_argument('--results', action='store_true',
                       help='Valide db_results')
    parser.add_argument('--pipeline', type=str,
                       help='Valide pipeline pour un Γ')
    parser.add_argument('--verdict', type=str,
                       help='Valide critères verdict pour un Γ')
    parser.add_argument('--config', type=str, default='weights_default',
                       help='Config pour validation verdict')
    
    args = parser.parse_args()
    
    all_errors = []
    
    if args.all or args.raw:
        print("\n" + "="*80)
        print("VALIDATION db_raw")
        print("="*80)
        errors = validate_db_raw()
        for error in errors:
            print(error)
        all_errors.extend(errors)
    
    if args.all or args.results:
        print("\n" + "="*80)
        print("VALIDATION db_results")
        print("="*80)
        errors = validate_db_results()
        for error in errors:
            print(error)
        all_errors.extend(errors)
    
    if args.pipeline:
        print(f"\n{'='*80}")
        print(f"VALIDATION PIPELINE: {args.pipeline}")
        print("="*80)
        errors = validate_pipeline(args.pipeline)
        for error in errors:
            print(error)
        all_errors.extend(errors)
    
    if args.verdict:
        print(f"\n{'='*80}")
        print(f"VALIDATION VERDICT: {args.verdict}")
        print("="*80)
        errors = validate_verdict_criteria(args.verdict, args.config)
        for error in errors:
            print(error)
        all_errors.extend(errors)
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ VALIDATION")
    print("="*80)
    
    n_errors = sum(1 for e in all_errors if e.level == "ERROR")
    n_warnings = sum(1 for e in all_errors if e.level == "WARNING")
    n_info = sum(1 for e in all_errors if e.level == "INFO")
    
    print(f"Erreurs: {n_errors}")
    print(f"Warnings: {n_warnings}")
    print(f"Info: {n_info}")
    
    if n_errors > 0:
        print("\n❌ Validation échouée")
        exit(1)
    elif n_warnings > 0:
        print("\n⚠ Validation avec warnings")
        exit(0)
    else:
        print("\n✅ Validation réussie")
        exit(0)


if __name__ == "__main__":
    main()