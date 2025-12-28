"""
tests/validation/validate_db_raw.py

Validation de la cohérence de db_raw (Section 14.7).

Vérifie :
  - Intégrité structurelle (tables, indexes)
  - Cohérence données (pas de NaN/Inf inattendus)
  - Complétude (snapshots, métriques)
  - Conformité schema
"""

import sqlite3
import numpy as np
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")


@dataclass
class ValidationResult:
    """Résultat d'une validation."""
    test_name: str
    passed: bool
    message: str
    details: Dict = None
    
    def __repr__(self):
        symbol = "✅" if self.passed else "❌"
        return f"{symbol} {self.test_name}: {self.message}"


# =============================================================================
# VALIDATIONS STRUCTURELLES
# =============================================================================

def validate_tables_exist(conn) -> ValidationResult:
    """Vérifie que toutes les tables existent."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table'
    """)
    tables = {row[0] for row in cursor.fetchall()}
    
    required_tables = {'Executions', 'Snapshots', 'Metrics', 'schema_metadata'}
    missing = required_tables - tables
    
    if missing:
        return ValidationResult(
            "tables_exist",
            False,
            f"Tables manquantes: {missing}"
        )
    
    return ValidationResult(
        "tables_exist",
        True,
        f"Toutes les tables présentes ({len(tables)} tables)"
    )


def validate_indexes_exist(conn) -> ValidationResult:
    """Vérifie que les indexes existent."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='index'
    """)
    indexes = {row[0] for row in cursor.fetchall()}
    
    # Indexes attendus (hors autoindex)
    expected_indexes = {
        'idx_exec_gamma', 'idx_exec_dbase', 'idx_exec_status',
        'idx_exec_params', 'idx_snapshot_exec', 'idx_metrics_exec'
    }
    
    present = expected_indexes & indexes
    missing = expected_indexes - indexes
    
    if missing:
        return ValidationResult(
            "indexes_exist",
            False,
            f"Indexes manquants: {missing}",
            {'present': present, 'missing': missing}
        )
    
    return ValidationResult(
        "indexes_exist",
        True,
        f"{len(present)} indexes présents"
    )


# =============================================================================
# VALIDATIONS DONNÉES
# =============================================================================

def validate_executions_integrity(conn) -> ValidationResult:
    """Vérifie intégrité table Executions."""
    cursor = conn.cursor()
    
    # Nombre total
    cursor.execute("SELECT COUNT(*) FROM Executions")
    n_total = cursor.fetchone()[0]
    
    if n_total == 0:
        return ValidationResult(
            "executions_integrity",
            True,
            "Aucune exécution (DB vide, normal)"
        )
    
    # Statuts
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM Executions 
        GROUP BY status
    """)
    status_counts = dict(cursor.fetchall())
    
    # Vérifier statuts valides
    valid_statuses = {'COMPLETED', 'ERROR', 'TIMEOUT'}
    invalid = set(status_counts.keys()) - valid_statuses
    
    if invalid:
        return ValidationResult(
            "executions_integrity",
            False,
            f"Statuts invalides: {invalid}"
        )
    
    # Vérifier gamma_id non NULL
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE gamma_id IS NULL")
    null_gamma = cursor.fetchone()[0]
    
    if null_gamma > 0:
        return ValidationResult(
            "executions_integrity",
            False,
            f"{null_gamma} exécutions avec gamma_id NULL"
        )
    
    return ValidationResult(
        "executions_integrity",
        True,
        f"{n_total} exécutions, statuts: {status_counts}"
    )


def validate_snapshots_completeness(conn) -> ValidationResult:
    """Vérifie complétude snapshots."""
    cursor = conn.cursor()
    
    # Nombre exécutions complétées
    cursor.execute("""
        SELECT COUNT(*) FROM Executions WHERE status = 'COMPLETED'
    """)
    n_completed = cursor.fetchone()[0]
    
    if n_completed == 0:
        return ValidationResult(
            "snapshots_completeness",
            True,
            "Aucune exécution complétée (DB vide, normal)"
        )
    
    # Vérifier que chaque exécution a des snapshots
    cursor.execute("""
        SELECT e.id, e.run_id, COUNT(s.id) as n_snapshots
        FROM Executions e
        LEFT JOIN Snapshots s ON e.id = s.exec_id
        WHERE e.status = 'COMPLETED'
        GROUP BY e.id, e.run_id
        HAVING n_snapshots = 0
    """)
    missing_snapshots = cursor.fetchall()
    
    if missing_snapshots:
        return ValidationResult(
            "snapshots_completeness",
            False,
            f"{len(missing_snapshots)} exécutions sans snapshots",
            {'examples': missing_snapshots[:5]}
        )
    
    # Statistiques snapshots
    cursor.execute("""
        SELECT 
            COUNT(*) as total_snapshots,
            AVG(cnt) as avg_per_exec,
            MIN(cnt) as min_per_exec,
            MAX(cnt) as max_per_exec
        FROM (
            SELECT exec_id, COUNT(*) as cnt
            FROM Snapshots
            GROUP BY exec_id
        )
    """)
    stats = cursor.fetchone()
    
    return ValidationResult(
        "snapshots_completeness",
        True,
        f"{stats[0]} snapshots, moy={stats[1]:.1f} par exec",
        {'total': stats[0], 'avg': stats[1], 'min': stats[2], 'max': stats[3]}
    )


def validate_snapshots_decompressible(conn) -> ValidationResult:
    """Vérifie que snapshots sont décompressibles."""
    cursor = conn.cursor()
    
    # Échantillon de snapshots
    cursor.execute("""
        SELECT id, exec_id, iteration, state_blob
        FROM Snapshots
        ORDER BY RANDOM()
        LIMIT 10
    """)
    samples = cursor.fetchall()
    
    if not samples:
        return ValidationResult(
            "snapshots_decompressible",
            True,
            "Aucun snapshot à vérifier"
        )
    
    errors = []
    for snap_id, exec_id, iteration, state_blob in samples:
        try:
            state_bytes = gzip.decompress(state_blob)
            state = pickle.loads(state_bytes)
            
            # Vérifier que c'est un np.ndarray
            if not isinstance(state, np.ndarray):
                errors.append(f"snap_id={snap_id}: pas un ndarray")
            
            # Vérifier pas de NaN/Inf
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                errors.append(f"snap_id={snap_id}: contient NaN/Inf")
        
        except Exception as e:
            errors.append(f"snap_id={snap_id}: {str(e)}")
    
    if errors:
        return ValidationResult(
            "snapshots_decompressible",
            False,
            f"{len(errors)}/{len(samples)} snapshots invalides",
            {'errors': errors}
        )
    
    return ValidationResult(
        "snapshots_decompressible",
        True,
        f"{len(samples)} snapshots vérifiés OK"
    )


def validate_metrics_consistency(conn) -> ValidationResult:
    """Vérifie cohérence métriques."""
    cursor = conn.cursor()
    
    # Vérifier valeurs numériques valides (pas de NaN/Inf en string)
    cursor.execute("""
        SELECT COUNT(*) FROM Metrics
        WHERE norm_frobenius IS NULL 
           OR norm_max IS NULL
           OR min_value IS NULL
           OR max_value IS NULL
    """)
    null_metrics = cursor.fetchone()[0]
    
    # Vérifier cohérence min/max
    cursor.execute("""
        SELECT COUNT(*) FROM Metrics
        WHERE min_value > max_value
    """)
    invalid_minmax = cursor.fetchone()[0]
    
    errors = []
    if null_metrics > 0:
        errors.append(f"{null_metrics} métriques avec valeurs NULL")
    if invalid_minmax > 0:
        errors.append(f"{invalid_minmax} métriques avec min > max")
    
    if errors:
        return ValidationResult(
            "metrics_consistency",
            False,
            ", ".join(errors)
        )
    
    # Statistiques
    cursor.execute("SELECT COUNT(*) FROM Metrics")
    n_metrics = cursor.fetchone()[0]
    
    return ValidationResult(
        "metrics_consistency",
        True,
        f"{n_metrics} métriques cohérentes"
    )


# =============================================================================
# VALIDATIONS CONFORMITÉ
# =============================================================================

def validate_append_only(conn) -> ValidationResult:
    """
    Vérifie propriété append-only (indirect).
    
    Note: Impossible de détecter UPDATE/DELETE passés,
    mais on peut vérifier absence de duplications.
    """
    cursor = conn.cursor()
    
    # Vérifier unicité run_id
    cursor.execute("""
        SELECT run_id, COUNT(*) as cnt
        FROM Executions
        GROUP BY run_id
        HAVING cnt > 1
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        return ValidationResult(
            "append_only",
            False,
            f"{len(duplicates)} run_id dupliqués",
            {'examples': duplicates[:5]}
        )
    
    return ValidationResult(
        "append_only",
        True,
        "Pas de duplication détectée"
    )


def validate_foreign_keys(conn) -> ValidationResult:
    """Vérifie intégrité référentielle."""
    cursor = conn.cursor()
    
    # Snapshots → Executions
    cursor.execute("""
        SELECT COUNT(*) FROM Snapshots s
        WHERE NOT EXISTS (
            SELECT 1 FROM Executions e WHERE e.id = s.exec_id
        )
    """)
    orphan_snapshots = cursor.fetchone()[0]
    
    # Metrics → Executions
    cursor.execute("""
        SELECT COUNT(*) FROM Metrics m
        WHERE NOT EXISTS (
            SELECT 1 FROM Executions e WHERE e.id = m.exec_id
        )
    """)
    orphan_metrics = cursor.fetchone()[0]
    
    errors = []
    if orphan_snapshots > 0:
        errors.append(f"{orphan_snapshots} snapshots orphelins")
    if orphan_metrics > 0:
        errors.append(f"{orphan_metrics} metrics orphelins")
    
    if errors:
        return ValidationResult(
            "foreign_keys",
            False,
            ", ".join(errors)
        )
    
    return ValidationResult(
        "foreign_keys",
        True,
        "Intégrité référentielle OK"
    )


# =============================================================================
# SUITE COMPLÈTE
# =============================================================================

def run_all_validations(db_path: Path = DB_RAW_PATH) -> List[ValidationResult]:
    """Exécute toutes les validations."""
    if not db_path.exists():
        return [ValidationResult(
            "db_exists",
            False,
            f"Base de données non trouvée: {db_path}"
        )]
    
    conn = sqlite3.connect(db_path)
    
    results = []
    
    # Validations structurelles
    results.append(validate_tables_exist(conn))
    results.append(validate_indexes_exist(conn))
    
    # Validations données
    results.append(validate_executions_integrity(conn))
    results.append(validate_snapshots_completeness(conn))
    results.append(validate_snapshots_decompressible(conn))
    results.append(validate_metrics_consistency(conn))
    
    # Validations conformité
    results.append(validate_append_only(conn))
    results.append(validate_foreign_keys(conn))
    
    conn.close()
    
    return results


def print_validation_report(results: List[ValidationResult]):
    """Affiche rapport de validation."""
    print("\n" + "="*70)
    print("RAPPORT VALIDATION db_raw")
    print("="*70 + "\n")
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    # Afficher résultats
    for result in results:
        print(result)
        if result.details and not result.passed:
            print(f"  Détails: {result.details}")
        print()
    
    # Résumé
    print("="*70)
    print(f"RÉSUMÉ: {passed} PASS, {failed} FAIL")
    
    if failed == 0:
        print("✅ db_raw est valide")
    else:
        print("❌ db_raw contient des erreurs")
    
    print("="*70 + "\n")
    
    return failed == 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_all_validations()
    valid = print_validation_report(results)
    
    exit(0 if valid else 1)