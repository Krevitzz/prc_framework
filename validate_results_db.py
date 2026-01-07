#!/usr/bin/env python3
# validate_results_db.py
"""
Script de validation prc_r0_results.db

Vérifie :
1. Distribution statuts observations
2. Cardinalité par test
3. Unicité (exec_id, test_name, params_config_id)
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


DB_PATH = Path("prc_automation/prc_database/prc_r0_results.db")


def check_db_exists() -> bool:
    """Vérifie existence base de données."""
    if not DB_PATH.exists():
        print(f"❌ Base de données introuvable: {DB_PATH}")
        return False
    print(f"✓ Base trouvée: {DB_PATH}\n")
    return True


def check_1_status_distribution(conn: sqlite3.Connection):
    """Check 1 : Distribution des statuts."""
    print("="*70)
    print("CHECK 1 : DISTRIBUTION STATUTS")
    print("="*70 + "\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM TestObservations
        GROUP BY status
        ORDER BY count DESC
    """)
    
    results = cursor.fetchall()
    
    if not results:
        print("⚠ Aucune observation trouvée dans TestObservations\n")
        return
    
    total = sum(count for _, count in results)
    
    print(f"Total observations : {total}\n")
    print(f"{'Statut':<20} {'Count':>10} {'%':>8}")
    print("-"*40)
    
    status_counts = {}
    for status, count in results:
        pct = (count / total) * 100
        print(f"{status:<20} {count:>10} {pct:>7.1f}%")
        status_counts[status] = count
    
    print("\n" + "-"*40)
    
    # Diagnostic
    print("\n📊 DIAGNOSTIC :")
    
    success_count = status_counts.get('SUCCESS', 0)
    error_count = status_counts.get('ERROR', 0)
    not_applicable_count = status_counts.get('NOT_APPLICABLE', 0)
    
    if success_count / total > 0.7:
        print("  ✓ Majorité SUCCESS (bon)")
    elif success_count / total > 0.5:
        print("  ⚠ SUCCESS modéré (acceptable)")
    else:
        print("  ❌ Trop peu de SUCCESS (problème)")
    
    if error_count / total > 0.1:
        print(f"  ❌ Trop d'ERROR ({error_count}) - investiguer logs")
    elif error_count > 0:
        print(f"  ⚠ Quelques ERROR ({error_count}) - vérifier si acceptable")
    else:
        print("  ✓ Aucune erreur")
    
    if not_applicable_count > 0:
        print(f"  ✓ {not_applicable_count} NOT_APPLICABLE (normal)")
    
    print()


def check_2_cardinality_per_test(conn: sqlite3.Connection):
    """Check 2 : Cardinalité par test."""
    print("="*70)
    print("CHECK 2 : CARDINALITÉ PAR TEST")
    print("="*70 + "\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT test_name, COUNT(*) as count
        FROM TestObservations
        GROUP BY test_name
        ORDER BY test_name
    """)
    
    results = cursor.fetchall()
    
    if not results:
        print("⚠ Aucune observation trouvée\n")
        return
    
    print(f"{'Test':<15} {'Count':>10}")
    print("-"*27)
    
    counts = [count for _, count in results]
    avg_count = sum(counts) / len(counts)
    min_count = min(counts)
    max_count = max(counts)
    
    for test_name, count in results:
        marker = ""
        if count < avg_count * 0.5:
            marker = " ⚠ (faible)"
        elif count > avg_count * 2:
            marker = " ⚠ (élevé)"
        
        print(f"{test_name:<15} {count:>10}{marker}")
    
    print("-"*27)
    print(f"{'Moyenne':<15} {avg_count:>10.1f}")
    print(f"{'Min':<15} {min_count:>10}")
    print(f"{'Max':<15} {max_count:>10}")
    
    # Diagnostic
    print("\n📊 DIAGNOSTIC :")
    
    if max_count > avg_count * 3:
        print(f"  ⚠ Disparité élevée détectée (max={max_count}, avg={avg_count:.1f})")
        print("    → Vérifier pourquoi certains tests ont beaucoup plus d'observations")
    
    if min_count < avg_count * 0.3:
        print(f"  ⚠ Certains tests ont peu d'observations (min={min_count})")
        print("    → Vérifier applicabilité réelle ou bugs filtrage")
    
    if max_count / min_count < 2:
        print("  ✓ Distribution homogène entre tests")
    
    print()


def check_3_uniqueness(conn: sqlite3.Connection):
    """Check 3 : Unicité (exec_id, test_name, params_config_id)."""
    print("="*70)
    print("CHECK 3 : UNICITÉ (exec_id, test_name, params_config_id)")
    print("="*70 + "\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT exec_id, test_name, params_config_id, COUNT(*) as duplicates
        FROM TestObservations
        GROUP BY exec_id, test_name, params_config_id
        HAVING COUNT(*) > 1
    """)
    
    results = cursor.fetchall()
    
    if not results:
        print("✓ AUCUN DOUBLON détecté")
        print("  Toutes les observations sont uniques (exec_id, test_name, params)\n")
        return
    
    print(f"❌ {len(results)} DOUBLONS détectés !\n")
    print(f"{'exec_id':<10} {'test_name':<15} {'params_config_id':<20} {'Count':>8}")
    print("-"*60)
    
    for exec_id, test_name, params_config_id, count in results[:10]:
        print(f"{exec_id:<10} {test_name:<15} {params_config_id:<20} {count:>8}")
    
    if len(results) > 10:
        print(f"... et {len(results) - 10} autres doublons")
    
    print("\n📊 DIAGNOSTIC :")
    print("  ❌ Doublons détectés = BUG dans batch_runner")
    print("  → La clause INSERT OR REPLACE ne fonctionne pas correctement")
    print("  → Vérifier UNIQUE constraint dans schema_results.sql")
    print()


def check_4_bonus_global_stats(conn: sqlite3.Connection):
    """Check bonus : Statistiques globales."""
    print("="*70)
    print("BONUS : STATISTIQUES GLOBALES")
    print("="*70 + "\n")
    
    cursor = conn.cursor()
    
    # Nombre total observations
    cursor.execute("SELECT COUNT(*) FROM TestObservations")
    total_obs = cursor.fetchone()[0]
    
    # Nombre exec_id uniques
    cursor.execute("SELECT COUNT(DISTINCT exec_id) FROM TestObservations")
    unique_execs = cursor.fetchone()[0]
    
    # Nombre tests uniques
    cursor.execute("SELECT COUNT(DISTINCT test_name) FROM TestObservations")
    unique_tests = cursor.fetchone()[0]
    
    # Nombre params configs uniques
    cursor.execute("SELECT COUNT(DISTINCT params_config_id) FROM TestObservations")
    unique_params = cursor.fetchone()[0]
    
    print(f"Total observations       : {total_obs}")
    print(f"Runs uniques (exec_id)   : {unique_execs}")
    print(f"Tests uniques            : {unique_tests}")
    print(f"Configs params uniques   : {unique_params}")
    
    if total_obs > 0 and unique_execs > 0:
        avg_obs_per_exec = total_obs / unique_execs
        print(f"Moyenne obs/run          : {avg_obs_per_exec:.1f}")
    
    print()


def main():
    print("\n" + "#"*70)
    print("# VALIDATION prc_r0_results.db")
    print("#"*70 + "\n")
    
    if not check_db_exists():
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        check_1_status_distribution(conn)
        check_2_cardinality_per_test(conn)
        check_3_uniqueness(conn)
        check_4_bonus_global_stats(conn)
        
        conn.close()
        
        print("#"*70)
        print("# VALIDATION TERMINÉE")
        print("#"*70 + "\n")
    
    except sqlite3.Error as e:
        print(f"\n❌ Erreur SQLite : {e}\n")
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}\n")


if __name__ == "__main__":
    main()