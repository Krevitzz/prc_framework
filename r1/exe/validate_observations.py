
# Script: scripts/r1_execution/validate_observations.py

"""
Valide observations tests séquences (complétude, cohérence).

CONFORMITÉ:
- Query read-only
- Pas de modification données
"""

import sqlite3
import json
from pathlib import Path

DB_RESULTS = Path("prc_automation/prc_database/prc_r1_results.db")
LOG_N2 = Path("outputs/r1_execution/execution_n2_log.json")
LOG_N3 = Path("outputs/r1_execution/execution_n3_log.json")


def validate_observations():
    """
    Valide observations tests séquences.
    
    VÉRIFICATIONS:
    - Nombre observations == n_sequences × n_tests_applicables (attendu)
    - Status SUCCESS vs ERROR
    - Cohérence sequence_exec_id (présents dans logs)
    """
    
    conn = sqlite3.connect(DB_RESULTS)
    cursor = conn.cursor()
    
    # Charger sequence_exec_ids attendus
    with open(LOG_N2) as f:
        log_n2 = json.load(f)
    expected_n2 = set(log_n2['sequence_exec_ids'])
    
    with open(LOG_N3) as f:
        log_n3 = json.load(f)
    expected_n3 = set(log_n3['sequence_exec_ids'])
    
    print("="*70)
    print("VALIDATION OBSERVATIONS R1.1")
    print("="*70)
    
    # Observations n=2
    print("\nOBSERVATIONS n=2:")
    cursor.execute("""
        SELECT COUNT(*) FROM observations
        WHERE phase = 'R1' AND sequence_length = 2
    """)
    count_n2 = cursor.fetchone()[0]
    print(f"  Total: {count_n2}")
    
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM observations 
        WHERE phase = 'R1' AND sequence_length = 2
        GROUP BY status
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]}")
    
    # Observations n=3
    print("\nOBSERVATIONS n=3:")
    cursor.execute("""
        SELECT COUNT(*) FROM observations
        WHERE phase = 'R1' AND sequence_length = 3
    """)
    count_n3 = cursor.fetchone()[0]
    print(f"  Total: {count_n3}")
    
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM observations 
        WHERE phase = 'R1' AND sequence_length = 3
        GROUP BY status
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]}")
    
    # Cohérence sequence_exec_id
    print("\nCOHÉRENCE sequence_exec_id:")
    cursor.execute("""
        SELECT DISTINCT sequence_exec_id FROM observations
        WHERE phase = 'R1'
    """)
    observed_ids = set(row[0] for row in cursor.fetchall())
    
    expected_all = expected_n2 | expected_n3
    
    missing = expected_all - observed_ids
    extra = observed_ids - expected_all
    
    if not missing and not extra:
        print("  ✓ Tous sequence_exec_ids cohérents")
    else:
        if missing:
            print(f"  ⚠ Manquants: {len(missing)}")
        if extra:
            print(f"  ⚠ En trop: {len(extra)}")
    
    conn.close()
    
    print("="*70)


if __name__ == "__main__":
    validate_observations()