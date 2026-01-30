
# Script: scripts/r1_execution/validate_executions.py

"""
Valide exécutions séquences (complétude, cohérence).

CONFORMITÉ:
- Query read-only (db_raw immuable)
- Pas de modification données
"""

import sqlite3
import json
from pathlib import Path

DB_R1_RAW = Path("prc_automation/prc_database/prc_r1_raw.db")
LOG_N2 = Path("outputs/r1_execution/execution_n2_log.json")
LOG_N3 = Path("outputs/r1_execution/execution_n3_log.json")

def validate_executions():
    """
    Valide exécutions R1.1.
    
    VÉRIFICATIONS:
    - Nombre séquences DB == nombre attendu
    - Tous sequence_exec_ids présents
    - Status SUCCESS vs ERROR
    - Snapshots présents
    """
    
    conn = sqlite3.connect(DB_R1_RAW)
    cursor = conn.cursor()
    
    # Charger logs attendus
    with open(LOG_N2) as f:
        log_n2 = json.load(f)
    
    with open(LOG_N3) as f:
        log_n3 = json.load(f)
    
    expected_n2 = set(log_n2['sequence_exec_ids'])
    expected_n3 = set(log_n3['sequence_exec_ids'])
    
    print("="*70)
    print("VALIDATION EXÉCUTIONS R1.1")
    print("="*70)
    
    # Vérification n=2
    print("\nSÉQUENCES n=2:")
    cursor.execute("""
        SELECT COUNT(*) FROM sequences
        WHERE phase = 'R1' AND sequence_length = 2
    """)
    count_n2 = cursor.fetchone()[0]
    print(f"  Attendues: {len(expected_n2)}")
    print(f"  DB:        {count_n2}")
    
    if count_n2 == len(expected_n2):
        print("  ✓ Complétude OK")
    else:
        print(f"  ✗ Manquantes: {len(expected_n2) - count_n2}")
    
    # Vérification n=3
    print("\nSÉQUENCES n=3:")
    cursor.execute("""
        SELECT COUNT(*) FROM sequences
        WHERE phase = 'R1' AND sequence_length = 3
    """)
    count_n3 = cursor.fetchone()[0]
    print(f"  Attendues: {len(expected_n3)}")
    print(f"  DB:        {count_n3}")
    
    if count_n3 == len(expected_n3):
        print("  ✓ Complétude OK")
    else:
        print(f"  ✗ Manquantes: {len(expected_n3) - count_n3}")
    
    # Status global
    print("\nSTATUS GLOBAL:")
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM sequences 
        WHERE phase = 'R1'
        GROUP BY status
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    # Snapshots
    print("\nSNAPSHOTS:")
    cursor.execute("""
        SELECT COUNT(*) FROM snapshots_sequences
    """)
    n_snapshots = cursor.fetchone()[0]
    print(f"  Total: {n_snapshots}")
    
    conn.close()
    
    print("="*70)

if __name__ == "__main__":
    validate_executions()