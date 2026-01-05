#!/usr/bin/env python3
# prc_automation/init_databases.py

import argparse
import sqlite3
from pathlib import Path

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")

SCHEMA_RAW_PATH = Path("prc_database/schema_raw.sql")
SCHEMA_RESULTS_PATH = Path("prc_database/schema_results.sql")


def init_db_raw(reset: bool = False):
    """Initialise db_raw (inchangé)."""
    print(f"\n{'='*70}")
    print("INITIALISATION db_raw")
    print(f"{'='*70}\n")
    
    if DB_RAW_PATH.exists():
        if reset:
            print(f"⚠ Suppression {DB_RAW_PATH}...")
            DB_RAW_PATH.unlink()
        else:
            print(f"✓ {DB_RAW_PATH} existe déjà")
            return
    
    DB_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not SCHEMA_RAW_PATH.exists():
        print(f"❌ Schema non trouvé: {SCHEMA_RAW_PATH}")
        return
    
    with open(SCHEMA_RAW_PATH, 'r') as f:
        schema = f.read()
    
    conn = sqlite3.connect(DB_RAW_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    
    print(f"✓ {DB_RAW_PATH} créée")
    print(f"  Tables: Executions, Snapshots, Metrics")


def init_db_results(reset: bool = False):
    """Initialise db_results (nouveau schema 5.4)."""
    print(f"\n{'='*70}")
    print("INITIALISATION db_results (Charter 5.4)")
    print(f"{'='*70}\n")
    
    if DB_RESULTS_PATH.exists():
        if reset:
            print(f"⚠ Suppression {DB_RESULTS_PATH}...")
            DB_RESULTS_PATH.unlink()
        else:
            print(f"✓ {DB_RESULTS_PATH} existe déjà")
            return
    
    DB_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not SCHEMA_RESULTS_PATH.exists():
        print(f"❌ Schema non trouvé: {SCHEMA_RESULTS_PATH}")
        return
    
    with open(SCHEMA_RESULTS_PATH, 'r') as f:
        schema = f.read()
    
    conn = sqlite3.connect(DB_RESULTS_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    
    print(f"✓ {DB_RESULTS_PATH} créée")
    print(f"  Tables: TestObservations")


def check_schemas():
    """Vérifie que les schemas existent."""
    print(f"\n{'='*70}")
    print("VÉRIFICATION SCHEMAS")
    print(f"{'='*70}\n")
    
    if SCHEMA_RAW_PATH.exists():
        print(f"✓ {SCHEMA_RAW_PATH}")
    else:
        print(f"❌ {SCHEMA_RAW_PATH} MANQUANT")
    
    if SCHEMA_RESULTS_PATH.exists():
        print(f"✓ {SCHEMA_RESULTS_PATH}")
    else:
        print(f"❌ {SCHEMA_RESULTS_PATH} MANQUANT")


def main():
    parser = argparse.ArgumentParser(description="Initialise bases de données Charter 5.4")
    parser.add_argument('--reset', action='store_true',
                       help='Supprime et recrée les bases (⚠ PERTE DONNÉES)')
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# INITIALISATION BASES DE DONNÉES")
    print("#"*70)
    
    check_schemas()
    init_db_raw(reset=args.reset)
    init_db_results(reset=args.reset)
    
    print("\n" + "#"*70)
    print("# INITIALISATION TERMINÉE")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()