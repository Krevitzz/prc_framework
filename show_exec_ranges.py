#!/usr/bin/env python3
# scripts/show_exec_ranges.py
"""
Affiche plages exec_id par d_encoding_id.

Usage:
    python scripts/show_exec_ranges.py
    python scripts/show_exec_ranges.py --with-shape
"""

import argparse
import sqlite3
import gzip
import pickle
from collections import defaultdict


DB_RAW_PATH = './prc_automation/prc_database/prc_r0_raw.db'


def load_executions():
    """Charge toutes les exécutions."""
    conn = sqlite3.connect(DB_RAW_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id as exec_id,
            run_id,
            gamma_id,
            d_encoding_id,
            modifier_id,
            seed
        FROM Executions
        ORDER BY id
    """)
    
    executions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return executions


def load_state_shape(exec_id):
    """Charge shape depuis premier snapshot."""
    conn = sqlite3.connect(DB_RAW_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
        LIMIT 1
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    state = pickle.loads(gzip.decompress(row[0]))
    return state.shape


def main():
    parser = argparse.ArgumentParser(
        description="Affiche plages exec_id par d_encoding_id"
    )
    parser.add_argument('--with-shape', action='store_true',
                       help="Charger state_shape depuis snapshots (plus lent)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("PLAGES EXEC_ID PAR D_ENCODING")
    print("=" * 80 + "\n")
    
    print("Chargement exécutions...")
    executions = load_executions()
    print(f"✓ {len(executions)} exécutions chargées\n")
    
    # Grouper par d_encoding_id
    by_encoding = defaultdict(list)
    for exec in executions:
        by_encoding[exec['d_encoding_id']].append(exec)
    
    # Afficher par d_encoding
    print(f"{'D_ENCODING':<15} {'COUNT':<8} {'EXEC_ID RANGE':<20} {'STATE_SHAPE':<15}")
    print("-" * 80)
    
    for d_encoding_id in sorted(by_encoding.keys()):
        execs = by_encoding[d_encoding_id]
        exec_ids = [e['exec_id'] for e in execs]
        
        min_id = min(exec_ids)
        max_id = max(exec_ids)
        count = len(exec_ids)
        
        range_str = f"{min_id} - {max_id}"
        
        # Optionnel : charger state_shape
        if args.with_shape:
            shape = load_state_shape(min_id)
            shape_str = str(shape) if shape else "N/A"
        else:
            shape_str = "(use --with-shape)"
        
        print(f"{d_encoding_id:<15} {count:<8} {range_str:<20} {shape_str:<15}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(by_encoding)} d_encodings, {len(executions)} exécutions")
    print("=" * 80 + "\n")
    
    # Suggestions
    print("💡 Pour tester applicability sur un d_encoding spécifique:")
    print("   python scripts/check_applicability_simple.py --exec-id <exec_id>\n")


if __name__ == "__main__":
    main()