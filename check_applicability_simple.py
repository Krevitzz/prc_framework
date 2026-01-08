#!/usr/bin/env python3
# scripts/check_applicability_simple.py
"""
Liste tests applicables par contexte d'exécution.

Simule la phase discovery + applicability du batch_runner SANS exécuter tests.
Permet de voir quels tests seraient exécutés avec l'applicability actuelle.

Usage:
    python scripts/check_applicability_simple.py
    python scripts/check_applicability_simple.py --exec-id 15
"""

import argparse
import sqlite3
import gzip
import pickle
from collections import defaultdict

# Imports framework
from tests.utilities.discovery import discover_active_tests
from tests.utilities.applicability import check as check_applicability


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_RAW_PATH = './prc_automation/prc_database/prc_r0_raw.db'


# =============================================================================
# CHARGEMENT CONTEXTES
# =============================================================================

def load_executions():
    """Charge contextes exécutions depuis db_raw."""
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
    
    executions = []
    for row in cursor.fetchall():
        executions.append({
            'exec_id': row['exec_id'],
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_encoding_id': row['d_encoding_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed'],
        })
    
    conn.close()
    return executions


def load_first_snapshot(exec_id):
    """
    Charge premier snapshot pour déduire state_shape.
    
    Identique à batch_runner.load_first_snapshot()
    """
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
        raise ValueError(f"Aucun snapshot pour exec_id={exec_id}")
    
    # Décompresser
    state = pickle.loads(gzip.decompress(row[0]))
    return state


# =============================================================================
# APPLICABILITY CHECK
# =============================================================================

def check_tests_for_execution(execution, all_tests):
    """
    Vérifie quels tests sont applicables pour un contexte donné.
    
    Identique à la logique batch_runner ligne 107-119.
    """
    # Charger premier snapshot pour déduire state_shape
    first_snapshot = load_first_snapshot(execution['exec_id'])
    
    context = {
        'gamma_id': execution['gamma_id'],
        'd_encoding_id': execution['d_encoding_id'],
        'modifier_id': execution['modifier_id'],
        'seed': execution['seed'],
        'state_shape': first_snapshot.shape,
        'exec_id': execution['exec_id'],  # Pour traçabilité
    }
    
    applicable_tests = {}
    not_applicable_tests = {}
    
    for test_id, test_module in all_tests.items():
        is_applicable, reason = check_applicability(test_module, context)
        
        if is_applicable:
            applicable_tests[test_id] = test_module
        else:
            not_applicable_tests[test_id] = reason
    
    return applicable_tests, not_applicable_tests


# =============================================================================
# RAPPORT
# =============================================================================

def generate_summary(all_tests, executions, results_by_exec):
    """Génère résumé global."""
    
    print("\n" + "=" * 80)
    print("APPLICABILITY SUMMARY")
    print("=" * 80 + "\n")
    
    # Stats globales
    total_tests = len(all_tests)
    total_execs = len(executions)
    
    print(f"Tests découverts:      {total_tests}")
    print(f"Exécutions analysées:  {total_execs}")
    print()
    
    # Stats par test
    print("Tests applicables par contexte:")
    print("-" * 80)
    
    test_applicable_counts = defaultdict(int)
    
    for exec_id, (applicable, _) in results_by_exec.items():
        for test_id in applicable.keys():
            test_applicable_counts[test_id] += 1
    
    for test_id in sorted(all_tests.keys()):
        count = test_applicable_counts[test_id]
        percentage = (count / total_execs) * 100
        print(f"  {test_id:12s} : {count:4d}/{total_execs:4d} exécutions ({percentage:5.1f}%)")
    
    print()
    
    # Analyser raisons de rejet (pour comprendre les 16.7% filtrés)
    rejection_reasons = defaultdict(int)
    
    for exec_id, (applicable, not_applicable) in results_by_exec.items():
        if not_applicable:  # Si au moins un test rejeté
            # Prendre raison du premier test (ils ont tous la même)
            first_reason = list(not_applicable.values())[0]
            rejection_reasons[first_reason] += 1
    
    if rejection_reasons:
        print("Raisons filtrage (exécutions NOT_APPLICABLE):")
        print("-" * 80)
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            percentage = (count / total_execs) * 100
            print(f"  {reason:50s} : {count:4d} exécutions ({percentage:5.1f}%)")
        print()
    
    # Tests jamais applicables
    never_applicable = [
        test_id for test_id in all_tests.keys() 
        if test_applicable_counts[test_id] == 0
    ]
    
    if never_applicable:
        print("⚠️  Tests JAMAIS applicables:")
        for test_id in never_applicable:
            print(f"  - {test_id}")
        print()
    
    # Tests toujours applicables
    always_applicable = [
        test_id for test_id in all_tests.keys()
        if test_applicable_counts[test_id] == total_execs
    ]
    
    if always_applicable:
        print("✓ Tests TOUJOURS applicables:")
        for test_id in always_applicable:
            print(f"  - {test_id}")
        print()
    
    print("=" * 80 + "\n")


def show_execution_detail(execution, applicable_tests, not_applicable_tests):
    """Affiche détail pour une exécution spécifique."""
    
    print("\n" + "=" * 80)
    print(f"EXECUTION DETAIL: exec_id={execution['exec_id']}")
    print("=" * 80 + "\n")
    
    # Charger state_shape depuis snapshot
    first_snapshot = load_first_snapshot(execution['exec_id'])
    state_shape = first_snapshot.shape
    
    print(f"Contexte:")
    print(f"  run_id:         {execution['run_id']}")
    print(f"  gamma_id:       {execution['gamma_id']}")
    print(f"  d_encoding_id:  {execution['d_encoding_id']}")
    print(f"  modifier_id:    {execution['modifier_id']}")
    print(f"  seed:           {execution['seed']}")
    print(f"  state_shape:    {state_shape}")
    print()
    
    print(f"Tests APPLICABLES ({len(applicable_tests)}):")
    if applicable_tests:
        for test_id in sorted(applicable_tests.keys()):
            print(f"  ✓ {test_id}")
    else:
        print("  (aucun)")
    print()
    
    print(f"Tests NOT_APPLICABLE ({len(not_applicable_tests)}):")
    if not_applicable_tests:
        for test_id, reason in sorted(not_applicable_tests.items()):
            print(f"  ✗ {test_id:12s} : {reason}")
    else:
        print("  (aucun)")
    print()
    
    print("=" * 80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Liste tests applicables (discovery + applicability check)",
    )
    
    parser.add_argument('--exec-id', type=int, default=None,
                       help="Afficher détail pour exec_id spécifique")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("APPLICABILITY CHECK (baseline actuelle)")
    print("=" * 80 + "\n")
    
    # Étape 1: Discovery
    print("1. Discovery tests actifs...")
    all_tests = discover_active_tests()
    print(f"   ✓ {len(all_tests)} tests découverts\n")
    
    # Étape 2: Charger contextes
    print("2. Chargement contextes exécution (db_raw)...")
    executions = load_executions()
    print(f"   ✓ {len(executions)} exécutions chargées\n")
    
    # Étape 3: Applicability check
    print("3. Vérification applicability pour chaque contexte...")
    
    results_by_exec = {}
    
    for i, execution in enumerate(executions, 1):
        if i % 100 == 0:
            print(f"   Progression: {i}/{len(executions)}")
        
        applicable, not_applicable = check_tests_for_execution(execution, all_tests)
        results_by_exec[execution['exec_id']] = (applicable, not_applicable)
    
    print(f"   ✓ Terminé\n")
    
    # Affichage résultats
    if args.exec_id is not None:
        # Mode détail pour exec_id spécifique
        execution = next((e for e in executions if e['exec_id'] == args.exec_id), None)
        
        if execution is None:
            print(f"❌ exec_id={args.exec_id} non trouvé")
            return
        
        applicable, not_applicable = results_by_exec[args.exec_id]
        show_execution_detail(execution, applicable, not_applicable)
    
    else:
        # Mode résumé global
        generate_summary(all_tests, executions, results_by_exec)


if __name__ == "__main__":
    main()