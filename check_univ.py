#!/usr/bin/env python3
# diagnostic_complet.py
"""
Diagnostic complet du framework PRC 5.5
"""

import sys
import sqlite3
from pathlib import Path

print("=" * 70)
print("DIAGNOSTIC COMPLET PRC 5.5")
print("=" * 70)

# =============================================================================
# 1. TESTS FICHIERS
# =============================================================================
print("\n1. FICHIERS TESTS")
print("-" * 70)

tests_dir = Path("tests")
test_files = list(tests_dir.glob("test_*.py"))
test_files = [f for f in test_files if "_deprecated" not in f.stem]

print(f"Fichiers test_*.py trouvés : {len(test_files)}")
for f in sorted(test_files):
    print(f"  - {f.name}")

# =============================================================================
# 2. TESTS DÉCOUVERTS
# =============================================================================
print("\n2. TESTS DÉCOUVERTS PAR FRAMEWORK")
print("-" * 70)

try:
    from tests.utilities.discovery import discover_active_tests
    
    active_tests = discover_active_tests()
    print(f"Tests découverts : {len(active_tests)}")
    
    for test_id in sorted(active_tests.keys()):
        module = active_tests[test_id]
        print(f"  - {test_id:12} (version {module.TEST_VERSION}, "
              f"category {module.TEST_CATEGORY})")
    
    if len(active_tests) < len(test_files):
        print(f"\n⚠️ PROBLÈME : {len(test_files)} fichiers mais "
              f"{len(active_tests)} découverts !")
        
except Exception as e:
    print(f"❌ ERREUR découverte : {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 3. APPLICABILITÉ UNIV-001
# =============================================================================
print("\n3. TEST APPLICABILITÉ UNIV-001")
print("-" * 70)

try:
    from tests.utilities.applicability import check as check_applicability
    import tests.test_uni_001 as univ_001
    
    context = {
        'gamma_id': 'GAM-001',
        'd_encoding_id': 'ASY-001',
        'modifier_id': 'M0',
        'seed': 1,
        'state_shape': (50, 50),
        'exec_id': 1
    }
    
    applicable, reason = check_applicability(univ_001, context)
    
    print(f"Context test : matrice {context['state_shape']}")
    print(f"APPLICABILITY_SPEC : {univ_001.APPLICABILITY_SPEC}")
    print(f"\nRésultat : {applicable}")
    if not applicable:
        print(f"Raison rejet : {reason}")
    else:
        print("✓ Test devrait s'exécuter")
        
except Exception as e:
    print(f"❌ ERREUR applicabilité : {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 4. NOMENCLATURE d_base_id vs d_encoding_id
# =============================================================================
print("\n4. NOMENCLATURE (d_base_id vs d_encoding_id)")
print("-" * 70)

try:
    conn = sqlite3.connect('prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    # Vérifier colonnes Executions
    cursor.execute("PRAGMA table_info(Executions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    print("Colonnes table Executions :")
    relevant = [c for c in columns if 'base' in c or 'encoding' in c]
    for col in relevant:
        print(f"  - {col}")
    
    if 'd_base_id' in columns and 'd_encoding_id' not in columns:
        print("\n⚠️ INCOHÉRENT : Schema utilise 'd_base_id' au lieu de 'd_encoding_id'")
        print("   Charter 5.5 Section 11-B exige 'd_encoding_id'")
    
    conn.close()
    
except Exception as e:
    print(f"❌ ERREUR nomenclature : {e}")

# =============================================================================
# 5. STATISTIQUES DB_RAW
# =============================================================================
print("\n5. CONTENU DB_RAW")
print("-" * 70)

try:
    conn = sqlite3.connect('prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    # Total executions
    cursor.execute("SELECT COUNT(*) FROM Executions")
    total_exec = cursor.fetchone()[0]
    print(f"Total executions : {total_exec}")
    
    # Par gamma
    cursor.execute("""
        SELECT gamma_id, COUNT(*) 
        FROM Executions 
        GROUP BY gamma_id
    """)
    print("\nPar gamma_id :")
    for row in cursor.fetchall():
        print(f"  {row[0]} : {row[1]} runs")
    
    # Par d_base_id (ou d_encoding_id)
    col_name = 'd_base_id' if 'd_base_id' in columns else 'd_encoding_id'
    cursor.execute(f"""
        SELECT {col_name}, COUNT(*) 
        FROM Executions 
        GROUP BY {col_name}
    """)
    print(f"\nPar {col_name} :")
    for row in cursor.fetchall():
        print(f"  {row[0]} : {row[1]} runs")
    
    conn.close()
    
except Exception as e:
    print(f"❌ ERREUR db_raw : {e}")

# =============================================================================
# 6. STATISTIQUES DB_RESULTS
# =============================================================================
print("\n6. CONTENU DB_RESULTS")
print("-" * 70)

try:
    conn = sqlite3.connect('prc_automation/prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    # Total observations
    cursor.execute("SELECT COUNT(*) FROM TestObservations")
    total_obs = cursor.fetchone()[0]
    print(f"Total observations : {total_obs}")
    
    # Par test
    cursor.execute("""
        SELECT test_name, COUNT(*) 
        FROM TestObservations 
        GROUP BY test_name
        ORDER BY test_name
    """)
    print("\nPar test :")
    for row in cursor.fetchall():
        print(f"  {row[0]:12} : {row[1]} observations")
    
    # Par status
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM TestObservations 
        GROUP BY status
    """)
    print("\nPar status :")
    for row in cursor.fetchall():
        print(f"  {row[0]:20} : {row[1]}")
    
    # Ratio observations/executions
    expected = total_exec * len(active_tests) if 'active_tests' in locals() else 0
    if expected > 0:
        ratio = (total_obs / expected) * 100
        print(f"\nRatio : {total_obs}/{expected} = {ratio:.1f}%")
        if ratio < 50:
            print("⚠️ PROBLÈME : Moins de 50% des observations attendues !")
    
    conn.close()
    
except Exception as e:
    print(f"❌ ERREUR db_results : {e}")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

issues = []

if len(test_files) != len(active_tests):
    issues.append(f"Discovery : {len(test_files)} fichiers, {len(active_tests)} découverts")

#if 'd_base_id' in columns and 'd_encoding_id' not in columns:
#    issues.append("Nomenclature : Utilise d_base_id au lieu de d_encoding_id")

if 'ratio' in locals() and ratio < 50:
    issues.append(f"Observations manquantes : {ratio:.1f}% au lieu de 100%")

if issues:
    print("\n⚠️ PROBLÈMES DÉTECTÉS :")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ Aucun problème majeur détecté")

print("\n" + "=" * 70)