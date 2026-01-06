#!/usr/bin/env python3
# diagnostic_exec_ids.py
"""
Diagnostic précis : Quels exec_id sont traités vs ignorés ?
"""

import sqlite3

print("=" * 70)
print("DIAGNOSTIC EXEC_IDS - Pourquoi seulement 200 runs ?")
print("=" * 70)

# Connexions
conn_raw = sqlite3.connect('prc_automation/prc_database/prc_r0_raw.db')
conn_results = sqlite3.connect('prc_automation/prc_database/prc_r0_results.db')

cursor_raw = conn_raw.cursor()
cursor_results = conn_results.cursor()

# =============================================================================
# 1. EXEC_IDS GAM-001 DISPONIBLES
# =============================================================================
print("\n1. EXEC_IDS GAM-001 DANS DB_RAW")
print("-" * 70)

cursor_raw.execute("""
    SELECT id, d_base_id, modifier_id, seed
    FROM Executions
    WHERE gamma_id = 'GAM-001'
    ORDER BY id
""")

gam001_raw = cursor_raw.fetchall()
print(f"Total GAM-001 dans db_raw : {len(gam001_raw)} runs")
print(f"Plage exec_id : {gam001_raw[0][0]} → {gam001_raw[-1][0]}")

# =============================================================================
# 2. EXEC_IDS TRAITÉS
# =============================================================================
print("\n2. EXEC_IDS TRAITÉS DANS DB_RESULTS")
print("-" * 70)

cursor_results.execute("""
    SELECT DISTINCT exec_id
    FROM TestObservations
    ORDER BY exec_id
""")

treated_ids = [row[0] for row in cursor_results.fetchall()]
print(f"Total exec_id traités : {len(treated_ids)}")
print(f"Plage : {min(treated_ids)} → {max(treated_ids)}")

# =============================================================================
# 3. GAM-001 TRAITÉS vs IGNORÉS
# =============================================================================
print("\n3. GAM-001 : TRAITÉS vs IGNORÉS")
print("-" * 70)

gam001_ids = [row[0] for row in gam001_raw]
treated_gam001 = [id for id in treated_ids if id in gam001_ids]
ignored_gam001 = [id for id in gam001_ids if id not in treated_ids]

print(f"GAM-001 traités : {len(treated_gam001)}/{len(gam001_ids)}")
print(f"GAM-001 ignorés : {len(ignored_gam001)}/{len(gam001_ids)}")

# =============================================================================
# 4. DISTRIBUTION TRAITÉS PAR D_BASE_ID
# =============================================================================
print("\n4. DISTRIBUTION PAR D_BASE_ID")
print("-" * 70)

cursor_raw.execute("""
    SELECT d_base_id, COUNT(*) as total
    FROM Executions
    WHERE gamma_id = 'GAM-001'
    GROUP BY d_base_id
""")

print("Dans db_raw (GAM-001) :")
for row in cursor_raw.fetchall():
    print(f"  {row[0]:10} : {row[1]:3} runs disponibles")

# Traités
cursor_results.execute("""
    SELECT e.d_base_id, COUNT(DISTINCT o.exec_id) as nb_treated
    FROM TestObservations o
    JOIN Executions e ON o.exec_id = e.id
    WHERE e.gamma_id = 'GAM-001'
    GROUP BY e.d_base_id
""")

print("\nDans db_results (GAM-001 traités) :")
results_by_d = {}
for row in cursor_results.fetchall():
    results_by_d[row[0]] = row[1]
    print(f"  {row[0]:10} : {row[1]:3} runs traités")

# =============================================================================
# 5. PATTERNS DES IGNORÉS
# =============================================================================
print("\n5. ANALYSE DES IGNORÉS")
print("-" * 70)

if ignored_gam001:
    # Prendre premiers et derniers ignorés
    sample_ignored = ignored_gam001[:5] + ignored_gam001[-5:]
    
    print(f"Échantillon des {len(ignored_gam001)} ignorés :")
    for exec_id in sample_ignored:
        cursor_raw.execute("""
            SELECT id, d_base_id, modifier_id, seed
            FROM Executions
            WHERE id = ?
        """, (exec_id,))
        row = cursor_raw.fetchone()
        print(f"  exec_id={row[0]:4} | d_base={row[1]:10} | mod={row[2]:3} | seed={row[3]}")
    
    # Pattern
    cursor_raw.execute("""
        SELECT d_base_id, COUNT(*) as nb
        FROM Executions
        WHERE id IN ({})
        GROUP BY d_base_id
    """.format(','.join('?' * len(ignored_gam001))), ignored_gam001)
    
    print("\nIgnorés par d_base_id :")
    for row in cursor_raw.fetchall():
        print(f"  {row[0]:10} : {row[1]} ignorés")

# =============================================================================
# 6. PREMIER vs DERNIER TRAITÉ
# =============================================================================
print("\n6. PREMIER vs DERNIER RUN TRAITÉ")
print("-" * 70)

if treated_gam001:
    first_id = min(treated_gam001)
    last_id = max(treated_gam001)
    
    cursor_raw.execute("""
        SELECT id, d_base_id, modifier_id, seed, run_id
        FROM Executions
        WHERE id IN (?, ?)
        ORDER BY id
    """, (first_id, last_id))
    
    for row in cursor_raw.fetchall():
        print(f"exec_id={row[0]:4} | d_base={row[1]:10} | mod={row[2]:3} | seed={row[3]} | run_id={row[4]}")

# =============================================================================
# 7. HYPOTHÈSE : Y A-T-IL UN SAUT ?
# =============================================================================
print("\n7. CONTINUITÉ DES EXEC_IDS TRAITÉS")
print("-" * 70)

treated_sorted = sorted(treated_gam001)
gaps = []

for i in range(len(treated_sorted) - 1):
    gap = treated_sorted[i+1] - treated_sorted[i]
    if gap > 1:
        gaps.append((treated_sorted[i], treated_sorted[i+1], gap))

if gaps:
    print(f"⚠️ {len(gaps)} sauts détectés dans la séquence :")
    for start, end, gap in gaps[:5]:
        print(f"  Saut : exec_id {start} → {end} (gap={gap})")
else:
    print("✓ Séquence continue (pas de sauts)")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

print(f"\nGAM-001 :")
print(f"  Disponibles : {len(gam001_ids)} runs")
print(f"  Traités     : {len(treated_gam001)} runs ({len(treated_gam001)/len(gam001_ids)*100:.1f}%)")
print(f"  Ignorés     : {len(ignored_gam001)} runs")

if len(treated_gam001) == 200:
    print(f"\n🔍 HYPOTHÈSE : Limite fixe à 200 runs ?")
    print(f"   → Vérifier batch_runner.py pour hardcoded limit")

if results_by_d:
    single_d = len(results_by_d) == 1
    if single_d:
        print(f"\n🔍 HYPOTHÈSE : Un seul d_base_id traité ?")
        print(f"   → Peut-être filtrage sur d_base_id spécifique")

conn_raw.close()
conn_results.close()

print("\n" + "=" * 70)