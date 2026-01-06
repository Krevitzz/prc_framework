# check_d_base.py
import sqlite3

conn = sqlite3.connect('prc_automation/prc_database/prc_r0_results.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT DISTINCT o.exec_id, e.d_base_id
    FROM TestObservations o
    JOIN (SELECT id, d_base_id FROM Executions) e 
        ON o.exec_id = e.id
    ORDER BY o.exec_id
    LIMIT 20
""")

print("Premiers 20 exec_id traités :")
for row in cursor.fetchall():
    print(f"  exec_id={row[0]:4} → d_encoding_id={row[1]}")

# Distribution complète
cursor.execute("""
    SELECT e.d_base_id, COUNT(DISTINCT o.exec_id) as nb_runs
    FROM TestObservations o
    JOIN (SELECT id, d_base_id FROM Executions) e 
        ON o.exec_id = e.id
    GROUP BY e.d_base_id
""")

print("\nDistribution par d_encoding_id :")
for row in cursor.fetchall():
    print(f"  {row[0]:10} : {row[1]} runs traités")

conn.close()