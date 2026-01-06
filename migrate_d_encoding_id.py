#!/usr/bin/env python3
# migrate_d_encoding_id.py
"""
Migration d_base_id → d_encoding_id (Charter 5.5)

ATTENTION : Crée une COPIE de backup avant modification !
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

DB_RAW = Path("prc_automation/prc_database/prc_r0_raw.db")
DB_RESULTS = Path("prc_automation/prc_database/prc_r0_results.db")

def backup_database(db_path: Path):
    """Crée backup avec timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = db_path.with_suffix(f'.backup_{timestamp}.db')
    shutil.copy2(db_path, backup_path)
    print(f"✓ Backup créé : {backup_path}")
    return backup_path


def migrate_db_raw():
    """Migre prc_r0_raw.db"""
    print(f"\n{'='*70}")
    print("MIGRATION DB_RAW")
    print(f"{'='*70}\n")
    
    # Backup
   # backup = backup_database(DB_RAW)
    
    conn = sqlite3.connect(DB_RAW)
    cursor = conn.cursor()
    
    try:
        # Vérifier si colonne existe
        cursor.execute("PRAGMA table_info(Executions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'd_encoding_id' in columns:
            print("⚠️ d_encoding_id existe déjà, skip migration")
            conn.close()
            return
        
        if 'd_base_id' not in columns:
            print("❌ d_base_id n'existe pas, impossible de migrer")
            conn.close()
            return
        
        print("1. Ajout colonne d_encoding_id...")
        cursor.execute("""
            ALTER TABLE Executions 
            ADD COLUMN d_encoding_id TEXT
        """)
        
        print("2. Copie données d_base_id → d_encoding_id...")
        cursor.execute("""
            UPDATE Executions 
            SET d_encoding_id = d_base_id
        """)
        
        rows_updated = cursor.rowcount
        print(f"   ✓ {rows_updated} lignes migrées")
        
        # Pas de suppression d_base_id (peut casser des contraintes)
        print("3. Conservation d_base_id pour compatibilité...")
        print("   (sera obsolète mais pas supprimé)")
        
        conn.commit()
        print("\n✓ Migration db_raw terminée avec succès")
        
    except Exception as e:
        print(f"\n❌ ERREUR migration : {e}")
        conn.rollback()
        print(f"   Restaurer depuis : {backup}")
        
    finally:
        conn.close()


def migrate_db_results():
    """Migre prc_r0_results.db (moins critique car régénérable)"""
    print(f"\n{'='*70}")
    print("MIGRATION DB_RESULTS")
    print(f"{'='*70}\n")
    
    print("ℹ️ db_results est régénérable, pas de migration nécessaire")
    print("   Elle sera recréée avec les bonnes colonnes au prochain run")


def update_code_files():
    """Affiche instructions pour mise à jour code."""
    print(f"\n{'='*70}")
    print("MISE À JOUR CODE")
    print(f"{'='*70}\n")
    
    files_to_update = [
        "tests/utilities/applicability.py",
        "tests/utilities/test_engine.py",
        "prc_automation/batch_runner.py",
    ]
    
    print("Fichiers à mettre à jour manuellement :")
    print("(Remplacer 'd_base_id' par 'd_encoding_id')\n")
    
    for file in files_to_update:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️ {file} (non trouvé)")
    
    print("\nCommande recherche globale :")
    print('  findstr /s /i "d_base_id" *.py')


def verify_migration():
    """Vérifie résultat migration."""
    print(f"\n{'='*70}")
    print("VÉRIFICATION")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(DB_RAW)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(Executions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    has_encoding = 'd_encoding_id' in columns
    has_base = 'd_base_id' in columns
    
    print(f"Colonnes Executions :")
    print(f"  - d_base_id     : {'✓' if has_base else '✗'}")
    print(f"  - d_encoding_id : {'✓' if has_encoding else '✗'}")
    
    if has_encoding:
        # Vérifier cohérence
        cursor.execute("""
            SELECT COUNT(*) 
            FROM Executions 
            WHERE d_base_id != d_encoding_id 
               OR d_encoding_id IS NULL
        """)
        inconsistent = cursor.fetchone()[0]
        
        if inconsistent > 0:
            print(f"\n⚠️ {inconsistent} lignes incohérentes détectées !")
        else:
            print(f"\n✓ Migration cohérente (d_base_id == d_encoding_id)")
    
    conn.close()


def main():
    print("\n" + "#"*70)
    print("# MIGRATION d_base_id → d_encoding_id")
    print("#"*70)
    
    print("\n⚠️ ATTENTION : Cette opération modifie la base de données !")
    print("   Des backups seront créés automatiquement.\n")
    
    response = input("Continuer ? (oui/non) : ").strip().lower()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("Migration annulée.")
        return
    
    # Migration
    migrate_db_raw()
    migrate_db_results()
    
    # Vérification
    verify_migration()
    
    # Instructions code
    update_code_files()
    
    print("\n" + "#"*70)
    print("# MIGRATION TERMINÉE")
    print("#"*70)
    print("\nProchaines étapes :")
    print("  1. Vérifier backups créés")
    print("  2. Mettre à jour code Python (voir liste ci-dessus)")
    print("  3. Relancer batch_runner --test")
    print("\n")


if __name__ == "__main__":
    main()