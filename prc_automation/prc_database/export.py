import sqlite3
import json
import base64
from pathlib import Path

def export_database_to_json(db_path: str, output_path: str, include_blobs: bool = False):
    """
    Exporte une base SQLite en JSON.
    
    Args:
        db_path: Chemin vers le fichier .db
        output_path: Chemin fichier JSON de sortie
        include_blobs: Si True, encode les BLOB en base64 (augmente taille)
    """
    print(f"\n{'='*70}")
    print(f"EXPORT: {db_path}")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Permet d'accéder aux colonnes par nom
    cursor = conn.cursor()
    
    # Lister toutes les tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Tables trouvées: {', '.join(tables)}\n")
    
    data = {
        'database': db_path,
        'tables': {}
    }
    
    for table_name in tables:
        print(f"Exportation table: {table_name}")
        
        # Récupérer infos colonnes
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        column_types = {col[1]: col[2] for col in columns_info}
        
        # Récupérer données
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        print(f"  - {len(rows)} lignes")
        print(f"  - Colonnes: {', '.join(column_names)}")
        
        # Convertir en format sérialisable
        table_data = []
        blobs_skipped = 0
        
        for row in rows:
            row_dict = {}
            for col_name in column_names:
                value = row[col_name]
                
                # Gérer les bytes
                if isinstance(value, bytes):
                    if include_blobs:
                        # Encoder en base64
                        row_dict[col_name] = {
                            '_type': 'blob',
                            '_size_bytes': len(value),
                            'data': base64.b64encode(value).decode('utf-8')
                        }
                    else:
                        # Juste indiquer la présence
                        row_dict[col_name] = {
                            '_type': 'blob',
                            '_size_bytes': len(value),
                            'data': '<BLOB_SKIPPED>'
                        }
                        blobs_skipped += 1
                else:
                    row_dict[col_name] = value
            
            table_data.append(row_dict)
        
        if blobs_skipped > 0:
            print(f"  - {blobs_skipped} BLOBs non exportés (utilisez include_blobs=True)")
        
        data['tables'][table_name] = {
            'columns': column_names,
            'column_types': column_types,
            'row_count': len(rows),
            'rows': table_data
        }
        print()
    
    conn.close()
    
    # Sauvegarder JSON
    print(f"Écriture vers: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
    print(f"✓ Export terminé ({file_size:.2f} MB)\n")


def export_database_schema_only(db_path: str, output_path: str):
    """
    Exporte uniquement le schéma (structure) sans les données.
    """
    print(f"\n{'='*70}")
    print(f"EXPORT SCHEMA: {db_path}")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Récupérer tout le schéma
    cursor.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
    schema_statements = [row[0] for row in cursor.fetchall()]
    
    # Stats tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    stats = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        stats[table] = count
        print(f"  - {table}: {count} lignes")
    
    conn.close()
    
    data = {
        'database': db_path,
        'schema': schema_statements,
        'table_stats': stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Schema exporté vers: {output_path}\n")


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export SQLite databases")
    parser.add_argument('--mode', choices=['full', 'schema'], default='schema',
                       help='Mode export: full (avec données) ou schema (structure seule)')
    parser.add_argument('--include-blobs', action='store_true',
                       help='Inclure BLOBs encodés en base64 (augmente taille)')
    
    args = parser.parse_args()
    
    # Chemins
    db_raw = "./prc_r0_raw.db"
    db_results = "./prc_r0_results.db"
    
    if args.mode == 'schema':
        # Export schema uniquement (léger)
        export_database_schema_only(db_raw, "prc_r0_raw_schema.json")
        export_database_schema_only(db_results, "prc_r0_results_schema.json")
        
        print("="*70)
        print("EXPORT TERMINÉ - Uploadez ces fichiers JSON:")
        print("  - prc_r0_raw_schema.json")
        print("  - prc_r0_results_schema.json")
        print("="*70)
    
    else:
        # Export complet (peut être lourd)
      #  export_database_to_json(db_raw, "prc_r0_raw_full.json", args.include_blobs)
        export_database_to_json(db_results, "prc_r0_results_full.json", args.include_blobs)
        
        print("="*70)
        print("EXPORT TERMINÉ - Uploadez ces fichiers JSON:")
        print("  - prc_r0_raw_full.json")
        print("  - prc_r0_results_full.json")
        print("="*70)