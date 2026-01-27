#!/usr/bin/env python3
"""
Utilitaires gestion bases de données PRC.

Modes:
- init: Création schéma vierge
- clean: Nettoyage sélectif (deprecated, rejected)
- backup: Sauvegarde timestampée
- export_schema: Export structure sans données
- export_json: Export complet en JSON

Usage:
    python -m prc_automation.utils_databases --mode init --phase R0
    python -m prc_automation.utils_databases --mode clean --phase R0
    python -m prc_automation.utils_databases --mode backup --phase R0
    python -m prc_automation.utils_databases --mode export_schema --phase R0
    python -m prc_automation.utils_databases --mode export_json --phase R0 [--include-blobs]
"""

import argparse
import sqlite3
import json
import base64
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_DIR = Path("./prc_automation/prc_database")
SCHEMA_RAW = DB_DIR / "schema_raw.sql"
SCHEMA_RESULTS = DB_DIR / "schema_results.sql"


def get_db_paths(phase: str = 'R0') -> Dict[str, Path]:
    """Retourne chemins bases pour phase donnée."""
    return {
        'raw': DB_DIR / f"prc_{phase.lower()}_raw.db",
        'results': DB_DIR / f"prc_{phase.lower()}_results.db"
    }


# =============================================================================
# MODE INIT
# =============================================================================

def init_databases(phase: str = 'R0') -> None:
    """
    Crée schéma bases vierges.
    
    Args:
        phase: Phase cible ('R0', 'R1', etc.)
    
    Raises:
        FileExistsError: Si bases existent déjà
    """
    print(f"\n{'='*70}")
    print(f"INITIALISATION BASES {phase}")
    print(f"{'='*70}\n")
    
    paths = get_db_paths(phase)
    
    # Vérifier existence
    if paths['raw'].exists() or paths['results'].exists():
        raise FileExistsError(
            f"Bases {phase} existent déjà. Utiliser --mode backup puis supprimer."
        )
    
    # Créer répertoire
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vérifier schemas
    if not SCHEMA_RAW.exists():
        raise FileNotFoundError(f"Schema manquant: {SCHEMA_RAW}")
    if not SCHEMA_RESULTS.exists():
        raise FileNotFoundError(f"Schema manquant: {SCHEMA_RESULTS}")
    
    # Créer db_raw
    print(f"Création {paths['raw']}...")
    with open(SCHEMA_RAW, 'r') as f:
        schema_raw = f.read()
    
    conn = sqlite3.connect(paths['raw'])
    conn.executescript(schema_raw)
    conn.commit()
    conn.close()
    print(f"  ✓ Tables: executions, snapshots, metrics")
    
    # Créer db_results
    print(f"\nCréation {paths['results']}...")
    with open(SCHEMA_RESULTS, 'r') as f:
        schema_results = f.read()
    
    conn = sqlite3.connect(paths['results'])
    conn.executescript(schema_results)
    conn.commit()
    conn.close()
    print(f"  ✓ Tables: observations")
    
    print(f"\n{'='*70}")
    print(f"INITIALISATION {phase} TERMINÉE")
    print(f"{'='*70}\n")


# =============================================================================
# MODE CLEAN
# =============================================================================

def clean_databases(phase: str = 'R0') -> Dict[str, Any]:
    """
    Supprime entrées obsolètes.
    
    Args:
        phase: Phase cible
    
    Returns:
        Rapport traçabilité:
        {
            'timestamp': str,
            'deleted_count': int,
            'deleted_entries': List[dict],
            'disk_freed_mb': float
        }
    """
    print(f"\n{'='*70}")
    print(f"NETTOYAGE BASES {phase}")
    print(f"{'='*70}\n")
    
    paths = get_db_paths(phase)
    
    if not paths['raw'].exists():
        raise FileNotFoundError(f"Base manquante: {paths['raw']}")
    
    # Taille initiale
    size_before = paths['raw'].stat().st_size + paths['results'].stat().st_size
    
    deleted_entries = []
    
    # Clean db_raw
    conn_raw = sqlite3.connect(paths['raw'])
    cursor_raw = conn_raw.cursor()
    
    # Deprecated (gamma_id/d_encoding_id/modifier_id contenant '_deprecated_')
    cursor_raw.execute("""
        SELECT exec_id, gamma_id, d_encoding_id, modifier_id
        FROM executions
        WHERE gamma_id LIKE '%_deprecated_%'
           OR d_encoding_id LIKE '%_deprecated_%'
           OR modifier_id LIKE '%_deprecated_%'
    """)
    
    for row in cursor_raw.fetchall():
        deleted_entries.append({
            'exec_id': row[0],
            'gamma_id': row[1],
            'd_encoding_id': row[2],
            'modifier_id': row[3],
            'reason': 'deprecated'
        })
    
    cursor_raw.execute("""
        DELETE FROM executions
        WHERE gamma_id LIKE '%_deprecated_%'
           OR d_encoding_id LIKE '%_deprecated_%'
           OR modifier_id LIKE '%_deprecated_%'
    """)
    
    # Rejected (nécessite lecture catalogues - TODO)
    # Pour l'instant skip (implémentation manuelle si nécessaire)
    
    conn_raw.commit()
    conn_raw.close()
    
    # Clean db_results (correspondance)
    conn_results = sqlite3.connect(paths['results'])
    cursor_results = conn_results.cursor()
    
    cursor_results.execute("""
        DELETE FROM observations
        WHERE gamma_id LIKE '%_deprecated_%'
           OR d_encoding_id LIKE '%_deprecated_%'
           OR modifier_id LIKE '%_deprecated_%'
    """)
    
    conn_results.commit()
    conn_results.close()
    
    # Taille après
    size_after = paths['raw'].stat().st_size + paths['results'].stat().st_size
    disk_freed_mb = (size_before - size_after) / (1024 * 1024)
    
    # Rapport
    report = {
        'timestamp': datetime.now().isoformat(),
        'phase': phase,
        'deleted_count': len(deleted_entries),
        'deleted_entries': deleted_entries,
        'disk_freed_mb': round(disk_freed_mb, 2)
    }
    
    # Sauvegarder rapport
    report_path = DB_DIR / f"cleaned_entries_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Entrées supprimées: {len(deleted_entries)}")
    print(f"Espace libéré:      {disk_freed_mb:.2f} MB")
    print(f"Rapport:            {report_path}")
    print(f"\n{'='*70}\n")
    
    return report


# =============================================================================
# MODE BACKUP
# =============================================================================

def backup_databases(phase: str = 'R0') -> str:
    """
    Sauvegarde timestampée bases.
    
    Args:
        phase: Phase cible
    
    Returns:
        Chemin archive créée
    """
    print(f"\n{'='*70}")
    print(f"BACKUP BASES {phase}")
    print(f"{'='*70}\n")
    
    paths = get_db_paths(phase)
    
    if not paths['raw'].exists():
        raise FileNotFoundError(f"Base manquante: {paths['raw']}")
    
    # Nom archive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"backup_{phase}_{timestamp}.tar.gz"
    archive_path = DB_DIR / archive_name
    
    # Créer archive
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(paths['raw'], arcname=paths['raw'].name)
        tar.add(paths['results'], arcname=paths['results'].name)
    
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    
    print(f"Archive créée: {archive_path}")
    print(f"Taille:        {size_mb:.2f} MB")
    print(f"\n{'='*70}\n")
    
    return str(archive_path)


# =============================================================================
# MODE EXPORT JSON
# =============================================================================

def export_database_to_json(db_path: str, output_path: str, include_blobs: bool = False):
    """
    Exporte une base SQLite en JSON.
    
    Args:
        db_path: Chemin vers le fichier .db
        output_path: Chemin fichier JSON de sortie
        include_blobs: Si True, encode les BLOB en base64 (augmente taille)
    """
    print(f"\n{'='*70}")
    print(f"EXPORT JSON: {db_path}")
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
        'export_timestamp': datetime.now().isoformat(),
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
            print(f"  - {blobs_skipped} BLOBs non exportés (utilisez --include-blobs)")
        
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
    
    Args:
        db_path: Chemin vers le fichier .db
        output_path: Chemin fichier JSON de sortie
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
        'export_timestamp': datetime.now().isoformat(),
        'schema': schema_statements,
        'table_stats': stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Schema exporté vers: {output_path}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Utilitaires gestion bases de données PRC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Initialiser bases R0
  python -m prc_automation.utils_databases --mode init --phase R0
  
  # Nettoyer entrées obsolètes
  python -m prc_automation.utils_databases --mode clean --phase R0
  
  # Backup avant modifications
  python -m prc_automation.utils_databases --mode backup --phase R0
  
  # Export schema uniquement (léger)
  python -m prc_automation.utils_databases --mode export_schema --phase R0
  
  # Export complet JSON (lourd)
  python -m prc_automation.utils_databases --mode export_json --phase R0
  
  # Export JSON avec BLOBs encodés
  python -m prc_automation.utils_databases --mode export_json --phase R0 --include-blobs
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['init', 'clean', 'backup', 'export_schema', 'export_json'],
                       help="Mode opération")
    
    parser.add_argument('--phase', default='R0',
                       help="Phase cible (défaut: R0)")
    
    parser.add_argument('--include-blobs', action='store_true',
                       help='Inclure BLOBs encodés en base64 (mode export_json uniquement)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'init':
            init_databases(args.phase)
        
        elif args.mode == 'clean':
            clean_databases(args.phase)
        
        elif args.mode == 'backup':
            backup_databases(args.phase)
        
        elif args.mode == 'export_schema':
            # Export schema des deux bases
            paths = get_db_paths(args.phase)
            
            if not paths['raw'].exists():
                raise FileNotFoundError(f"Base manquante: {paths['raw']}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            export_database_schema_only(
                str(paths['raw']),
                str(DB_DIR / f"prc_{args.phase.lower()}_raw_schema_{timestamp}.json")
            )
            
            export_database_schema_only(
                str(paths['results']),
                str(DB_DIR / f"prc_{args.phase.lower()}_results_schema_{timestamp}.json")
            )
        
        elif args.mode == 'export_json':
            # Export complet des deux bases
            paths = get_db_paths(args.phase)
            
            if not paths['raw'].exists():
                raise FileNotFoundError(f"Base manquante: {paths['raw']}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            export_database_to_json(
                str(paths['raw']),
                str(DB_DIR / f"prc_{args.phase.lower()}_raw_full_{timestamp}.json"),
                args.include_blobs
            )
            
            export_database_to_json(
                str(paths['results']),
                str(DB_DIR / f"prc_{args.phase.lower()}_results_full_{timestamp}.json"),
                args.include_blobs
            )
    
    except Exception as e:
        print(f"\n✗ Erreur: {e}\n")
        raise


if __name__ == "__main__":
    main()