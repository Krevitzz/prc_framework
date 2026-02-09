#!/usr/bin/env python3
"""
Utilitaires gestion bases de données PRC (V2 - Sans db_raw).

CHANGEMENTS V2:
- Suppression db_raw (pipeline unifié batch_runner_v3)
- R0 : Une seule DB (prc_r0_results.db)
- R1 : Deux DBs (prc_r1_results.db + prc_r1_sequences.db)

Modes:
- init: Création schéma vierge
- clean: Nettoyage sélectif (deprecated)
- backup: Sauvegarde timestampée
- export_schema: Export structure sans données
- export_json: Export complet en JSON

Usage:
    # R0 (1 DB)
    python -m prc_automation.utils_databases --mode init --phase R0
    
    # R1 (2 DBs)
    python -m prc_automation.utils_databases --mode init --phase R1
    
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
from typing import Dict, List


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_DIR = Path("./prc_automation/prc_database")

# Schemas par phase
SCHEMAS = {
    'R0': {
        'results': DB_DIR / "schema_results.sql"
    },
    'R1': {
        'results': DB_DIR / "schema_results_r1.sql",
        'sequences': DB_DIR / "schema_sequences_r1.sql"
    }
}


def get_db_paths(phase: str) -> Dict[str, Path]:
    """
    Retourne chemins DBs pour phase donnée.
    
    Args:
        phase: 'R0' ou 'R1'
    
    Returns:
        R0: {'results': prc_r0_results.db}
        R1: {'results': prc_r1_results.db, 'sequences': prc_r1_sequences.db}
    """
    phase_lower = phase.lower()
    
    if phase == 'R0':
        return {
            'results': DB_DIR / f"prc_{phase_lower}_results.db"
        }
    elif phase == 'R1':
        return {
            'results': DB_DIR / f"prc_{phase_lower}_results.db",
            'sequences': DB_DIR / f"prc_{phase_lower}_sequences.db"
        }
    else:
        raise ValueError(f"Phase inconnue: {phase} (attendu 'R0' ou 'R1')")


# =============================================================================
# MODE INIT
# =============================================================================

def init_databases(phase: str) -> None:
    """
    Crée schéma bases vierges.
    
    R0: Crée prc_r0_results.db
    R1: Crée prc_r1_results.db + prc_r1_sequences.db
    
    Args:
        phase: Phase cible ('R0', 'R1')
    
    Raises:
        FileExistsError: Si bases existent déjà
        FileNotFoundError: Si schemas manquants
    """
    print(f"\n{'='*70}")
    print(f"INITIALISATION BASES {phase}")
    print(f"{'='*70}\n")
    
    # Vérifier phase supportée
    if phase not in SCHEMAS:
        raise ValueError(f"Phase non supportée: {phase} (disponibles: {list(SCHEMAS.keys())})")
    
    paths = get_db_paths(phase)
    schemas = SCHEMAS[phase]
    
    # Vérifier existence
    existing = [name for name, path in paths.items() if path.exists()]
    if existing:
        raise FileExistsError(
            f"Bases {phase} existent déjà: {', '.join(existing)}\n"
            f"→ Utiliser --mode backup puis supprimer manuellement."
        )
    
    # Créer répertoire
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vérifier schemas disponibles
    missing_schemas = [name for name, path in schemas.items() if not path.exists()]
    if missing_schemas:
        raise FileNotFoundError(
            f"Schemas manquants pour {phase}: {', '.join(missing_schemas)}\n"
            f"Attendus:\n" + '\n'.join(f"  - {schemas[name]}" for name in missing_schemas)
        )
    
    # Créer bases
    for db_name, db_path in paths.items():
        schema_path = schemas[db_name]
        
        print(f"Création {db_path}...")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.commit()
        
        # Lister tables créées
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"  ✓ Tables: {', '.join(tables)}")
        print()
    
    print(f"{'='*70}")
    print(f"INITIALISATION {phase} TERMINÉE")
    print(f"{'='*70}\n")


# =============================================================================
# MODE CLEAN
# =============================================================================

def clean_databases(phase: str) -> Dict[str, any]:
    """
    Supprime entrées obsolètes (deprecated).
    
    Args:
        phase: Phase cible
    
    Returns:
        Rapport traçabilité:
        {
            'timestamp': str,
            'phase': str,
            'deleted_count': int,
            'deleted_entries': List[dict],
            'disk_freed_mb': float
        }
    """
    print(f"\n{'='*70}")
    print(f"NETTOYAGE BASES {phase}")
    print(f"{'='*70}\n")
    
    paths = get_db_paths(phase)
    
    # Vérifier existence
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Bases manquantes pour {phase}: {', '.join(missing)}"
        )
    
    # Taille initiale
    size_before = sum(path.stat().st_size for path in paths.values())
    
    deleted_entries = []
    
    # Clean db_results (observations deprecated)
    results_path = paths['results']
    conn = sqlite3.connect(results_path)
    cursor = conn.cursor()
    
    # Lister deprecated avant suppression
    cursor.execute("""
        SELECT test_name, gamma_id, d_encoding_id, modifier_id, seed
        FROM observations
        WHERE gamma_id LIKE '%_deprecated_%'
           OR d_encoding_id LIKE '%_deprecated_%'
           OR modifier_id LIKE '%_deprecated_%'
    """)
    
    for row in cursor.fetchall():
        deleted_entries.append({
            'test_name': row[0],
            'gamma_id': row[1],
            'd_encoding_id': row[2],
            'modifier_id': row[3],
            'seed': row[4],
            'reason': 'deprecated',
            'db': 'results'
        })
    
    # Supprimer
    cursor.execute("""
        DELETE FROM observations
        WHERE gamma_id LIKE '%_deprecated_%'
           OR d_encoding_id LIKE '%_deprecated_%'
           OR modifier_id LIKE '%_deprecated_%'
    """)
    
    conn.commit()
    conn.close()
    
    # Clean db_sequences si R1
    if 'sequences' in paths:
        sequences_path = paths['sequences']
        conn_seq = sqlite3.connect(sequences_path)
        cursor_seq = conn_seq.cursor()
        
        # Lister deprecated sequences
        cursor_seq.execute("""
            SELECT sequence_exec_id, sequence_gammas, d_encoding_id, modifier_id
            FROM sequences
            WHERE d_encoding_id LIKE '%_deprecated_%'
               OR modifier_id LIKE '%_deprecated_%'
        """)
        
        for row in cursor_seq.fetchall():
            deleted_entries.append({
                'sequence_exec_id': row[0],
                'sequence_gammas': row[1],
                'd_encoding_id': row[2],
                'modifier_id': row[3],
                'reason': 'deprecated',
                'db': 'sequences'
            })
        
        # Supprimer
        cursor_seq.execute("""
            DELETE FROM sequences
            WHERE d_encoding_id LIKE '%_deprecated_%'
               OR modifier_id LIKE '%_deprecated_%'
        """)
        
        conn_seq.commit()
        conn_seq.close()
    
    # Taille après
    size_after = sum(path.stat().st_size for path in paths.values())
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
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Entrées supprimées: {len(deleted_entries)}")
    print(f"  - db_results:     {sum(1 for e in deleted_entries if e['db'] == 'results')}")
    if 'sequences' in paths:
        print(f"  - db_sequences:   {sum(1 for e in deleted_entries if e['db'] == 'sequences')}")
    print(f"Espace libéré:      {disk_freed_mb:.2f} MB")
    print(f"Rapport:            {report_path}")
    print(f"\n{'='*70}\n")
    
    return report


# =============================================================================
# MODE BACKUP
# =============================================================================

def backup_databases(phase: str) -> str:
    """
    Sauvegarde timestampée bases.
    
    R0: Archive prc_r0_results.db
    R1: Archive prc_r1_results.db + prc_r1_sequences.db
    
    Args:
        phase: Phase cible
    
    Returns:
        Chemin archive créée
    """
    print(f"\n{'='*70}")
    print(f"BACKUP BASES {phase}")
    print(f"{'='*70}\n")
    
    paths = get_db_paths(phase)
    
    # Vérifier existence
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Bases manquantes pour {phase}: {', '.join(missing)}"
        )
    
    # Nom archive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"backup_{phase}_{timestamp}.tar.gz"
    archive_path = DB_DIR / archive_name
    
    # Créer archive
    with tarfile.open(archive_path, 'w:gz') as tar:
        for db_name, db_path in paths.items():
            print(f"Ajout {db_path.name}...")
            tar.add(db_path, arcname=db_path.name)
    
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Backup créé: {archive_path}")
    print(f"  Taille: {archive_size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    return str(archive_path)


# =============================================================================
# MODE EXPORT JSON
# =============================================================================

def export_database_to_json(db_path: Path, output_path: Path, include_blobs: bool = False):
    """
    Exporte une base SQLite en JSON.
    
    Args:
        db_path: Chemin vers le fichier .db
        output_path: Chemin fichier JSON de sortie
        include_blobs: Si True, encode les BLOB en base64 (augmente taille)
    """
    print(f"\n{'='*70}")
    print(f"EXPORT JSON: {db_path.name}")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Lister tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Tables: {', '.join(tables)}\n")
    
    data = {
        'database': str(db_path),
        'export_timestamp': datetime.now().isoformat(),
        'tables': {}
    }
    
    for table_name in tables:
        print(f"Export {table_name}...")
        
        # Colonnes
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        column_types = {col[1]: col[2] for col in columns_info}
        
        # Données
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        print(f"  - {len(rows)} lignes")
        print(f"  - {len(column_names)} colonnes")
        
        # Convertir
        table_data = []
        blobs_skipped = 0
        
        for row in rows:
            row_dict = {}
            for col_name in column_names:
                value = row[col_name]
                
                if isinstance(value, bytes):
                    if include_blobs:
                        row_dict[col_name] = {
                            '_type': 'blob',
                            '_size_bytes': len(value),
                            'data': base64.b64encode(value).decode('utf-8')
                        }
                    else:
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
            print(f"  - {blobs_skipped} BLOBs skippés (--include-blobs pour exporter)")
        
        data['tables'][table_name] = {
            'columns': column_names,
            'column_types': column_types,
            'row_count': len(rows),
            'rows': table_data
        }
        print()
    
    conn.close()
    
    # Sauvegarder
    print(f"Écriture {output_path.name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Export terminé ({file_size_mb:.2f} MB)\n")


def export_database_schema_only(db_path: Path, output_path: Path):
    """
    Exporte uniquement le schéma (structure) sans les données.
    
    Args:
        db_path: Chemin vers le fichier .db
        output_path: Chemin fichier JSON de sortie
    """
    print(f"\n{'='*70}")
    print(f"EXPORT SCHEMA: {db_path.name}")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Schema complet
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
        'database': str(db_path),
        'export_timestamp': datetime.now().isoformat(),
        'schema': schema_statements,
        'table_stats': stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Schema exporté: {output_path.name}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Utilitaires gestion bases de données PRC (V2 - Sans db_raw)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Initialiser R0 (1 DB)
  python -m prc_automation.utils_databases --mode init --phase R0
  
  # Initialiser R1 (2 DBs)
  python -m prc_automation.utils_databases --mode init --phase R1
  
  # Nettoyer entrées obsolètes
  python -m prc_automation.utils_databases --mode clean --phase R0
  
  # Backup avant modifications
  python -m prc_automation.utils_databases --mode backup --phase R0
  
  # Export schema uniquement (léger)
  python -m prc_automation.utils_databases --mode export_schema --phase R0
  
  # Export complet JSON (lourd)
  python -m prc_automation.utils_databases --mode export_json --phase R0
  
  # Export JSON avec BLOBs encodés
  python -m prc_automation.utils_databases --mode export_json --phase R1 --include-blobs
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['init', 'clean', 'backup', 'export_schema', 'export_json'],
                       help="Mode opération")
    
    parser.add_argument('--phase', default='R0',
                       help="Phase cible (défaut: R0)")
    
    parser.add_argument('--include-blobs', action='store_true',
                       help='Inclure BLOBs encodés base64 (mode export_json uniquement)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'init':
            init_databases(args.phase)
        
        elif args.mode == 'clean':
            clean_databases(args.phase)
        
        elif args.mode == 'backup':
            backup_databases(args.phase)
        
        elif args.mode == 'export_schema':
            paths = get_db_paths(args.phase)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for db_name, db_path in paths.items():
                if not db_path.exists():
                    print(f"⚠️  {db_path.name} n'existe pas (skip)")
                    continue
                
                output_name = f"{db_path.stem}_schema_{timestamp}.json"
                export_database_schema_only(
                    db_path,
                    DB_DIR / output_name
                )
        
        elif args.mode == 'export_json':
            paths = get_db_paths(args.phase)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for db_name, db_path in paths.items():
                if not db_path.exists():
                    print(f"⚠️  {db_path.name} n'existe pas (skip)")
                    continue
                
                output_name = f"{db_path.stem}_full_{timestamp}.json"
                export_database_to_json(
                    db_path,
                    DB_DIR / output_name,
                    args.include_blobs
                )
    
    except Exception as e:
        print(f"\n✗ Erreur: {e}\n")
        raise


if __name__ == "__main__":
    main()