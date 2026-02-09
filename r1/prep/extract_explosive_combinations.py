#!/usr/bin/env python3
"""
Phase 0 - Étape 0.2.1 : Extraction combinaisons explosives R0

RESPONSABILITÉ:
- Query db_raw.executions (status='SUCCESS')
- Query db_results.observations (identifier rejections)
- Extraire 26 combinaisons (gamma, encoding, modifier, seed)

CONFORMITÉ Charter 6.1 Section 4.5:
- Étape 1 MÉTIER validée (voir extract_explosive_combinations_spec.md)
- Étape 2 ORGANIGRAMME validée
- Étape 3 STRUCTURE validée
- Étape 4 CODE (cette implémentation)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths DB
DB_RAW = Path("prc_automation/prc_database/prc_r0_raw.db")
DB_RESULTS = Path("prc_automation/prc_database/prc_r0_results.db")
OUTPUT_DIR = Path("outputs/phase0")


def extract_explosive_combinations():
    """
    Extrait combinaisons explosives R0.
    
    CRITÈRE EXPLOSION:
    - Observation db_results avec artefacts numériques (inf/nan)
    - OU exec_id db_raw avec status='ERROR'
    
    Returns:
        DataFrame avec colonnes:
        - gamma_id, d_encoding_id, modifier_id, seed
        - n_errors (nombre observations invalides)
        - tests_affected (liste tests en erreur)
    """
    
    print("="*70)
    print("EXTRACTION COMBINAISONS EXPLOSIVES R0")
    print("="*70)
    
    # =========================================================================
    # 1. Charger observations invalides db_results
    # =========================================================================
    
    print("\n1. Chargement observations db_results...")
    
    if not DB_RESULTS.exists():
        raise FileNotFoundError(f"DB manquante: {DB_RESULTS}")
    
    conn_results = sqlite3.connect(DB_RESULTS)
    
    # Query observations ERROR ou avec valeurs invalides
    query = """
    SELECT 
        gamma_id, 
        d_encoding_id, 
        modifier_id, 
        seed,
        test_name,
        status,
        message
    FROM observations
    WHERE phase = 'R0'
      AND (status = 'ERROR' OR message LIKE '%inf%' OR message LIKE '%nan%')
    """
    
    df_invalid = pd.read_sql_query(query, conn_results)
    conn_results.close()
    
    print(f"   ✓ {len(df_invalid)} observations invalides")
    
    if len(df_invalid) == 0:
        print("\n✓ Aucune combinaison explosive détectée (dataset parfait)")
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Créer CSV vide avec colonnes attendues
        df_empty = pd.DataFrame(columns=[
            'gamma_id', 'd_encoding_id', 'modifier_id', 'seed',
            'n_errors', 'tests_affected'
        ])
        df_empty.to_csv(OUTPUT_DIR / "explosive_combinations_r0.csv", index=False)
        
        return df_empty
    
    # =========================================================================
    # 2. Grouper par combinaison (gamma, encoding, modifier, seed)
    # =========================================================================
    
    print("\n2. Groupement par combinaison...")
    
    grouped = df_invalid.groupby([
        'gamma_id', 'd_encoding_id', 'modifier_id', 'seed'
    ]).agg({
        'test_name': lambda x: list(x),
        'status': 'count'  # Compte nombre erreurs
    }).reset_index()
    
    grouped.columns = [
        'gamma_id', 'd_encoding_id', 'modifier_id', 'seed',
        'tests_affected', 'n_errors'
    ]
    
    print(f"   ✓ {len(grouped)} combinaisons explosives")
    
    # =========================================================================
    # 3. Afficher statistiques
    # =========================================================================
    
    print("\n3. Statistiques:")
    print(f"   Total combinaisons explosives: {len(grouped)}")
    print(f"   Total observations invalides:  {len(df_invalid)}")
    
    # Gammas dominants
    gamma_counts = df_invalid['gamma_id'].value_counts()
    print(f"\n   Gammas dominants (top 5):")
    for gamma_id, count in gamma_counts.head(5).items():
        print(f"     {gamma_id}: {count} observations invalides")
    
    # Encodings dominants
    encoding_counts = df_invalid['d_encoding_id'].value_counts()
    print(f"\n   Encodings dominants (top 5):")
    for encoding_id, count in encoding_counts.head(5).items():
        print(f"     {encoding_id}: {count} observations invalides")
    
    # =========================================================================
    # 4. Sauvegarder
    # =========================================================================
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = OUTPUT_DIR / "explosive_combinations_r0.csv"
    grouped.to_csv(output_path, index=False)
    
    print(f"\n✓ Sauvegardé: {output_path}")
    print("="*70)
    
    return grouped


if __name__ == "__main__":
    df = extract_explosive_combinations()