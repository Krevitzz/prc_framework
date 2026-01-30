# Script: extract_explosive_combinations.py

"""
Extrait les 26 combinaisons explosives identifiées en R0.

CONFORMITÉ:
- Utilise data_loading.py (pas réimplémentation)
- Query read-only (db_raw immuable)
- Format output compatible profiling existant
"""

import sqlite3
import pandas as pd
from pathlib import Path
from tests.utilities.utils.data_loading import load_all_observations

DB_RAW = Path("prc_automation/prc_database/prc_r0_raw.db")
OUTPUT_DIR = Path("outputs/phase0")

def extract_explosive_combinations():
    """
    Extrait combinaisons avec status='ERROR' ou explosions détectées.
    
    SOURCES:
    - db_raw.executions (status)
    - db_results.observations (rejection_stats)
    
    DÉLÉGATION:
    - Utilise load_all_observations() pour cohérence
    - Pas de réimplémentation query
    """
    
    # 1. Charger observations R0
    observations = load_all_observations(
        params_config_id='params_default_v1',
        phase='R0'
    )
    
    # 2. Filtrer artefacts numériques
    from tests.utilities.utils.statistical_utils import filter_numeric_artifacts
    valid_obs, rejection_stats = filter_numeric_artifacts(observations)
    
    # 3. Extraire combinaisons rejetées
    conn = sqlite3.connect(DB_RAW)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT 
            gamma_id, 
            d_encoding_id, 
            modifier_id,
            COUNT(*) as n_seeds_affected
        FROM executions
        WHERE phase = 'R0' 
          AND (status = 'ERROR' OR status = 'NON_APPLICABLE')
        GROUP BY gamma_id, d_encoding_id, modifier_id
        ORDER BY gamma_id, d_encoding_id
    """)
    
    explosive_combinations = []
    for row in cursor.fetchall():
        explosive_combinations.append({
            'gamma_id': row[0],
            'd_encoding_id': row[1],
            'modifier_id': row[2],
            'n_seeds_affected': row[3],
            'severity': 'CRITICAL' if row[3] == 5 else 'PARTIAL'
        })
    
    conn.close()
    
    # 4. Sauvegarder
    df = pd.DataFrame(explosive_combinations)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_DIR / "explosive_combinations_r0.csv", index=False)
    
    print(f"✓ {len(explosive_combinations)} combinaisons explosives extraites")
    print(f"  CRITICAL (5/5 seeds): {len(df[df['severity']=='CRITICAL'])}")
    print(f"  PARTIAL (<5 seeds):   {len(df[df['severity']=='PARTIAL'])}")
    
    return df

if __name__ == "__main__":
    df = extract_explosive_combinations()
    print("\nDétail par gamma:")
    print(df.groupby('gamma_id').size())