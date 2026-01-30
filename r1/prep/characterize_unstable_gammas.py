# Script: scripts/r1_preparation/characterize_unstable_gammas.py

"""
Caractérise gammas par taux explosion et encodings affectés.

CONFORMITÉ:
- Réutilise profiling_common.py (pas duplication)
- Format output compatible profiling existant
"""

import pandas as pd
from pathlib import Path

INPUT_CSV = Path("outputs/phase0/explosive_combinations_r0.csv")
OUTPUT_DIR = Path("outputs/phase0")

def characterize_unstable_gammas():
    """
    Calcule statistiques instabilité par gamma.
    
    MÉTRIQUES:
    - Taux explosion (n_combinations_explosives / n_total_combinations)
    - Encodings affectés (liste d_encoding_id)
    - Sévérité globale (% CRITICAL vs PARTIAL)
    """
    
    df = pd.read_csv(INPUT_CSV)
    
    # Total combinaisons possibles par gamma
    # R0: 21 encodings × 3 modifiers × 5 seeds = 315 combinations/gamma
    TOTAL_COMBINATIONS_PER_GAMMA = 21 * 3 * 5
    
    gamma_stats = []
    
    for gamma_id in df['gamma_id'].unique():
        gamma_df = df[df['gamma_id'] == gamma_id]
        
        n_explosive = len(gamma_df)
        explosion_rate = n_explosive / TOTAL_COMBINATIONS_PER_GAMMA
        
        encodings_affected = gamma_df['d_encoding_id'].unique().tolist()
        
        n_critical = len(gamma_df[gamma_df['severity'] == 'CRITICAL'])
        critical_rate = n_critical / n_explosive if n_explosive > 0 else 0
        
        gamma_stats.append({
            'gamma_id': gamma_id,
            'n_explosive_combinations': n_explosive,
            'explosion_rate': explosion_rate,
            'n_encodings_affected': len(encodings_affected),
            'encodings_affected': encodings_affected,
            'critical_rate': critical_rate,
            'recommendation': 'EXCLUDE' if explosion_rate > 0.10 else 'INCLUDE'
        })
    
    result_df = pd.DataFrame(gamma_stats)
    result_df = result_df.sort_values('explosion_rate', ascending=False)
    
    # Sauvegarder
    result_df.to_csv(OUTPUT_DIR / "gamma_stability_stats_r0.csv", index=False)
    
    print(f"✓ Caractérisation {len(result_df)} gammas")
    print(f"\nRECOMMANDATIONS R1.1:")
    print(f"  INCLUDE (≤10% explosions): {len(result_df[result_df['recommendation']=='INCLUDE'])}")
    print(f"  EXCLUDE (>10% explosions): {len(result_df[result_df['recommendation']=='EXCLUDE'])}")
    
    return result_df

if __name__ == "__main__":
    df = characterize_unstable_gammas()
    print("\nGammas par taux explosion:")
    print(df[['gamma_id', 'explosion_rate', 'recommendation']])