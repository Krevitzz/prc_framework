
# Script: scripts/r1_2_compensation/identify_explosive_gammas.py

"""
Identifie gammas explosifs (exclus R1.1) pour tests compensation.

CONFORMITÉ:
- Basé gamma_stability_stats_r0.csv (données empiriques)
- Pas de hardcoding (lecture CSV)
"""

import pandas as pd
from pathlib import Path
import json

INPUT_STATS = Path("outputs/phase0/gamma_stability_stats_r0.csv")
OUTPUT_DIR = Path("outputs/r1_2_compensation")


def identify_explosive_gammas():
    """
    Sélectionne gammas exclus R1.1 (explosion_rate > 10%).
    
    CRITÈRES:
    - explosion_rate > 10% (exclusion R1.1)
    - Au moins 1 gamma explosif disponible
    
    Returns:
        {
            'gamma_ids_explosive': List[str],
            'n_gammas_explosive': int,
            'stats_per_gamma': Dict
        }
    """
    
    df = pd.read_csv(INPUT_STATS)
    
    # Filtrer EXCLUDE (explosion_rate > 10%)
    explosive = df[df['recommendation'] == 'EXCLUDE']
    
    if len(explosive) == 0:
        raise ValueError("Aucun gamma explosif trouvé (tous < 10% explosions)")
    
    gamma_ids_explosive = explosive['gamma_id'].tolist()
    
    # Stats détaillées
    stats_per_gamma = {}
    for _, row in explosive.iterrows():
        stats_per_gamma[row['gamma_id']] = {
            'explosion_rate': row['explosion_rate'],
            'n_encodings_affected': row['n_encodings_affected'],
            'encodings_affected': eval(row['encodings_affected']) if isinstance(row['encodings_affected'], str) else row['encodings_affected']
        }
    
    result = {
        'gamma_ids_explosive': gamma_ids_explosive,
        'n_gammas_explosive': len(gamma_ids_explosive),
        'stats_per_gamma': stats_per_gamma
    }
    
    print("="*70)
    print("GAMMAS EXPLOSIFS (EXCLUS R1.1)")
    print("="*70)
    print(f"\nTotal: {len(gamma_ids_explosive)}")
    print(f"IDs:   {', '.join(gamma_ids_explosive)}")
    
    print(f"\nDétail par gamma:")
    for gamma_id, stats in stats_per_gamma.items():
        print(f"  {gamma_id}:")
        print(f"    Explosion rate: {stats['explosion_rate']:.2%}")
        print(f"    Encodings affectés: {stats['n_encodings_affected']}")
    
    print("="*70)
    
    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / "explosive_gammas.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'explosive_gammas.json'}")
    
    return result


if __name__ == "__main__":
    identify_explosive_gammas()