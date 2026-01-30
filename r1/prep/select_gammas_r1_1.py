# Script: scripts/r1_preparation/select_gammas_r1_1.py

"""
Sélectionne gammas pour Phase R1.1 (composition robuste).

CRITÈRES:
- Taux explosion ≤ 10%
- Au moins 3 gammas différents (diversité familles)

CONFORMITÉ:
- Basé sur données R0 (pas hypothèses)
- Validation explicite critères
"""

import pandas as pd
from pathlib import Path
import json

INPUT_CSV = Path("outputs/phase0/gamma_stability_stats_r0.csv")
OUTPUT_DIR = Path("outputs/phase0")

def select_gammas_r1_1():
    """
    Sélectionne gammas non-explosifs pour R1.1.
    
    OUTPUT:
    - Liste gamma_ids sélectionnés
    - Statistiques sélection
    - Justification critères
    """
    
    df = pd.read_csv(INPUT_CSV)
    
    # Filtrer INCLUDE
    selected = df[df['recommendation'] == 'INCLUDE']
    
    if len(selected) < 3:
        raise ValueError(f"Insuffisant gammas INCLUDE: {len(selected)} < 3 requis")
    
    # Extraire IDs
    gamma_ids_selected = selected['gamma_id'].tolist()
    
    # Statistiques
    stats = {
        'n_gammas_selected': len(gamma_ids_selected),
        'gamma_ids': gamma_ids_selected,
        'explosion_rate_mean': float(selected['explosion_rate'].mean()),
        'explosion_rate_max': float(selected['explosion_rate'].max()),
        'selection_criteria': {
            'explosion_rate_threshold': 0.10,
            'minimum_gammas': 3
        },
        'n_excluded': len(df) - len(selected),
        'excluded_gamma_ids': df[df['recommendation'] == 'EXCLUDE']['gamma_id'].tolist()
    }
    
    # Sauvegarder
    with open(OUTPUT_DIR / "gammas_selected_r1_1.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Sélection {len(gamma_ids_selected)} gammas pour R1.1")
    print(f"  IDs: {', '.join(gamma_ids_selected)}")
    print(f"  Taux explosion moyen: {stats['explosion_rate_mean']:.2%}")
    print(f"\n  Exclus: {', '.join(stats['excluded_gamma_ids'])}")
    
    return stats

if __name__ == "__main__":
    stats = select_gammas_r1_1()