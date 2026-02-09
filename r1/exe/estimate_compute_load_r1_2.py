
# Script: scripts/r1_2_compensation/estimate_compute_load_r1_2.py

"""
Estime charge calcul Phase R1.2.

CONFORMITÉ:
- Similaire estimate_compute_load.py (Phase R1.1)
- Basé statistiques empiriques R0
"""

import json
from pathlib import Path

INPUT_SEQUENCES = Path("outputs/r1_2_compensation/sequences_compensated.json")

# Constantes R0 (empiriques)
ENCODINGS_R1_2 = 3  # Échantillon (ajustable)
MODIFIERS_R1_2 = 1  # M0 uniquement (baseline)
SEEDS_R1_2 = 2      # 2 seeds (ajustable)

TIME_PER_RUN_R0 = 30  # secondes
SNAPSHOTS_PER_RUN_R0 = 200
STORAGE_PER_SNAPSHOT = 50  # KB


def estimate_compute_load_r1_2():
    """Estime temps + stockage R1.2."""
    
    with open(INPUT_SEQUENCES) as f:
        data = json.load(f)
    
    n_sequences = data['n_sequences']
    
    # Calculs
    n_runs = n_sequences * ENCODINGS_R1_2 * MODIFIERS_R1_2 * SEEDS_R1_2
    time_hours = (n_runs * TIME_PER_RUN_R0 * 3) / 3600  # 3 gammas/seq
    storage_gb = (n_runs * SNAPSHOTS_PER_RUN_R0 * STORAGE_PER_SNAPSHOT * 3) / (1024**2)
    
    print("="*70)
    print("ESTIMATION CHARGE CALCUL R1.2")
    print("="*70)
    print(f"\nSéquences compensées: {n_sequences}")
    print(f"Encodings:            {ENCODINGS_R1_2}")
    print(f"Modifiers:            {MODIFIERS_R1_2}")
    print(f"Seeds:                {SEEDS_R1_2}")
    
    print(f"\nRuns totaux:          {n_runs:,}")
    print(f"Temps estimé:         {time_hours:.1f} heures ({time_hours/24:.1f} jours)")
    print(f"Stockage estimé:      {storage_gb:.2f} GB")
    
    print(f"\nRECOMMANDATIONS:")
    if time_hours < 48:  # 2 jours
        print("  ✓ Temps acceptable (< 2 jours)")
    else:
        print(f"  ⚠ Temps élevé ({time_hours/24:.1f} jours)")
        print("    → Envisager réduction échantillon")
    
    if storage_gb < 50:
        print("  ✓ Stockage acceptable (< 50 GB)")
    else:
        print(f"  ⚠ Stockage élevé ({storage_gb:.1f} GB)")
    
    print("="*70)
    
    # Sauvegarder
    output = {
        'n_sequences': n_sequences,
        'n_runs': n_runs,
        'time_hours': time_hours,
        'time_days': time_hours / 24,
        'storage_gb': storage_gb,
        'config': {
            'encodings': ENCODINGS_R1_2,
            'modifiers': MODIFIERS_R1_2,
            'seeds': SEEDS_R1_2
        }
    }
    
    OUTPUT_DIR = Path("outputs/r1_2_compensation")
    with open(OUTPUT_DIR / "compute_load_estimate_r1_2.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == "__main__":
    estimate_compute_load_r1_2()