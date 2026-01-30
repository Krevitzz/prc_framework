
# Script: scripts/r1_execution/estimate_compute_load.py

"""
Estime charge calcul R1.1 (temps, stockage).

CONFORMITÉ:
- Basé sur statistiques R0 (pas hypothèses)
- Validation faisabilité avant lancement
"""

import json
from pathlib import Path

INPUT_N2 = Path("outputs/r1_execution/sequences_n2.json")
INPUT_N3 = Path("outputs/r1_execution/sequences_n3_sample.json")

# Constantes R0 (statistiques empiriques)
ENCODINGS_R1 = 21  # Tous encodings
MODIFIERS_R1 = 3   # M0, M1, M2
SEEDS_R1 = 5       # 42, 123, 456, 789, 1011

TIME_PER_RUN_R0 = 30  # secondes (moyenne empirique R0)
SNAPSHOTS_PER_RUN_R0 = 200  # snapshots moyens R0
STORAGE_PER_SNAPSHOT = 50  # KB (empirique)

def estimate_compute_load():
    """
    Estime temps + stockage R1.1.
    
    FORMULES:
    - N_runs = N_sequences × N_encodings × N_modifiers × N_seeds
    - Time_total = N_runs × TIME_PER_RUN × N_gammas_per_seq
    - Storage_total = N_runs × SNAPSHOTS × STORAGE × N_gammas_per_seq
    """
    
    # Charger séquences
    with open(INPUT_N2) as f:
        n2_data = json.load(f)
    n_sequences_n2 = n2_data['n_sequences']
    
    with open(INPUT_N3) as f:
        n3_data = json.load(f)
    n_sequences_n3 = n3_data['n_sequences_sampled']
    
    # Calculs n=2
    n_runs_n2 = n_sequences_n2 * ENCODINGS_R1 * MODIFIERS_R1 * SEEDS_R1
    time_n2_hours = (n_runs_n2 * TIME_PER_RUN_R0 * 2) / 3600  # 2 gammas/seq
    storage_n2_gb = (n_runs_n2 * SNAPSHOTS_PER_RUN_R0 * STORAGE_PER_SNAPSHOT * 2) / (1024**2)
    
    # Calculs n=3
    n_runs_n3 = n_sequences_n3 * ENCODINGS_R1 * MODIFIERS_R1 * SEEDS_R1
    time_n3_hours = (n_runs_n3 * TIME_PER_RUN_R0 * 3) / 3600  # 3 gammas/seq
    storage_n3_gb = (n_runs_n3 * SNAPSHOTS_PER_RUN_R0 * STORAGE_PER_SNAPSHOT * 3) / (1024**2)
    
    # Total
    total_runs = n_runs_n2 + n_runs_n3
    total_time_hours = time_n2_hours + time_n3_hours
    total_storage_gb = storage_n2_gb + storage_n3_gb
    
    # Rapport
    print("="*70)
    print("ESTIMATION CHARGE CALCUL R1.1")
    print("="*70)
    print(f"\nSÉQUENCES n=2:")
    print(f"  Séquences:     {n_sequences_n2}")
    print(f"  Runs totaux:   {n_runs_n2:,}")
    print(f"  Temps estimé:  {time_n2_hours:.1f} heures ({time_n2_hours/24:.1f} jours)")
    print(f"  Stockage:      {storage_n2_gb:.2f} GB")
    
    print(f"\nSÉQUENCES n=3 (échantillon):")
    print(f"  Séquences:     {n_sequences_n3}")
    print(f"  Runs totaux:   {n_runs_n3:,}")
    print(f"  Temps estimé:  {time_n3_hours:.1f} heures ({time_n3_hours/24:.1f} jours)")
    print(f"  Stockage:      {storage_n3_gb:.2f} GB")
    
    print(f"\nTOTAL R1.1:")
    print(f"  Runs totaux:   {total_runs:,}")
    print(f"  Temps estimé:  {total_time_hours:.1f} heures ({total_time_hours/24:.1f} jours)")
    print(f"  Stockage:      {total_storage_gb:.2f} GB")
    
    print(f"\nRECOMMANDATIONS:")
    if total_time_hours < 168:  # 1 semaine
        print("  ✓ Temps acceptable (< 1 semaine)")
    else:
        print(f"  ⚠ Temps élevé ({total_time_hours/24:.1f} jours)")
        print("    → Envisager parallélisation ou réduction échantillon")
    
    if total_storage_gb < 100:
        print("  ✓ Stockage acceptable (< 100 GB)")
    else:
        print(f"  ⚠ Stockage élevé ({total_storage_gb:.1f} GB)")
        print("    → Envisager compression additionnelle")
    
    print("="*70)
    
    # Sauvegarder
    output = {
        'n2': {
            'n_sequences': n_sequences_n2,
            'n_runs': n_runs_n2,
            'time_hours': time_n2_hours,
            'storage_gb': storage_n2_gb
        },
        'n3': {
            'n_sequences': n_sequences_n3,
            'n_runs': n_runs_n3,
            'time_hours': time_n3_hours,
            'storage_gb': storage_n3_gb
        },
        'total': {
            'n_runs': total_runs,
            'time_hours': total_time_hours,
            'time_days': total_time_hours / 24,
            'storage_gb': total_storage_gb
        }
    }
    
    OUTPUT_DIR = Path("outputs/r1_execution")
    with open(OUTPUT_DIR / "compute_load_estimate.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    estimate_compute_load()