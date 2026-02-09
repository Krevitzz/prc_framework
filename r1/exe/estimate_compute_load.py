# Script: scripts/r1_execution/estimate_compute_load.py

"""
Estime charge calcul R1.1 (temps, stockage).

CONFORMITÉ:
- Basé sur statistiques R0 RÉELLES (pas hypothèses)
- Validation faisabilité avant lancement

MESURES R0 EMPIRIQUES:
- Durée totale: 10h pour 4095 exécutions kernel
- Stockage: 150 GB (raw + results)
"""

import json
from pathlib import Path

INPUT_N2 = Path("outputs/r1_execution/sequences_n2.json")
INPUT_N3 = Path("outputs/r1_execution/sequences_n3_sample.json")

# Constantes R0 (CONFIGURATION RÉELLE)
ENCODINGS_R1 = 21  # Tous encodings
MODIFIERS_R1 = 3   # M0, M1, M2
SEEDS_R1 = 5       # 42, 123, 456, 789, 1011

# Constantes MESURÉES R0 (vos mesures réelles)
TIME_PER_RUN_R0 = 8.8  # secondes/run (10h / 4095 runs)
STORAGE_PER_RUN_R0 = 37.5  # MB/run (150 GB / 4095 runs)

def estimate_compute_load():
    """
    Estime temps + stockage R1.1.
    
    FORMULES:
    - N_runs = N_sequences × N_encodings × N_modifiers × N_seeds
    - Time_total = N_runs × TIME_PER_RUN × N_gammas_per_seq
    - Storage_total = N_runs × STORAGE_PER_RUN × N_gammas_per_seq
    
    FACTEUR SÉQUENCE:
    - R0: 1 gamma/run
    - R1 n=2: 2 gammas/run → 2× temps, 2× stockage
    - R1 n=3: 3 gammas/run → 3× temps, 3× stockage
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
    storage_n2_gb = (n_runs_n2 * STORAGE_PER_RUN_R0 * 2) / 1024  # 2 gammas/seq
    
    # Calculs n=3
    n_runs_n3 = n_sequences_n3 * ENCODINGS_R1 * MODIFIERS_R1 * SEEDS_R1
    time_n3_hours = (n_runs_n3 * TIME_PER_RUN_R0 * 3) / 3600  # 3 gammas/seq
    storage_n3_gb = (n_runs_n3 * STORAGE_PER_RUN_R0 * 3) / 1024  # 3 gammas/seq
    
    # Total
    total_runs = n_runs_n2 + n_runs_n3
    total_time_hours = time_n2_hours + time_n3_hours
    total_storage_gb = storage_n2_gb + storage_n3_gb
    
    # Rapport
    print("="*70)
    print("ESTIMATION CHARGE CALCUL R1.1")
    print("="*70)
    print(f"\nBASE MESURES R0:")
    print(f"  Temps/run:    {TIME_PER_RUN_R0:.1f} secondes (mesuré)")
    print(f"  Stockage/run: {STORAGE_PER_RUN_R0:.1f} MB (mesuré)")
    
    print(f"\nSÉQUENCES n=2:")
    print(f"  Séquences:     {n_sequences_n2}")
    print(f"  Runs totaux:   {n_runs_n2:,}")
    print(f"  Temps estimé:  {time_n2_hours:.1f} heures ({time_n2_hours/24:.1f} jours)")
    print(f"  Stockage:      {storage_n2_gb:.1f} GB")
    
    print(f"\nSÉQUENCES n=3 (échantillon):")
    print(f"  Séquences:     {n_sequences_n3}")
    print(f"  Runs totaux:   {n_runs_n3:,}")
    print(f"  Temps estimé:  {time_n3_hours:.1f} heures ({time_n3_hours/24:.1f} jours)")
    print(f"  Stockage:      {storage_n3_gb:.1f} GB")
    
    print(f"\nTOTAL R1.1:")
    print(f"  Runs totaux:   {total_runs:,}")
    print(f"  Temps estimé:  {total_time_hours:.1f} heures ({total_time_hours/24:.1f} jours)")
    print(f"  Stockage:      {total_storage_gb:.1f} GB")
    
    print(f"\nRECOMMANDATIONS:")
    if total_time_hours < 168:  # 1 semaine
        print(f"  ✓ Temps acceptable ({total_time_hours/24:.1f} jours < 7 jours)")
    else:
        print(f"  ⚠ Temps élevé ({total_time_hours/24:.1f} jours)")
        print("    → Envisager parallélisation ou réduction échantillon")
    
    if total_storage_gb < 500:
        print(f"  ✓ Stockage acceptable ({total_storage_gb:.0f} GB < 500 GB)")
    elif total_storage_gb < 1000:
        print(f"  ⚠ Stockage élevé ({total_storage_gb:.0f} GB)")
        print("    → Vérifier espace disque disponible")
    else:
        print(f"  ❌ Stockage critique ({total_storage_gb:.0f} GB > 1 TB)")
        print("    → RÉDUIRE échantillon obligatoire")
    
    # Options réduction
    if total_storage_gb > 500:
        print(f"\nOPTIONS RÉDUCTION:")
        
        # Option 1: Réduire encodings
        storage_reduced_enc = total_storage_gb * (3/21)
        time_reduced_enc = total_time_hours * (3/21)
        print(f"  Option 1: ENCODINGS_R1 = 3 (au lieu de 21)")
        print(f"    → Stockage: {storage_reduced_enc:.0f} GB")
        print(f"    → Temps:    {time_reduced_enc:.1f}h ({time_reduced_enc/24:.1f} jours)")
        
        # Option 2: Réduire seeds
        storage_reduced_seeds = total_storage_gb * (2/5)
        time_reduced_seeds = total_time_hours * (2/5)
        print(f"  Option 2: SEEDS_R1 = 2 (au lieu de 5)")
        print(f"    → Stockage: {storage_reduced_seeds:.0f} GB")
        print(f"    → Temps:    {time_reduced_seeds:.1f}h ({time_reduced_seeds/24:.1f} jours)")
        
        # Option 3: Combiné
        storage_combined = total_storage_gb * (3/21) * (2/5)
        time_combined = total_time_hours * (3/21) * (2/5)
        print(f"  Option 3: ENCODINGS=3 + SEEDS=2")
        print(f"    → Stockage: {storage_combined:.0f} GB")
        print(f"    → Temps:    {time_combined:.1f}h ({time_combined/24:.1f} jours)")
    
    print("="*70)
    
    # Sauvegarder
    output = {
        'base_measures_r0': {
            'time_per_run_seconds': TIME_PER_RUN_R0,
            'storage_per_run_mb': STORAGE_PER_RUN_R0
        },
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