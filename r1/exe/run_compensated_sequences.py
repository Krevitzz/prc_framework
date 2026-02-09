
# Script: scripts/r1_2_compensation/run_compensated_sequences.py

"""
Exécute séquences compensées R1.2.

CONFORMITÉ:
- Réutilise composition_runner.run_batch_composition()
- Pattern identique run_sequences_n3.py (R1.1)
"""

import json
from pathlib import Path
from prc_automation.composition_runner import run_batch_composition

INPUT_SEQUENCES = Path("outputs/r1_2_compensation/sequences_compensated.json")
OUTPUT_LOG = Path("outputs/r1_2_compensation/execution_compensated_log.json")

# Configuration R1.2 (basée estimate)
ENCODINGS_R1_2 = ['SYM-001', 'SYM-002', 'SYM-003']
MODIFIERS_R1_2 = ['M0']
SEEDS_R1_2 = [42, 123]


def run_compensated_sequences():
    """Exécute batch séquences compensées."""
    
    # Charger séquences
    with open(INPUT_SEQUENCES) as f:
        data = json.load(f)
    
    sequences = data['sequences']
    
    print(f"Séquences compensées à exécuter: {len(sequences)}")
    print(f"Encodings: {ENCODINGS_R1_2}")
    print(f"Modifiers: {MODIFIERS_R1_2}")
    print(f"Seeds:     {SEEDS_R1_2}")
    
    total_runs = len(sequences) * len(ENCODINGS_R1_2) * len(MODIFIERS_R1_2) * len(SEEDS_R1_2)
    print(f"\nTotal runs: {total_runs}")
    
    # Confirmation
    confirm = input(f"\nProcéder à l'exécution ? (y/n): ")
    if confirm.lower() != 'y':
        print("Annulé.")
        return
    
    # Exécuter
    print("\nDémarrage exécution...")
    
    sequence_exec_ids = run_batch_composition(
        sequences=sequences,
        encodings=ENCODINGS_R1_2,
        modifiers=MODIFIERS_R1_2,
        seeds=SEEDS_R1_2,
        phase='R1'
    )
    
    # Sauvegarder logs
    log = {
        'phase': 'R1.2',
        'type': 'compensated_sequences',
        'n_sequences_executed': len(sequences),
        'n_runs_completed': len(sequence_exec_ids),
        'sequence_exec_ids': sequence_exec_ids,
        'config': {
            'encodings': ENCODINGS_R1_2,
            'modifiers': MODIFIERS_R1_2,
            'seeds': SEEDS_R1_2
        }
    }
    
    with open(OUTPUT_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\n✓ Logs sauvegardés: {OUTPUT_LOG}")
    
    return sequence_exec_ids


if __name__ == "__main__":
    run_compensated_sequences()