
# Script: scripts/r1_execution/run_sequences_n3.py

"""
Exécute échantillon séquences n=3 (triplets).

CONFORMITÉ:
- Identique run_sequences_n2.py (réutilisation pattern)
"""

import json
from pathlib import Path
from prc_automation.composition_runner import run_batch_composition

INPUT_SEQUENCES = Path("outputs/r1_execution/sequences_n3_sample.json")
OUTPUT_LOG = Path("outputs/r1_execution/execution_n3_log.json")

# Configuration R1.1 (identique n=2)
ENCODINGS_R1 = ['SYM-001', 'SYM-002', 'SYM-003']
MODIFIERS_R1 = ['M0']
SEEDS_R1 = [42, 123]

def run_sequences_n3():
    """Exécute batch séquences n=3."""
    
    # Charger séquences
    with open(INPUT_SEQUENCES) as f:
        data = json.load(f)
    
    sequences = data['sequences']
    
    print(f"Séquences à exécuter: {len(sequences)}")
    print(f"Encodings: {ENCODINGS_R1}")
    print(f"Modifiers: {MODIFIERS_R1}")
    print(f"Seeds:     {SEEDS_R1}")
    
    total_runs = len(sequences) * len(ENCODINGS_R1) * len(MODIFIERS_R1) * len(SEEDS_R1)
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
        encodings=ENCODINGS_R1,
        modifiers=MODIFIERS_R1,
        seeds=SEEDS_R1,
        phase='R1'
    )
    
    # Sauvegarder logs
    log = {
        'n': 3,
        'n_sequences_executed': len(sequences),
        'n_runs_completed': len(sequence_exec_ids),
        'sequence_exec_ids': sequence_exec_ids,
        'config': {
            'encodings': ENCODINGS_R1,
            'modifiers': MODIFIERS_R1,
            'seeds': SEEDS_R1
        }
    }
    
    with open(OUTPUT_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\n✓ Logs sauvegardés: {OUTPUT_LOG}")
    
    return sequence_exec_ids

if __name__ == "__main__":
    run_sequences_n3()