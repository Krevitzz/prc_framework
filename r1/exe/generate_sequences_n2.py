
# Script: scripts/r1_execution/generate_sequences_n2.py

"""
Génère toutes paires (Γ₁, Γ₂) depuis gammas sélectionnés R0.

CONFORMITÉ:
- Réutilise composition_runner.generate_sequences()
- Pas de hardcoding gammas (lecture JSON Phase 0)
"""

import json
from pathlib import Path
from prc_automation.composition_runner import generate_sequences

INPUT_JSON = Path("outputs/phase0/gammas_selected_r1_1.json")
OUTPUT_DIR = Path("outputs/r1_execution")

def generate_sequences_n2():
    """
    Génère paires Γ₁→Γ₂ (permutations avec répétition).
    
    EXEMPLE:
    Si gammas = ['GAM-001', 'GAM-002', 'GAM-003']
    → 9 paires (3²)
    """
    
    # Charger gammas sélectionnés
    with open(INPUT_JSON) as f:
        selected = json.load(f)
    
    gamma_ids = selected['gamma_ids']
    
    print(f"Gammas sélectionnés: {len(gamma_ids)}")
    print(f"  IDs: {', '.join(gamma_ids)}")
    
    # Générer séquences n=2
    sequences = generate_sequences(gamma_ids, n=2, allow_repetition=True)
    
    print(f"\n✓ {len(sequences)} séquences générées (n=2)")
    print(f"  Exemple: {sequences[0]}")
    
    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output = {
        'n': 2,
        'gamma_ids_source': gamma_ids,
        'n_sequences': len(sequences),
        'sequences': sequences
    }
    
    with open(OUTPUT_DIR / "sequences_n2.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'sequences_n2.json'}")
    
    return sequences

if __name__ == "__main__":
    sequences = generate_sequences_n2()