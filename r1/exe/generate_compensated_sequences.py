
# Script: scripts/r1_2_compensation/generate_compensated_sequences.py

"""
Génère séquences Γ_stable₁ → Γ_explosif → Γ_stable₂.

CONFORMITÉ:
- Réutilise composition_runner.generate_sequences() (pattern)
- Validation explicite structure (3 gammas obligatoire)
"""

import json
from pathlib import Path
from itertools import product

INPUT_EXPLOSIVE = Path("outputs/r1_2_compensation/explosive_gammas.json")
INPUT_FRAMING = Path("outputs/r1_2_compensation/framing_gammas.json")
OUTPUT_DIR = Path("outputs/r1_2_compensation")


def generate_compensated_sequences():
    """
    Génère séquences compensées (n=3 obligatoire).
    
    STRUCTURE: [Γ_stable₁, Γ_explosif, Γ_stable₂]
    
    COMBINATOIRE:
    - Γ_stable₁ ∈ gammas_framing
    - Γ_explosif ∈ gammas_explosive
    - Γ_stable₂ ∈ gammas_framing
    - Γ_stable₁ peut == Γ_stable₂ (autorisé)
    
    Returns:
        List[List[str]] (séquences)
    """
    
    # Charger gammas
    with open(INPUT_EXPLOSIVE) as f:
        explosive_data = json.load(f)
    gamma_ids_explosive = explosive_data['gamma_ids_explosive']
    
    with open(INPUT_FRAMING) as f:
        framing_data = json.load(f)
    gamma_ids_framing = framing_data['gamma_ids_framing']
    
    print("="*70)
    print("GÉNÉRATION SÉQUENCES COMPENSÉES")
    print("="*70)
    print(f"\nGammas explosifs:  {len(gamma_ids_explosive)} - {gamma_ids_explosive}")
    print(f"Gammas encadrants: {len(gamma_ids_framing)} - {gamma_ids_framing}")
    
    # Générer combinaisons
    sequences = []
    
    for gamma_stable_1 in gamma_ids_framing:
        for gamma_explosif in gamma_ids_explosive:
            for gamma_stable_2 in gamma_ids_framing:
                sequences.append([gamma_stable_1, gamma_explosif, gamma_stable_2])
    
    print(f"\n✓ {len(sequences)} séquences compensées générées")
    print(f"  Combinatoire: {len(gamma_ids_framing)} × {len(gamma_ids_explosive)} × {len(gamma_ids_framing)}")
    print(f"  Exemple: {sequences[0]}")
    
    # Sauvegarder
    output = {
        'n': 3,
        'n_sequences': len(sequences),
        'gamma_ids_explosive': gamma_ids_explosive,
        'gamma_ids_framing': gamma_ids_framing,
        'sequences': sequences
    }
    
    with open(OUTPUT_DIR / "sequences_compensated.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'sequences_compensated.json'}")
    print("="*70)
    
    return sequences


if __name__ == "__main__":
    generate_compensated_sequences()