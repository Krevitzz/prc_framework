
# Script: scripts/r1_execution/generate_sequences_n3_sample.py

"""
Génère échantillon triplets (Γ₁, Γ₂, Γ₃) pour R1.1.

STRATÉGIE ÉCHANTILLONNAGE:
- Pas toutes permutations (explosion combinatoire)
- Échantillon stratifié : paires robustes R0 + extension Γ₃
"""

import json
import random
from pathlib import Path
from prc_automation.composition_runner import generate_sequences

INPUT_N2 = Path("outputs/r1_execution/sequences_n2.json")
INPUT_GAMMAS = Path("outputs/phase0/gammas_selected_r1_1.json")
OUTPUT_DIR = Path("outputs/r1_execution")

SAMPLE_SIZE = 50  # Échantillon 50 triplets (ajustable)

def generate_sequences_n3_sample():
    """
    Génère échantillon triplets stratifiés.
    
    STRATÉGIE:
    1. Charger paires robustes n=2 (depuis résultats R1.1 si disponibles)
    2. Étendre chaque paire avec Γ₃ ∈ gammas_selected
    3. Échantillonner aléatoirement SAMPLE_SIZE triplets
    """
    
    # Charger gammas
    with open(INPUT_GAMMAS) as f:
        selected = json.load(f)
    gamma_ids = selected['gamma_ids']
    
    # Charger paires n=2
    with open(INPUT_N2) as f:
        n2_data = json.load(f)
    pairs = n2_data['sequences']
    
    print(f"Gammas disponibles: {len(gamma_ids)}")
    print(f"Paires n=2: {len(pairs)}")
    
    # Générer triplets (extension paires)
    triplets = []
    for pair in pairs:
        for gamma3 in gamma_ids:
            triplets.append(pair + [gamma3])
    
    print(f"\n✓ {len(triplets)} triplets théoriques (n=3)")
    
    # Échantillonner
    random.seed(42)  # Reproductibilité
    sampled = random.sample(triplets, min(SAMPLE_SIZE, len(triplets)))
    
    print(f"✓ {len(sampled)} triplets échantillonnés")
    print(f"  Exemple: {sampled[0]}")
    
    # Sauvegarder
    output = {
        'n': 3,
        'gamma_ids_source': gamma_ids,
        'n_sequences_total': len(triplets),
        'n_sequences_sampled': len(sampled),
        'sample_size_target': SAMPLE_SIZE,
        'sequences': sampled
    }
    
    with open(OUTPUT_DIR / "sequences_n3_sample.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'sequences_n3_sample.json'}")
    
    return sampled

if __name__ == "__main__":
    sequences = generate_sequences_n3_sample()