
# Script: scripts/r1_3_convergence/filter_survivors.py

"""
Filtre gammas survivants (score > threshold).

CONFORMITÉ:
- Réutilise convergence_evaluator.filter_survivors()
- Threshold configurable (transparence)
"""

import json
from pathlib import Path
from tests.utilities.utils.convergence_evaluator import filter_survivors

INPUT_SCORES = Path("outputs/r1_3_convergence/gamma_scores.json")
OUTPUT_DIR = Path("outputs/r1_3_convergence")

# Threshold (ajustable)
THRESHOLD = 0.5


def main():
    """Filtre survivants."""
    
    print("="*70)
    print("FILTRAGE GAMMAS SURVIVANTS")
    print("="*70)
    
    # Charger scores
    print("\n1. Chargement scores...")
    with open(INPUT_SCORES) as f:
        scores = json.load(f)
    
    print(f"   ✓ {len(scores)} gammas")
    
    # Filtrer
    print(f"\n2. Filtrage (threshold={THRESHOLD})...")
    
    survivors, stats = filter_survivors(scores, threshold=THRESHOLD)
    
    print(f"\n   ✓ Statistiques:")
    print(f"     Candidats initiaux: {stats['n_candidates_initial']}")
    print(f"     Survivants:         {stats['n_survivors']}")
    print(f"     Facteur réduction:  {stats['reduction_factor']:.2f} ({stats['reduction_factor']*100:.1f}%)")
    print(f"     Threshold utilisé:  {stats['threshold_used']}")
    
    print(f"\n   Survivants:")
    for gamma_id in survivors:
        score = scores[gamma_id]['score']
        rank = scores[gamma_id]['rank']
        print(f"     {rank}. {gamma_id}: {score:.3f}")
    
    # Sauvegarder
    output = {
        'survivors': survivors,
        'stats': stats,
        'threshold': THRESHOLD
    }
    
    with open(OUTPUT_DIR / "survivors.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'survivors.json'}")
    print("="*70)


if __name__ == "__main__":
    main()