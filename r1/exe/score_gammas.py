
# Script: scripts/r1_3_convergence/score_gammas.py

"""
Calcule scores admissibilité tous gammas.

CONFORMITÉ:
- Réutilise convergence_evaluator.score_all_gammas()
- Poids configurables (transparence)
"""

import json
from pathlib import Path
from tests.utilities.utils.convergence_evaluator import score_all_gammas

INPUT_CONSTRAINTS = Path("outputs/r1_3_convergence/constraints_compiled.json")
OUTPUT_DIR = Path("outputs/r1_3_convergence")

# Poids scoring (modifiables)
WEIGHTS = {
    'r0_explosion_penalty': -10.0,
    'r0_recommendation': 5.0,
    'r1_1_robustness': 10.0,
    'r1_2_compensable': 3.0,
    'r1_2_effective_framing': 3.0
}


def main():
    """Calcule scores admissibilité."""
    
    print("="*70)
    print("SCORING ADMISSIBILITÉ GAMMAS")
    print("="*70)
    
    # Charger contraintes
    print("\n1. Chargement contraintes...")
    with open(INPUT_CONSTRAINTS) as f:
        constraints = json.load(f)
    
    print(f"   ✓ {len(constraints)} gammas")
    
    # Scorer
    print("\n2. Calcul scores...")
    print(f"   Poids utilisés:")
    for key, val in WEIGHTS.items():
        print(f"     {key}: {val}")
    
    scores = score_all_gammas(constraints, WEIGHTS)
    
    print(f"\n   ✓ {len(scores)} gammas scorés")
    
    # Afficher top 10
    print(f"\n   Top 10:")
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for rank, (gamma_id, data) in enumerate(ranked[:10], 1):
        print(f"     {rank}. {gamma_id}: {data['score']:.3f}")
    
    # Sauvegarder
    with open(OUTPUT_DIR / "gamma_scores.json", 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'gamma_scores.json'}")
    print("="*70)


if __name__ == "__main__":
    main()