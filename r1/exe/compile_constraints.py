
# Script: scripts/r1_3_convergence/compile_constraints.py

"""
Compile contraintes totales R0 + R1.1 + R1.2.

CONFORMITÉ:
- Réutilise convergence_evaluator.py
- Gestion optionnelle R1.2 (si skip)
"""

import json
import pandas as pd
from pathlib import Path
from tests.utilities.utils.convergence_evaluator import (
    compile_constraints_r0,
    compile_constraints_r1_1,
    compile_constraints_r1_2,
    compile_all_constraints
)

# Inputs
INPUT_GAMMA_STATS = Path("outputs/phase0/gamma_stability_stats_r0.csv")
INPUT_CLASSIFICATION_R1_1 = Path("outputs/r1_analysis/classification_sequences_r1.json")
INPUT_COMPENSATION_R1_2 = Path("outputs/r1_2_compensation/compensation_analysis.json")

OUTPUT_DIR = Path("outputs/r1_3_convergence")


def main():
    """Compile contraintes totales."""
    
    print("="*70)
    print("COMPILATION CONTRAINTES TOTALES")
    print("="*70)
    
    # 1. Charger données
    print("\n1. Chargement données...")
    
    df_gamma_stats = pd.read_csv(INPUT_GAMMA_STATS)
    print(f"   ✓ Gamma stats R0: {len(df_gamma_stats)} gammas")
    
    with open(INPUT_CLASSIFICATION_R1_1) as f:
        classification_r1_1 = json.load(f)
    print(f"   ✓ Classification R1.1")
    
    # R1.2 optionnel
    r1_2_available = INPUT_COMPENSATION_R1_2.exists()
    if r1_2_available:
        with open(INPUT_COMPENSATION_R1_2) as f:
            compensation_r1_2 = json.load(f)
        print(f"   ✓ Compensation R1.2 (disponible)")
    else:
        compensation_r1_2 = None
        print(f"   ⊘ Compensation R1.2 (skip)")
    
    # 2. Compiler contraintes
    print("\n2. Compilation contraintes...")
    
    constraints_r0 = compile_constraints_r0(df_gamma_stats)
    print(f"   ✓ Contraintes R0: {len(constraints_r0)} gammas")
    
    constraints_r1_1 = compile_constraints_r1_1(classification_r1_1)
    print(f"   ✓ Contraintes R1.1: {len(constraints_r1_1)} gammas")
    
    if r1_2_available:
        constraints_r1_2 = compile_constraints_r1_2(compensation_r1_2)
        print(f"   ✓ Contraintes R1.2: {len(constraints_r1_2)} gammas")
    else:
        constraints_r1_2 = None
    
    # 3. Fusionner
    print("\n3. Fusion contraintes...")
    
    all_constraints = compile_all_constraints(
        constraints_r0,
        constraints_r1_1,
        constraints_r1_2
    )
    
    print(f"   ✓ Contraintes totales: {len(all_constraints)} gammas")
    
    # 4. Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / "constraints_compiled.json", 'w') as f:
        json.dump(all_constraints, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'constraints_compiled.json'}")
    print("="*70)


if __name__ == "__main__":
    main()