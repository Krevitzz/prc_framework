
# Script: scripts/r1_analysis/compute_r0_baselines.py

"""
Calcule baselines R0 pour comparaison R1.

CONFORMITÉ:
- Réutilise sequence_analyzer.py
- Basé sur observations R0 SUCCESS
"""

import json
from pathlib import Path
from tests.utilities.utils.data_loading import load_all_observations
from tests.utilities.utils.sequence_analyzer import compute_r0_baselines

OUTPUT_DIR = Path("outputs/r1_analysis")


def main():
    """Calcule et sauvegarde baselines R0."""
    
    print("="*70)
    print("CALCUL BASELINES R0")
    print("="*70)
    
    # Charger observations R0
    print("\n1. Chargement observations R0...")
    observations_r0 = load_all_observations(
        params_config_id='params_default_v1',
        phase='R0'
    )
    print(f"   ✓ {len(observations_r0)} observations")
    
    # Calculer baselines
    print("\n2. Calcul baselines...")
    baselines = compute_r0_baselines(observations_r0)
    
    print(f"\n✓ Baselines R0:")
    print(f"  Observations:       {baselines['n_observations']}")
    print(f"  Rejection rate:     {baselines['rejection_rate']:.4%}")
    print(f"  Explosion rate:     {baselines['explosion_rate']:.4%}")
    print(f"  Regime concordance: {baselines['regime_concordance']:.2%}")
    
    print(f"\n  Tests robustes (outlier rates):")
    for test_name, rate in sorted(baselines['test_robustness'].items(), key=lambda x: x[1]):
        print(f"    {test_name}: {rate:.2%}")
    
    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / "baselines_r0.json", 'w') as f:
        json.dump(baselines, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'baselines_r0.json'}")
    print("="*70)


if __name__ == "__main__":
    main()