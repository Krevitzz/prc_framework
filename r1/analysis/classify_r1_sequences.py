
# Script: scripts/r1_analysis/classify_r1_sequences.py

"""
Classifie séquences R1.1 (robustes/dégradantes/instables).

CONFORMITÉ:
- Réutilise sequence_analyzer.py
- Baselines R0 en input
"""

import json
from pathlib import Path
from tests.utilities.utils.data_loading import load_all_observations
from tests.utilities.utils.sequence_analyzer import classify_sequences

INPUT_BASELINES = Path("outputs/r1_analysis/baselines_r0.json")
OUTPUT_DIR = Path("outputs/r1_analysis")


def main():
    """Classifie séquences R1.1."""
    
    print("="*70)
    print("CLASSIFICATION SÉQUENCES R1.1")
    print("="*70)
    
    # Charger baselines R0
    print("\n1. Chargement baselines R0...")
    with open(INPUT_BASELINES) as f:
        baselines_r0 = json.load(f)
    print(f"   ✓ Baseline explosion: {baselines_r0['explosion_rate']:.4%}")
    
    # Charger observations R1
    print("\n2. Chargement observations R1...")
    observations_r1 = load_all_observations(
        params_config_id='params_default_v1',
        phase='R1'
    )
    print(f"   ✓ {len(observations_r1)} observations")
    
    # Classifier
    print("\n3. Classification séquences...")
    classification = classify_sequences(
        observations_r1,
        baselines_r0,
        tolerance_factor=2.0
    )
    
    print(f"\n✓ Classification:")
    print(f"  Total séquences:    {classification['n_sequences_total']}")
    print(f"  ROBUSTES:           {classification['n_sequences_robustes']} ({classification['n_sequences_robustes']/classification['n_sequences_total']*100:.1f}%)")
    print(f"  DÉGRADANTES:        {classification['n_sequences_degradantes']} ({classification['n_sequences_degradantes']/classification['n_sequences_total']*100:.1f}%)")
    print(f"  INSTABLES:          {classification['n_sequences_instables']} ({classification['n_sequences_instables']/classification['n_sequences_total']*100:.1f}%)")
    
    print(f"\n  Métriques R1:")
    print(f"    Explosion rate: {classification['metrics']['explosion_rate_r1']:.4%}")
    print(f"    Concordance:    {classification['metrics']['concordance_r1']:.2%}")
    
    # Sauvegarder
    with open(OUTPUT_DIR / "classification_sequences_r1.json", 'w') as f:
        json.dump(classification, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'classification_sequences_r1.json'}")
    print("="*70)


if __name__ == "__main__":
    main()