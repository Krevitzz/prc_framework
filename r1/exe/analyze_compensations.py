
# Script: scripts/r1_2_compensation/analyze_compensations.py

"""
Analyse mécanismes compensation R1.2.

CONFORMITÉ:
- Réutilise sequence_analyzer.analyze_compensations()
- Baselines R0 en input
"""

import json
from pathlib import Path
from tests.utilities.utils.data_loading import load_all_observations
from tests.utilities.utils.sequence_analyzer import analyze_compensations

INPUT_BASELINES = Path("outputs/r1_analysis/baselines_r0.json")
INPUT_LOG = Path("outputs/r1_2_compensation/execution_compensated_log.json")
OUTPUT_DIR = Path("outputs/r1_2_compensation")


def main():
    """Analyse compensations R1.2."""
    
    print("="*70)
    print("ANALYSE COMPENSATIONS R1.2")
    print("="*70)
    
    # Charger baselines R0
    print("\n1. Chargement baselines R0...")
    with open(INPUT_BASELINES) as f:
        baselines_r0 = json.load(f)
    
    baseline_explosion = baselines_r0['explosion_rate']
    print(f"   ✓ Baseline explosion: {baseline_explosion:.4%}")
    
    # Charger sequence_exec_ids compensées
    print("\n2. Chargement observations compensées...")
    with open(INPUT_LOG) as f:
        log = json.load(f)
    
    sequence_exec_ids = log['sequence_exec_ids']
    
    # Charger observations (filtrer par sequence_exec_id)
    # NOTE: Nécessite query DB spécifique
    # Pour MVP, charger toutes observations R1 puis filtrer
    
    all_observations = load_all_observations(
        params_config_id='params_default_v1',
        phase='R1'
    )
    
    # Filtrer séquences compensées
    observations_compensated = [
        obs for obs in all_observations
        if obs.get('observation_data', {}).get('run_metadata', {}).get('sequence_exec_id') in sequence_exec_ids
    ]
    
    print(f"   ✓ {len(observations_compensated)} observations")
    
    # Analyser
    print("\n3. Analyse compensations...")
    analysis = analyze_compensations(observations_compensated, baseline_explosion)
    
    print(f"\n✓ Résultats:")
    print(f"  Total séquences:       {analysis['n_sequences_total']}")
    print(f"  Compensations SUCCESS: {analysis['n_compensations_success']} ({analysis['compensation_rate']*100:.1f}%)")
    print(f"  Compensations FAILED:  {analysis['n_compensations_failed']} ({(1-analysis['compensation_rate'])*100:.1f}%)")
    
    print(f"\n  Patterns identifiés:")
    print(f"    Gammas explosifs compensables: {len(analysis['compensable_explosive_gammas'])}")
    for gid in analysis['compensable_explosive_gammas']:
        print(f"      - {gid}")
    
    print(f"\n    Gammas encadrants efficaces:   {len(analysis['effective_framing_gammas'])}")
    for gid in analysis['effective_framing_gammas']:
        print(f"      - {gid}")
    
    # Sauvegarder
    with open(OUTPUT_DIR / "compensation_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'compensation_analysis.json'}")
    print("="*70)


if __name__ == "__main__":
    main()