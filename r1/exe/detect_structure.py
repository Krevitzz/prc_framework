
# Script: scripts/r1_3_convergence/detect_structure.py

"""
Détecte structure convergente (unique, famille, classes).

CONFORMITÉ:
- Réutilise convergence_evaluator.detect_convergent_structure()
- Basé métadonnées gammas (METADATA)
"""

import json
from pathlib import Path
from tests.utilities.utils.convergence_evaluator import (
    detect_convergent_structure,
    identify_common_properties
)
from tests.utilities.utils.data_loading import discover_entities

INPUT_SURVIVORS = Path("outputs/r1_3_convergence/survivors.json")
OUTPUT_DIR = Path("outputs/r1_3_convergence")


def main():
    """Détecte structure convergente."""
    
    print("="*70)
    print("DÉTECTION STRUCTURE CONVERGENTE")
    print("="*70)
    
    # 1. Charger survivants
    print("\n1. Chargement survivants...")
    with open(INPUT_SURVIVORS) as f:
        data = json.load(f)
    
    survivors = data['survivors']
    print(f"   ✓ {len(survivors)} survivants")
    
    # 2. Charger métadonnées gammas
    print("\n2. Découverte métadonnées gammas...")
    
    all_gammas = discover_entities('gamma', phase='R0')
    
    gamma_metadata = {}
    for gamma_info in all_gammas:
        gamma_id = gamma_info['id']
        metadata = gamma_info.get('metadata', {})
        gamma_metadata[gamma_id] = metadata
    
    print(f"   ✓ {len(gamma_metadata)} gammas métadonnées")
    
    # 3. Détecter structure
    print("\n3. Détection structure...")
    
    structure_type, description = detect_convergent_structure(
        survivors,
        gamma_metadata
    )
    
    print(f"\n   ✓ Structure détectée:")
    print(f"     Type:        {structure_type}")
    print(f"     Description: {description}")
    
    # 4. Identifier propriétés communes
    print("\n4. Identification propriétés communes...")
    
    common_props = identify_common_properties(
        survivors,
        gamma_metadata
    )
    
    print(f"\n   ✓ Propriétés communes:")
    print(f"     Familles ({common_props['n_families']}): {', '.join(common_props['families'])}")
    print(f"     d_applicability commune: {', '.join(common_props['d_applicability_common']) if common_props['d_applicability_common'] else 'Aucune'}")
    
    # 5. Sauvegarder
    output = {
        'structure_type': structure_type,
        'description': description,
        'n_survivors': len(survivors),
        'survivors': survivors,
        'common_properties': common_props
    }
    
    with open(OUTPUT_DIR / "convergent_structure.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'convergent_structure.json'}")
    print("="*70)


if __name__ == "__main__":
    main()