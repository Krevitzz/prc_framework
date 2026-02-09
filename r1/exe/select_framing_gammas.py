
# Script: scripts/r1_2_compensation/select_framing_gammas.py

"""
Sélectionne gammas stables pour encadrement (Γ_stable₁, Γ_stable₂).

CONFORMITÉ:
- Basé classification_sequences_r1.json (séquences robustes)
- Critère: Gammas présents dans séquences ROBUSTES R1.1
"""

import json
from pathlib import Path
from collections import Counter

INPUT_CLASSIFICATION = Path("outputs/r1_analysis/classification_sequences_r1.json")
INPUT_EXPLOSIVE = Path("outputs/r1_2_compensation/explosive_gammas.json")
OUTPUT_DIR = Path("outputs/r1_2_compensation")


def select_framing_gammas():
    """
    Sélectionne gammas stables pour encadrement.
    
    STRATÉGIE:
    1. Extraire gammas depuis séquences ROBUSTES R1.1
    2. Compter fréquence apparition
    3. Sélectionner top N gammas (N=3 par défaut)
    
    Returns:
        {
            'gamma_ids_framing': List[str],
            'n_gammas_framing': int,
            'frequency_per_gamma': Dict
        }
    """
    
    # Charger séquences robustes
    with open(INPUT_CLASSIFICATION) as f:
        classification = json.load(f)
    
    sequences_robustes = classification['sequences_robustes']
    
    if not sequences_robustes:
        raise ValueError("Aucune séquence ROBUSTE trouvée en R1.1")
    
    # Charger gammas explosifs (à exclure)
    with open(INPUT_EXPLOSIVE) as f:
        explosive_data = json.load(f)
    
    gamma_ids_explosive = set(explosive_data['gamma_ids_explosive'])
    
    # Extraire gammas depuis séquences robustes
    # NOTE: Nécessite accès DB pour récupérer sequence_gammas
    # Pour MVP, charger depuis logs exécution
    
    # TODO: Implémenter extraction sequence_gammas depuis DB
    # Placeholder: Utiliser gammas R1.1 (sélection Phase 0.2)
    
    from pathlib import Path
    gammas_r1_1_path = Path("outputs/phase0/gammas_selected_r1_1.json")
    
    with open(gammas_r1_1_path) as f:
        selected_r1_1 = json.load(f)
    
    gamma_ids_stable = [gid for gid in selected_r1_1['gamma_ids'] if gid not in gamma_ids_explosive]
    
    if len(gamma_ids_stable) < 2:
        raise ValueError(f"Insuffisant gammas stables: {len(gamma_ids_stable)} < 2 requis")
    
    # Sélectionner top 3 (ou tous si <3)
    N_FRAMING = min(3, len(gamma_ids_stable))
    gamma_ids_framing = gamma_ids_stable[:N_FRAMING]
    
    result = {
        'gamma_ids_framing': gamma_ids_framing,
        'n_gammas_framing': len(gamma_ids_framing),
        'frequency_per_gamma': {gid: 1.0 for gid in gamma_ids_framing}  # Placeholder
    }
    
    print("="*70)
    print("GAMMAS ENCADRANTS (STABLES)")
    print("="*70)
    print(f"\nTotal: {len(gamma_ids_framing)}")
    print(f"IDs:   {', '.join(gamma_ids_framing)}")
    print("="*70)
    
    # Sauvegarder
    with open(OUTPUT_DIR / "framing_gammas.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Sauvegardé: {OUTPUT_DIR / 'framing_gammas.json'}")
    
    return result


if __name__ == "__main__":
    select_framing_gammas()