#!/usr/bin/env python3
# tests/validate_scoring_configs.py
"""
Valide tous les fichiers scoring YAML.

Usage:
    python tests/validate_scoring_configs.py
    python tests/validate_scoring_configs.py --verbose
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List


VALID_PATHOLOGY_TYPES = [
    'S1_COLLAPSE',
    'S2_EXPLOSION',
    'S3_PLATEAU',
    'S4_INSTABILITY',
    'MAPPING'
]

VALID_AGGREGATION_MODES = [
    'max',
    'weighted_mean',
    'weighted_max'
]


def validate_scoring_config(config_path: Path, verbose: bool = False) -> List[str]:
    """
    Valide un fichier scoring YAML.
    
    Args:
        config_path: Chemin fichier YAML
        verbose: Afficher détails
    
    Returns:
        Liste erreurs (vide si valide)
    """
    errors = []
    
    # Charger YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return [f"Erreur parse YAML: {e}"]
    
    # Metadata obligatoires
    for key in ['version', 'config_id', 'description']:
        if key not in config:
            errors.append(f"Manque metadata: {key}")
    
    # Tests présents
    if 'tests' not in config:
        errors.append("Manque section 'tests'")
        return errors
    
    # Valider chaque test
    for test_id, test_config in config['tests'].items():
        prefix = f"Test {test_id}"
        
        # Test weight
        if 'test_weight' not in test_config:
            errors.append(f"{prefix}: manque 'test_weight'")
        
        # Aggregation mode
        if 'aggregation_mode' in test_config:
            mode = test_config['aggregation_mode']
            if mode not in VALID_AGGREGATION_MODES:
                errors.append(
                    f"{prefix}: aggregation_mode invalide '{mode}'. "
                    f"Valides: {VALID_AGGREGATION_MODES}"
                )
        
        # Scoring rules
        if 'scoring_rules' not in test_config:
            errors.append(f"{prefix}: manque 'scoring_rules'")
            continue
        
        # Valider chaque règle
        for metric_name, rule in test_config['scoring_rules'].items():
            rule_prefix = f"{prefix}.{metric_name}"
            
            # Pathology type
            if 'pathology_type' not in rule:
                errors.append(f"{rule_prefix}: manque 'pathology_type'")
                continue
            
            ptype = rule['pathology_type']
            if ptype not in VALID_PATHOLOGY_TYPES:
                errors.append(
                    f"{rule_prefix}: pathology_type invalide '{ptype}'. "
                    f"Valides: {VALID_PATHOLOGY_TYPES}"
                )
                continue
            
            # Validation selon type
            if ptype == 'S1_COLLAPSE':
                if 'threshold_low' not in rule:
                    errors.append(f"{rule_prefix}: S1 nécessite 'threshold_low'")
            
            elif ptype == 'S2_EXPLOSION':
                if 'threshold_high' not in rule:
                    errors.append(f"{rule_prefix}: S2 nécessite 'threshold_high'")
            
            elif ptype == 'S3_PLATEAU':
                if 'interval_toxic' not in rule:
                    errors.append(f"{rule_prefix}: S3 nécessite 'interval_toxic'")
                elif not isinstance(rule['interval_toxic'], list) or len(rule['interval_toxic']) != 2:
                    errors.append(f"{rule_prefix}: interval_toxic doit être [low, high]")
            
            elif ptype == 'S4_INSTABILITY':
                if 'delta_critical' not in rule:
                    errors.append(f"{rule_prefix}: S4 nécessite 'delta_critical'")
            
            elif ptype == 'MAPPING':
                if 'mapping' not in rule:
                    errors.append(f"{rule_prefix}: MAPPING nécessite 'mapping'")
                elif not isinstance(rule['mapping'], dict):
                    errors.append(f"{rule_prefix}: mapping doit être dict")
            
            # Weight optionnel mais doit être numérique
            if 'weight' in rule:
                try:
                    float(rule['weight'])
                except:
                    errors.append(f"{rule_prefix}: weight doit être numérique")
    
    if verbose and not errors:
        print(f"✓ {config_path.name} valide")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Valide configs scoring YAML")
    parser.add_argument('--verbose', action='store_true',
                       help='Afficher détails')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VALIDATION CONFIGS SCORING")
    print("="*70 + "\n")
    
    # Trouver tous fichiers scoring
    config_dir = Path("tests/config")
    
    global_configs = list(config_dir.glob("global/scoring_*.yaml"))
    test_configs = list(config_dir.glob("tests/*/scoring_*.yaml"))
    
    all_configs = global_configs + test_configs
    
    print(f"Configs trouvées:")
    print(f"  Global: {len(global_configs)}")
    print(f"  Tests:  {len(test_configs)}")
    print(f"  Total:  {len(all_configs)}\n")
    
    # Valider
    total_errors = 0
    
    for config_path in all_configs:
        errors = validate_scoring_config(config_path, args.verbose)
        
        if errors:
            print(f"❌ {config_path}")
            for err in errors:
                print(f"   - {err}")
            print()
            total_errors += len(errors)
    
    # Résumé
    print("="*70)
    if total_errors == 0:
        print("✓ TOUTES LES CONFIGS VALIDES")
    else:
        print(f"❌ {total_errors} ERREURS DÉTECTÉES")
    print("="*70 + "\n")
    
    return 0 if total_errors == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())