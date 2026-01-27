#!/usr/bin/env python3
"""
Diagnostic Phase 10 - Compare comportement avant/après refactor.
"""

import sys
from pathlib import Path

# Imports
from tests.utilities.utils.data_loading import discover_entities

def diagnose_discovery():
    """Diagnostic 1: Discovery entities."""
    print("="*80)
    print("DIAGNOSTIC 1: DISCOVERY ENTITIES")
    print("="*80)
    
    for entity_type in ['gamma', 'encoding', 'modifier']:
        print(f"\n--- {entity_type.upper()} ---")
        entities = discover_entities(entity_type, phase='R0')
        
        for entity in entities:
            print(f"\nID: {entity['id']}")
            print(f"  module_path: {entity['module_path']}")
            print(f"  function_name: {entity['function_name']}")
            print(f"  phase: {entity['phase']}")
            
            metadata = entity.get('metadata', {})
            if metadata:
                print(f"  metadata keys: {list(metadata.keys())}")
                if 'd_applicability' in metadata:
                    print(f"    d_applicability: {metadata['d_applicability']}")
                else:
                    print(f"     d_applicability MANQUANT")
            else:
                print(f"   metadata VIDE")


def diagnose_modifier_signatures():
    """Diagnostic 2: Signatures modifiers."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: SIGNATURES MODIFIERS")
    print("="*80)
    
    modifiers = discover_entities('modifier', phase='R0')
    
    for mod_info in modifiers:
        print(f"\n--- {mod_info['id']} ---")
        module = mod_info['module']
        func_name = mod_info['function_name']
        func = getattr(module, func_name, None)
        
        if func:
            import inspect
            sig = inspect.signature(func)
            print(f"  Fonction: {func_name}{sig}")
            print(f"  Params: {list(sig.parameters.keys())}")
        else:
            print(f"   Fonction {func_name} INTROUVABLE")


def diagnose_applicability_check():
    """Diagnostic 3: Test logique applicabilité."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: LOGIQUE APPLICABILITÉ")
    print("="*80)
    
    gammas = discover_entities('gamma', phase='R0')
    encodings = discover_entities('encoding', phase='R0')
    
    print(f"\nTotal gammas: {len(gammas)}")
    print(f"Total encodings: {len(encodings)}")
    
    total_pairs = len(gammas) * len(encodings)
    compatible_pairs = 0
    incompatible_pairs = 0
    
    incompatible_list = []
    
    for gamma in gammas:
        gamma_id = gamma['id']
        gamma_metadata = gamma.get('metadata', {})
        d_applicability = gamma_metadata.get('d_applicability', [])
        
        for encoding in encodings:
            encoding_id = encoding['id']
            encoding_prefix = encoding_id.split('-')[0]
            
            # Test compatibilité
            is_compatible = True
            if d_applicability:
                if encoding_prefix not in d_applicability:
                    is_compatible = False
            
            if is_compatible:
                compatible_pairs += 1
            else:
                incompatible_pairs += 1
                incompatible_list.append((gamma_id, encoding_id, d_applicability))
    
    print(f"\nTotal paires possibles: {total_pairs}")
    print(f"Compatibles: {compatible_pairs}")
    print(f"Incompatibles: {incompatible_pairs}")
    
    if incompatible_list:
        print(f"\nDétail incompatibles:")
        for gamma_id, encoding_id, d_app in incompatible_list:
            encoding_prefix = encoding_id.split('-')[0]
            print(f"  {gamma_id} × {encoding_id}")
            print(f"    → d_applicability={d_app}, encoding_prefix={encoding_prefix}")


def diagnose_batch_runner_logic():
    """Diagnostic 4: Logique detect_missing_combinations."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 4: DETECT_MISSING_COMBINATIONS")
    print("="*80)
    
    from prc_automation.batch_runner import detect_missing_combinations, load_execution_registry
    
    registry = load_execution_registry('R0')
    active_entities = {
        'gammas': discover_entities('gamma', 'R0'),
        'encodings': discover_entities('encoding', 'R0'),
        'modifiers': discover_entities('modifier', 'R0'),
    }
    
    print(f"\nRegistry loaded:")
    print(f"  Gammas: {registry['counts']['gammas']}")
    print(f"  Encodings: {registry['counts']['encodings']}")
    print(f"  Modifiers: {registry['counts']['modifiers']}")
    
    print(f"\nActive entities discovered:")
    print(f"  Gammas: {len(active_entities['gammas'])}")
    print(f"  Encodings: {len(active_entities['encodings'])}")
    print(f"  Modifiers: {len(active_entities['modifiers'])}")
    
    # Simuler detect_missing sans DB
    print(f"\nSimulation cartésien (sans filtrage):")
    total_without_filter = (
        len(active_entities['gammas']) *
        len(active_entities['encodings']) *
        len(active_entities['modifiers']) *
        5  # SEEDS
    )
    print(f"  Total brut: {total_without_filter}")
    
    print(f"\nSimulation avec filtrage applicabilité:")
    expected = set()
    skipped = 0
    
    for gamma in active_entities['gammas']:
        gamma_metadata = gamma.get('metadata', {})
        d_applicability = gamma_metadata.get('d_applicability', [])
        
        for encoding in active_entities['encodings']:
            encoding_prefix = encoding['id'].split('-')[0]
            
            if d_applicability and encoding_prefix not in d_applicability:
                skipped += 5  # 5 seeds par paire
                continue
            
            for modifier in active_entities['modifiers']:
                for seed in [42, 123, 456, 789, 1011]:
                    expected.add((
                        gamma['id'],
                        encoding['id'],
                        modifier['id'],
                        seed
                    ))
    
    print(f"  Après filtrage: {len(expected)}")
    print(f"  Skipped: {skipped}")
    print(f"  Différence: {total_without_filter - len(expected)}")


if __name__ == "__main__":
    print("DIAGNOSTIC PHASE 10 - REFACTOR BATCH_RUNNER\n")
    
    try:
        diagnose_discovery()
        diagnose_modifier_signatures()
        diagnose_applicability_check()
        diagnose_batch_runner_logic()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC TERMINÉ")
        print("="*80)
    
    except Exception as e:
        print(f"\n✗ ERREUR DIAGNOSTIC: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)