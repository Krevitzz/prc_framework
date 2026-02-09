
# Script: scripts/r1_2_compensation/run_tests_compensated.py

"""
Applique tests sur séquences compensées.

CONFORMITÉ:
- Réutilise run_tests_sequences.py (R1.1)
- Pattern identique (délégation TestEngine)
"""

import json
from pathlib import Path
from datetime import datetime

from prc_automation.sequence_test_utils import (
    load_sequence_context,
    load_sequence_history,
    load_first_sequence_snapshot,
    store_sequence_test_observation
)
from tests.utilities.HUB.test_engine import TestEngine
from tests.utilities.utils.data_loading import discover_entities, check_applicability

INPUT_LOG = Path("outputs/r1_2_compensation/execution_compensated_log.json")
DB_RESULTS = Path("prc_automation/prc_database/prc_r1_results.db")
PARAMS_CONFIG_ID = 'params_default_v1'
PHASE = 'R1'


def run_tests_compensated():
    """Applique tests sur séquences compensées."""
    
    print(f"\n{'='*70}")
    print(f"APPLICATION TESTS - Séquences Compensées R1.2")
    print(f"{'='*70}\n")
    
    # Charger sequence_exec_ids
    with open(INPUT_LOG) as f:
        log = json.load(f)
    
    sequence_exec_ids = log['sequence_exec_ids']
    
    print(f"Séquences à tester: {len(sequence_exec_ids)}")
    
    # Découvrir tests actifs
    print(f"\n1. Découverte tests...")
    active_tests = discover_entities('test', PHASE)
    print(f"   ✓ {len(active_tests)} tests actifs")
    
    # Initialiser TestEngine
    engine = TestEngine()
    
    # Boucle séquences (identique run_tests_sequences.py)
    print(f"\n2. Application tests...")
    
    total_observations = 0
    errors = 0
    
    for i, sequence_exec_id in enumerate(sequence_exec_ids, 1):
        print(f"\n[{i}/{len(sequence_exec_ids)}] {sequence_exec_id}")
        
        try:
            # Charger contexte
            context = load_sequence_context(sequence_exec_id, PHASE)
            
            print(f"  SEQ: {' → '.join(context['sequence_gammas'])} | {context['d_encoding_id']} | {context['modifier_id']} | s{context['seed']}")
            
            # Charger premier snapshot (state_shape)
            first_snapshot = load_first_sequence_snapshot(sequence_exec_id, PHASE)
            context['state_shape'] = first_snapshot.shape
            
            # Filtrer tests applicables
            applicable_tests = []
            for test_info in active_tests:
                test_module = test_info['module']
                applicable, reason = check_applicability(test_module, context)
                if applicable:
                    applicable_tests.append(test_info)
            
            print(f"  {len(applicable_tests)}/{len(active_tests)} tests applicables")
            
            if not applicable_tests:
                continue
            
            # Charger history
            history = load_sequence_history(sequence_exec_id, PHASE)
            
            # Appliquer tests
            for test_info in applicable_tests:
                test_module = test_info['module']
                
                try:
                    observation = engine.execute_test(
                        test_module, 
                        context, 
                        history, 
                        PARAMS_CONFIG_ID
                    )
                    
                    store_sequence_test_observation(
                        DB_RESULTS, PHASE,
                        sequence_exec_id,
                        context['sequence_gammas'],
                        context['sequence_length'],
                        context['d_encoding_id'],
                        context['modifier_id'],
                        context['seed'],
                        observation
                    )
                    
                    total_observations += 1
                    print(f"    ✓ {test_info['id']}: {observation['status']}")
                
                except Exception as e:
                    errors += 1
                    print(f"    ✗ {test_info['id']}: {e}")
        
        except Exception as e:
            print(f"  ✗ Erreur run: {e}")
            errors += 1
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ APPLICATION TESTS")
    print(f"{'='*70}")
    print(f"Observations: {total_observations}")
    print(f"Erreurs:      {errors}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_tests_compensated()