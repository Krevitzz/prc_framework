
# Script: scripts/r1_execution/run_tests_sequences.py

"""
Applique tests R0 sur séquences R1.1 exécutées.

CONFORMITÉ:
- Réutilise TestEngine (pas duplication)
- Pattern similaire batch_runner.run_batch_test()
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

# Configuration
INPUT_LOG_N2 = Path("outputs/r1_execution/execution_n2_log.json")
INPUT_LOG_N3 = Path("outputs/r1_execution/execution_n3_log.json")
DB_RESULTS = Path("prc_automation/prc_database/prc_r1_results.db")
PARAMS_CONFIG_ID = 'params_default_v1'
PHASE = 'R1'


def run_tests_sequences():
    """
    Applique tests sur séquences exécutées.
    
    WORKFLOW:
    1. Charger sequence_exec_ids (logs n=2, n=3)
    2. Découvrir tests actifs
    3. Boucle sur séquences:
       a. Charger contexte + history
       b. Filtrer tests applicables
       c. Exécuter tests (TestEngine)
       d. Stocker observations
    """
    
    print(f"\n{'='*70}")
    print(f"APPLICATION TESTS - Séquences R1.1")
    print(f"{'='*70}\n")
    
    # 1. Charger sequence_exec_ids
    with open(INPUT_LOG_N2) as f:
        log_n2 = json.load(f)
    sequence_exec_ids_n2 = log_n2['sequence_exec_ids']
    
    with open(INPUT_LOG_N3) as f:
        log_n3 = json.load(f)
    sequence_exec_ids_n3 = log_n3['sequence_exec_ids']
    
    all_sequence_exec_ids = sequence_exec_ids_n2 + sequence_exec_ids_n3
    
    print(f"Séquences à tester:")
    print(f"  n=2: {len(sequence_exec_ids_n2)}")
    print(f"  n=3: {len(sequence_exec_ids_n3)}")
    print(f"  Total: {len(all_sequence_exec_ids)}")
    
    # 2. Découvrir tests actifs
    print(f"\n1. Découverte tests...")
    active_tests = discover_entities('test', PHASE)
    print(f"   ✓ {len(active_tests)} tests actifs")
    
    # 3. Initialiser TestEngine
    engine = TestEngine()
    
    # 4. Boucle séquences
    print(f"\n2. Application tests...")
    
    total_observations = 0
    errors = 0
    
    for i, sequence_exec_id in enumerate(all_sequence_exec_ids, 1):
        print(f"\n[{i}/{len(all_sequence_exec_ids)}] {sequence_exec_id}")
        
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
            
            # Charger history (états finaux chaque gamma)
            history = load_sequence_history(sequence_exec_id, PHASE)
            
            # Appliquer tests
            for test_info in applicable_tests:
                test_module = test_info['module']
                
                try:
                    # Exécuter test (délégation TestEngine)
                    observation = engine.execute_test(
                        test_module, 
                        context, 
                        history, 
                        PARAMS_CONFIG_ID
                    )
                    
                    # Stocker observation
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
    run_tests_sequences()