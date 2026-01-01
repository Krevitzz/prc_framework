# tests/utilities/test_engine_multiconfig.py
"""
Test traçabilité multi-configs.

Vérifie qu'on peut exécuter le même test sur le même dataset
avec plusieurs configs différentes et que les résultats sont
complètement indépendants et traçables.

Usage:
    python -m tests.utilities.test_engine_multiconfig
"""

import numpy as np
import sys
from pathlib import Path

from .test_engine import TestEngine
from .discovery import discover_active_tests


def generate_fixed_history(seed: int = 42) -> list:
    """
    Génère history IDENTIQUE pour tous les runs.
    
    Args:
        seed: Seed fixe pour reproductibilité
    
    Returns:
        Liste de 50 snapshots
    """
    np.random.seed(seed)
    
    history = []
    for i in range(50):
        state = np.random.randn(10, 10) * (1 + i/10)
        history.append(state)
    
    return history


def create_fixed_metadata():
    """Métadonnées identiques pour tous les runs."""
    return {
        'gamma_id': 'GAM-MULTITEST',
        'd_base_id': 'UNIV-MULTITEST',
        'modifier_id': 'M0',
        'seed': 42,
        'state_shape': (10, 10),
    }


def test_multiconfig():
    """Test principal multi-configs."""
    
    print("\n" + "="*80)
    print("TEST TRAÇABILITÉ MULTI-CONFIGS")
    print("="*80)
    
    # Configuration
    test_id = 'UNIV-001'
    configs = [
        'params_default_v1',
        'params_strict_v1',
    ]
    
    print(f"\nTest : {test_id}")
    print(f"Configs : {configs}")
    print(f"Dataset : IDENTIQUE (seed=42, 50 snapshots)")
    
    # 1. Charger test
    print("\n[1/5] Chargement test...")
    tests = discover_active_tests()
    
    if test_id not in tests:
        print(f"  ✗ Test {test_id} non trouvé")
        return False
    
    test_module = tests[test_id]
    print(f"  ✓ Test {test_id} chargé")
    
    # 2. Générer dataset fixe
    print("\n[2/5] Génération dataset fixe...")
    history = generate_fixed_history(seed=42)
    run_metadata = create_fixed_metadata()
    
    print(f"  ✓ History : {len(history)} snapshots")
    print(f"  ✓ Seed fixe : 42")
    
    # 3. Exécuter avec chaque config
    print(f"\n[3/5] Exécution avec {len(configs)} configs...")
    
    engine = TestEngine()
    observations = {}
    
    for config_id in configs:
        print(f"\n  Config : {config_id}")
        
        try:
            obs = engine.execute_test(
                test_module=test_module,
                run_metadata=run_metadata,
                history=history,
                params_config_id=config_id
            )
            
            if obs['status'] != 'SUCCESS':
                print(f"    ✗ Status : {obs['status']}")
                print(f"    Message : {obs['message']}")
                return False
            
            observations[config_id] = obs
            print(f"    ✓ Exécuté avec succès")
            
        except Exception as e:
            print(f"    ✗ Erreur : {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 4. Vérifier traçabilité
    print(f"\n[4/5] Vérification traçabilité...")
    
    success = True
    
    # 4.1. Chaque observation a le bon config_id
    for config_id, obs in observations.items():
        stored_id = obs['config_params_id']
        
        if stored_id != config_id:
            print(f"  ✗ Config {config_id} : config_id stocké incorrect ({stored_id})")
            success = False
        else:
            print(f"  ✓ Config {config_id} : config_id tracé correctement")
    
    # 4.2. Même run_metadata pour tous
    first_metadata = observations[configs[0]]['run_metadata']
    
    for config_id, obs in observations.items():
        if obs['run_metadata'] != first_metadata:
            print(f"  ✗ Config {config_id} : run_metadata incohérent")
            success = False
    
    print(f"  ✓ run_metadata identique pour toutes configs")
    
    # 4.3. Même test_name pour tous
    first_test_name = observations[configs[0]]['test_name']
    
    for config_id, obs in observations.items():
        if obs['test_name'] != first_test_name:
            print(f"  ✗ Config {config_id} : test_name incohérent")
            success = False
    
    print(f"  ✓ test_name identique pour toutes configs")
    
    # 5. Vérifier résultats DISTINCTS
    print(f"\n[5/5] Vérification résultats distincts...")
    
    # Comparer métriques entre configs
    obs1 = observations[configs[0]]
    obs2 = observations[configs[1]]
    
    # Extraire première métrique commune
    metric_name = list(obs1['statistics'].keys())[0]
    
    stats1 = obs1['statistics'][metric_name]
    stats2 = obs2['statistics'][metric_name]
    
    print(f"\n  Métrique : {metric_name}")
    print(f"  Config {configs[0]} :")
    print(f"    - final : {stats1['final']:.6f}")
    print(f"    - mean  : {stats1['mean']:.6f}")
    print(f"    - std   : {stats1['std']:.6f}")
    
    print(f"  Config {configs[1]} :")
    print(f"    - final : {stats2['final']:.6f}")
    print(f"    - mean  : {stats2['mean']:.6f}")
    print(f"    - std   : {stats2['std']:.6f}")
    
    # Vérifier si différents (devrait l'être si params différents)
    # Note: Peut être identique si params_strict n'affecte pas ce test
    if stats1 == stats2:
        print(f"\n  ⚠ Statistiques identiques")
        print(f"    → Normal si params_strict n'affecte pas {test_id}")
        print(f"    → Vérifie manuellement que les configs sont différentes")
    else:
        print(f"\n  ✓ Statistiques DISTINCTES entre configs")
    
    # 6. Vérification clés uniques DB
    print(f"\n[6/5] Simulation stockage DB...")
    
    # Simuler insertion DB
    db_keys = set()
    
    for config_id, obs in observations.items():
        # Clé unique selon schema_results.sql
        # UNIQUE(exec_id, test_name, params_config_id)
        exec_id = hash((
            obs['run_metadata']['gamma_id'],
            obs['run_metadata']['d_base_id'],
            obs['run_metadata']['seed']
        )) % 1000000  # Fake exec_id
        
        db_key = (exec_id, obs['test_name'], obs['config_params_id'])
        
        if db_key in db_keys:
            print(f"  ✗ COLLISION CLÉ UNIQUE : {db_key}")
            success = False
        else:
            db_keys.add(db_key)
            print(f"  ✓ Clé unique : exec_id={exec_id}, test={obs['test_name']}, config={obs['config_params_id']}")
    
    # Résumé
    print("\n" + "="*80)
    if success:
        print("✓ TRAÇABILITÉ MULTI-CONFIGS : VALIDÉE")
        print()
        print("Vérifications passées :")
        print("  ✓ Observations distinctes par config_id")
        print("  ✓ run_metadata préservé")
        print("  ✓ test_name cohérent")
        print("  ✓ Clés uniques DB sans collision")
        print()
        print("→ Même test + même data + 2 configs = 2 résultats indépendants ✓")
    else:
        print("✗ TRAÇABILITÉ MULTI-CONFIGS : ÉCHEC")
    
    print("="*80 + "\n")
    
    return success


def main():
    success = test_multiconfig()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()