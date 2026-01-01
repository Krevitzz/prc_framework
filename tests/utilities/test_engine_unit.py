# tests/utilities/test_engine_unit.py
"""
Test unitaire TestEngine avec données synthétiques.

Vérifie que le pipeline complet fonctionne :
1. Chargement config params
2. Validation COMPUTATION_SPECS via RegistryManager
3. Exécution sur history synthétique
4. Génération observations valides

Usage:
    python -m tests.utilities.test_engine_unit
    python -m tests.utilities.test_engine_unit --test SYM-001
"""

import numpy as np
import argparse
import sys
from pathlib import Path

from .test_engine import TestEngine
from .discovery import discover_active_tests


def generate_synthetic_history(test_id: str, num_snapshots: int = 100) -> list:
    """
    Génère history synthétique selon test.
    
    Args:
        test_id: ID du test
        num_snapshots: Nombre d'états
    
    Returns:
        Liste de np.ndarray
    """
    print(f"\n  Génération history synthétique pour {test_id}...")
    
    if test_id == "UNIV-001":
        # Matrice qui croît puis se stabilise
        history = []
        for i in range(num_snapshots):
            if i < 50:
                # Croissance
                state = np.random.randn(10, 10) * (1 + i/10)
            else:
                # Stabilisation
                state = np.random.randn(10, 10) * 5.0
            history.append(state)
        
        print(f"    Histoire UNIV-001 : croissance puis stabilisation")
        return history
    
    elif test_id == "SYM-001":
        # Matrice symétrique qui devient asymétrique
        history = []
        for i in range(num_snapshots):
            base = np.random.randn(10, 10)
            if i < 50:
                # Symétrique
                state = (base + base.T) / 2
            else:
                # Devient asymétrique
                state = base + i/100 * np.triu(base, 1)
            history.append(state)
        
        print(f"    Histoire SYM-001 : symétrique → asymétrique")
        return history
    
    else:
        # Générique : matrice stable
        history = [np.random.randn(10, 10) for _ in range(num_snapshots)]
        print(f"    Histoire générique : bruit stable")
        return history


def create_fake_run_metadata(test_id: str) -> dict:
    """Crée métadonnées fake pour test."""
    return {
        'gamma_id': 'GAM-TEST',
        'd_base_id': 'SYM-TEST' if 'SYM' in test_id else 'UNIV-TEST',
        'modifier_id': 'M0',
        'seed': 42,
        'state_shape': (10, 10),
    }


def validate_observation(observation: dict, test_id: str) -> bool:
    """
    Valide structure observation retournée.
    
    Args:
        observation: Dict retourné par test_engine
        test_id: ID du test
    
    Returns:
        True si valide
    """
    print(f"\n  Validation observation {test_id}...")
    
    # Structure de base
    required_keys = [
        'run_metadata',
        'test_name',
        'test_category',
        'test_version',
        'config_params_id',
        'status',
        'message',
        'statistics',
        'evolution',
        'metadata',
    ]
    
    for key in required_keys:
        if key not in observation:
            print(f"    ✗ Clé manquante : {key}")
            return False
    
    print(f"    ✓ Structure de base OK")
    
    # Status SUCCESS
    if observation['status'] != 'SUCCESS':
        print(f"    ✗ Status : {observation['status']}")
        print(f"    Message : {observation['message']}")
        if 'traceback' in observation:
            print(f"    Traceback complet :")
            print(observation['traceback'])
        return False  
    print(f"    ✓ Status : SUCCESS")
    
    # Statistics non vide
    if not observation['statistics']:
        print(f"    ✗ Statistics vide")
        return False
    
    print(f"    ✓ Statistics : {list(observation['statistics'].keys())}")
    
    # Evolution non vide
    if not observation['evolution']:
        print(f"    ✗ Evolution vide")
        return False
    
    print(f"    ✓ Evolution : {list(observation['evolution'].keys())}")
    
    # Valider une métrique en détail
    first_metric = list(observation['statistics'].keys())[0]
    stats = observation['statistics'][first_metric]
    
    required_stats = ['initial', 'final', 'min', 'max', 'mean', 'std']
    for stat_key in required_stats:
        if stat_key not in stats:
            print(f"    ✗ Statistics.{first_metric} manque : {stat_key}")
            return False
        
        if not np.isfinite(stats[stat_key]):
            print(f"    ✗ Statistics.{first_metric}.{stat_key} non fini : {stats[stat_key]}")
            return False
    
    print(f"    ✓ Statistics.{first_metric} complète")
    
    # Valider evolution
    evol = observation['evolution'][first_metric]
    
    required_evol = ['transition', 'trend', 'slope', 'volatility', 'relative_change']
    for evol_key in required_evol:
        if evol_key not in evol:
            print(f"    ✗ Evolution.{first_metric} manque : {evol_key}")
            return False
    
    print(f"    ✓ Evolution.{first_metric} complète")
    print(f"      - transition : {evol['transition']}")
    print(f"      - trend : {evol['trend']}")
    print(f"      - slope : {evol['slope']:.6f}")
    
    return True


def test_single_test(test_id: str, params_config_id: str = 'params_default_v1'):
    """
    Test complet d'un seul test.
    
    Args:
        test_id: ID du test à tester
        params_config_id: Config params à utiliser
    """
    print("\n" + "="*80)
    print(f"TEST : {test_id}")
    print("="*80)
    
    # 1. Découvrir tests
    print("\n[1/5] Discovery tests...")
    tests = discover_active_tests()
    
    if test_id not in tests:
        print(f"  ✗ Test {test_id} non trouvé")
        print(f"  Tests disponibles : {list(tests.keys())}")
        return False
    
    test_module = tests[test_id]
    print(f"  ✓ Test {test_id} chargé")
    
    # 2. Créer engine
    print("\n[2/5] Initialisation TestEngine...")
    try:
        engine = TestEngine()
        print(f"  ✓ TestEngine initialisé")
    except Exception as e:
        print(f"  ✗ Erreur initialisation : {e}")
        return False
    
    # 3. Générer données synthétiques
    print("\n[3/5] Génération données synthétiques...")
    try:
        history = generate_synthetic_history(test_id, num_snapshots=100)
        run_metadata = create_fake_run_metadata(test_id)
        print(f"  ✓ History : {len(history)} snapshots de shape {history[0].shape}")
    except Exception as e:
        print(f"  ✗ Erreur génération : {e}")
        return False
    
    # 4. Exécuter test
    print(f"\n[4/5] Exécution test avec config '{params_config_id}'...")
    try:
        observation = engine.execute_test(
            test_module=test_module,
            run_metadata=run_metadata,
            history=history,
            params_config_id=params_config_id
        )
        print(f"  ✓ Test exécuté")
    except Exception as e:
        print(f"  ✗ Erreur exécution : {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Valider observation
    print(f"\n[5/5] Validation observation...")
    valid = validate_observation(observation, test_id)
    
    if valid:
        print(f"\n{'='*80}")
        print(f"✓ TEST {test_id} : SUCCÈS")
        print(f"{'='*80}\n")
        return True
    else:
        print(f"\n{'='*80}")
        print(f"✗ TEST {test_id} : ÉCHEC")
        print(f"{'='*80}\n")
        return False


def test_all_tests():
    """Test tous les tests actifs."""
    print("\n" + "="*80)
    print("TEST COMPLET : TOUS LES TESTS")
    print("="*80)
    
    tests = discover_active_tests()
    
    results = {}
    
    for test_id in tests.keys():
        success = test_single_test(test_id)
        results[test_id] = success
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ GLOBAL")
    print("="*80)
    
    for test_id, success in results.items():
        status = "✓ SUCCÈS" if success else "✗ ÉCHEC"
        print(f"  {test_id} : {status}")
    
    all_success = all(results.values())
    
    print("\n" + "="*80)
    if all_success:
        print("✓ TOUS LES TESTS PASSÉS")
    else:
        failed = [tid for tid, s in results.items() if not s]
        print(f"✗ {len(failed)} TEST(S) ÉCHOUÉ(S) : {failed}")
    print("="*80 + "\n")
    
    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Test unitaire TestEngine"
    )
    parser.add_argument(
        '--test',
        help='ID test spécifique (ex: UNIV-001)',
        default=None
    )
    parser.add_argument(
        '--params',
        help='Config params à utiliser',
        default='params_default_v1'
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = test_single_test(args.test, args.params)
    else:
        success = test_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()