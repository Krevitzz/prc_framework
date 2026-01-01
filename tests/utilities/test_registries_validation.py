#!/usr/bin/env python3
"""
tests/test_registries_validation.py

Script validation complète architecture registres.

Usage:
    python test_registries_validation.py
    python test_registries_validation.py --verbose
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Ajouter path si nécessaire
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utilities.registries.registry_manager import RegistryManager


def test_registry_loading():
    """Test 1: Chargement registres."""
    print("\n" + "="*70)
    print("TEST 1: CHARGEMENT REGISTRES")
    print("="*70)
    
    try:
        rm = RegistryManager()
        print("✓ RegistryManager instancié")
        
        registries = list(rm.registries.keys())
        print(f"✓ {len(registries)} registres chargés:")
        for reg in sorted(registries):
            num_funcs = len(rm.registries[reg].list_functions())
            print(f"  - {reg}: {num_funcs} fonctions")
        
        expected = ['algebra', 'spectral', 'statistical', 'spatial', 'pattern', 'topological', 'graph']
        missing = set(expected) - set(registries)
        
        if missing:
            print(f"\n⚠ Registres manquants: {missing}")
            return False
        
        print("\n✓ Tous registres attendus présents")
        return True
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_retrieval():
    """Test 2: Récupération fonctions."""
    print("\n" + "="*70)
    print("TEST 2: RÉCUPÉRATION FONCTIONS")
    print("="*70)
    
    rm = RegistryManager()
    
    test_cases = [
        "algebra.matrix_norm",
        "algebra.frobenius_norm",
        "spectral.eigenvalue_max",
        "statistical.entropy",
        "spatial.gradient_magnitude",
        "pattern.periodicity",
        "topological.connected_components",
        "graph.density",
    ]
    
    success = True
    for key in test_cases:
        try:
            func = rm.get_function(key)
            print(f"✓ {key}")
        except Exception as e:
            print(f"❌ {key}: {e}")
            success = False
    
    return success


def test_function_execution():
    """Test 3: Exécution fonctions."""
    print("\n" + "="*70)
    print("TEST 3: EXÉCUTION FONCTIONS")
    print("="*70)
    
    rm = RegistryManager()
    
    # Créer états test
    state_2d_sym = np.eye(10)
    state_2d_asym = np.random.randn(10, 10)
    state_3d = np.random.randn(5, 5, 5)
    
    test_cases = [
        # (registry_key, state, params, expected_type)
        ("algebra.matrix_norm", state_2d_sym, {'norm_type': 'frobenius'}, float),
        ("algebra.matrix_asymmetry", state_2d_asym, {'norm_type': 'frobenius', 'normalize': True}, float),
        ("algebra.trace_value", state_2d_sym, {}, float),
        ("algebra.determinant_value", state_2d_sym, {}, float),
        
        ("spectral.eigenvalue_max", state_2d_sym, {'absolute': True}, float),
        ("spectral.spectral_radius", state_2d_sym, {}, float),
        
        ("statistical.entropy", state_2d_asym, {'bins': 50}, float),
        ("statistical.variance", state_2d_asym, {'normalize': False}, float),
        ("statistical.sparsity", state_2d_asym, {'threshold': 1e-6}, float),
        
        ("spatial.gradient_magnitude", state_2d_asym, {}, float),
        ("spatial.laplacian_energy", state_2d_asym, {}, float),
        
        ("pattern.diversity", state_2d_asym, {'bins': 50}, float),
        ("pattern.uniformity", state_2d_asym, {'bins': 50}, float),
        
        ("topological.connected_components", state_2d_asym, {'threshold': 0.0}, float),
        ("topological.euler_characteristic", state_2d_asym, {'threshold': 0.0}, float),
        
        ("graph.density", state_2d_asym, {'threshold': 0.5}, float),
        ("graph.degree_variance", state_2d_asym, {'threshold': 0.5}, float),
    ]
    
    success = True
    for key, state, params, expected_type in test_cases:
        try:
            func = rm.get_function(key)
            result = func(state, **params)
            
            if not isinstance(result, expected_type):
                print(f"❌ {key}: Type incorrect (attendu {expected_type}, reçu {type(result)})")
                success = False
                continue
            
            if not np.isfinite(result):
                print(f"⚠ {key}: Valeur non finie ({result})")
            
            print(f"✓ {key}: {result:.6f}")
        
        except Exception as e:
            print(f"❌ {key}: {e}")
            success = False
    
    return success


def test_computation_spec_validation():
    """Test 4: Validation COMPUTATION_SPECS."""
    print("\n" + "="*70)
    print("TEST 4: VALIDATION COMPUTATION_SPECS")
    print("="*70)
    
    rm = RegistryManager()
    
    test_specs = [
        {
            'registry_key': 'algebra.matrix_norm',
            'default_params': {'norm_type': 'frobenius'},
        },
        {
            'registry_key': 'spectral.eigenvalue_max',
            'default_params': {'absolute': True},
            'post_process': 'round_4',
        },
        {
            'registry_key': 'statistical.entropy',
            'default_params': {'bins': 50, 'normalize': True},
            'post_process': 'log',
        },
    ]
    
    success = True
    for i, spec in enumerate(test_specs):
        try:
            validated = rm.validate_computation_spec(spec)
            
            assert 'function' in validated
            assert 'params' in validated
            assert 'post_process' in validated
            assert 'registry_key' in validated
            
            print(f"✓ Spec {i+1}: {spec['registry_key']}")
        
        except Exception as e:
            print(f"❌ Spec {i+1}: {e}")
            success = False
    
    return success


def test_error_handling():
    """Test 5: Gestion erreurs."""
    print("\n" + "="*70)
    print("TEST 5: GESTION ERREURS")
    print("="*70)
    
    rm = RegistryManager()
    
    # Test fonction inexistante
    try:
        rm.get_function('nonexistent.function')
        print("❌ Devrait lever KeyError pour fonction inexistante")
        return False
    except KeyError as e:
        print(f"✓ KeyError correcte: {e}")
    
    # Test registre inexistant
    try:
        rm.get_function('invalid_registry.function')
        print("❌ Devrait lever KeyError pour registre inexistant")
        return False
    except KeyError as e:
        print(f"✓ KeyError correcte: {e}")
    
    # Test paramètres invalides
    try:
        spec = {
            'registry_key': 'algebra.matrix_norm',
            'default_params': {'invalid_param': 123},
        }
        rm.validate_computation_spec(spec)
        print("❌ Devrait lever ValueError pour paramètre invalide")
        return False
    except ValueError as e:
        print(f"✓ ValueError correcte: {e}")
    
    # Test état invalide
    try:
        func = rm.get_function('algebra.matrix_asymmetry')
        state_1d = np.array([1, 2, 3])  # Devrait être 2D
        func(state_1d)
        print("❌ Devrait lever ValueError pour état 1D")
        return False
    except ValueError as e:
        print(f"✓ ValueError correcte: {e}")
    
    print("\n✓ Toutes erreurs gérées correctement")
    return True


def test_post_processors():
    """Test 6: Post-processors."""
    print("\n" + "="*70)
    print("TEST 6: POST-PROCESSORS")
    print("="*70)
    
    from tests.utilities.registries.post_processors import POST_PROCESSORS, get_post_processor
    
    test_values = [
        (3.14159265, 'round_2', 3.14),
        (3.14159265, 'round_4', 3.1416),
        (-5.0, 'abs', 5.0),
        (0.5, 'clip_01', 0.5),
        (1.5, 'clip_01', 1.0),
    ]
    
    success = True
    for value, key, expected in test_values:
        try:
            processor = get_post_processor(key)
            result = processor(value)
            
            if abs(result - expected) > 1e-6:
                print(f"❌ {key}({value}): Attendu {expected}, reçu {result}")
                success = False
            else:
                print(f"✓ {key}({value}) = {result}")
        
        except Exception as e:
            print(f"❌ {key}: {e}")
            success = False
    
    print(f"\n✓ {len(POST_PROCESSORS)} post-processors disponibles")
    return success


def test_complete_workflow():
    """Test 7: Workflow complet."""
    print("\n" + "="*70)
    print("TEST 7: WORKFLOW COMPLET")
    print("="*70)
    
    rm = RegistryManager()
    
    # Simuler COMPUTATION_SPECS d'un test
    computation_specs = {
        'frobenius_norm': {
            'registry_key': 'algebra.frobenius_norm',
            'default_params': {},
            'post_process': 'round_4',
        },
        'asymmetry': {
            'registry_key': 'algebra.matrix_asymmetry',
            'default_params': {'norm_type': 'frobenius', 'normalize': True},
            'post_process': 'round_6',
        },
        'entropy': {
            'registry_key': 'statistical.entropy',
            'default_params': {'bins': 50, 'normalize': True},
            'post_process': 'round_4',
        },
    }
    
    # Valider specs
    prepared = {}
    for name, spec in computation_specs.items():
        try:
            validated = rm.validate_computation_spec(spec)
            prepared[name] = validated
            print(f"✓ Spec validée: {name}")
        except Exception as e:
            print(f"❌ Échec validation {name}: {e}")
            return False
    
    # Créer états test
    history = [np.random.randn(10, 10) for _ in range(10)]
    
    # Simuler calcul sur history
    results = {name: [] for name in prepared.keys()}
    
    for iteration, state in enumerate(history):
        for name, computation in prepared.items():
            try:
                func = computation['function']
                params = computation['params']
                post_proc = computation['post_process']
                
                # Calculer
                value = func(state, **params)
                
                # Post-process
                if post_proc:
                    value = post_proc(value)
                
                results[name].append(value)
            
            except Exception as e:
                print(f"❌ Erreur iteration {iteration}, métrique {name}: {e}")
                return False
    
    # Vérifier résultats
    print(f"\n✓ {len(history)} itérations traitées")
    for name, values in results.items():
        print(f"  {name}: {len(values)} valeurs, range [{min(values):.4f}, {max(values):.4f}]")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# VALIDATION ARCHITECTURE REGISTRES")
    print("#"*70)
    
    tests = [
        ("Chargement registres", test_registry_loading),
        ("Récupération fonctions", test_function_retrieval),
        ("Exécution fonctions", test_function_execution),
        ("Validation specs", test_computation_spec_validation),
        ("Gestion erreurs", test_error_handling),
        ("Post-processors", test_post_processors),
        ("Workflow complet", test_complete_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' a planté: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Résumé
    print("\n" + "#"*70)
    print("# RÉSUMÉ")
    print("#"*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n{passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS PASSENT")
        return 0
    else:
        print(f"\n⚠ {total - passed} tests ont échoué")
        return 1


if __name__ == "__main__":
    sys.exit(main())