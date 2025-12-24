#!/usr/bin/env python3
"""
validate_phase0.py

Script de validation avant exécution Phase 0.

Vérifie:
1. Structure des fichiers
2. Imports corrects
3. Opérateurs fonctionnels
4. Tests modulaires fonctionnels
5. Encodings D disponibles
6. TM-GAM-001 exécutable

Usage:
    python validate_phase0.py
    python validate_phase0.py --verbose
    python validate_phase0.py --quick  # Validation rapide
"""

import sys
import importlib
from pathlib import Path


class Colors:
    """Codes couleur ANSI pour terminal."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    """Affiche header formaté."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_success(msg):
    """Affiche message succès."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg):
    """Affiche message erreur."""
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_warning(msg):
    """Affiche warning."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_info(msg):
    """Affiche info."""
    print(f"  {msg}")


# ============================================================================
# VALIDATION 1 : STRUCTURE FICHIERS
# ============================================================================

def validate_file_structure(verbose=False):
    """Vérifie que tous les fichiers nécessaires existent."""
    print_header("VALIDATION 1 : Structure des Fichiers")
    
    required_files = {
        'core': [
            'core/__init__.py',
            'core/kernel.py',
            'core/state_preparation.py',
        ],
        'encodings': [
            'D_encodings/__init__.py',
            'D_encodings/rank2_symmetric.py',
            'D_encodings/rank2_asymmetric.py',
            'D_encodings/rank3_correlations.py',
            'D_encodings/verification_tests.py',
        ],
        'modifiers': [
            'modifiers/__init__.py',
            'modifiers/noise.py',
        ],
        'operators': [
            'operators/__init__.py',
            'operators/gamma_hyp_001.py',
        ],
        'tests_utilities': [
            'tests/utilities/__init__.py',
            'tests/utilities/test_symmetry.py',
            'tests/utilities/test_norm.py',
            'tests/utilities/test_diversity.py',
            'tests/utilities/test_convergence.py',
        ],
        'tests': [
            'tests/TM-GAM-001.py',
        ],
    }
    
    all_ok = True
    for category, files in required_files.items():
        if verbose:
            print(f"\n{category.upper()}:")
        
        category_ok = True
        for filepath in files:
            path = Path(filepath)
            if path.exists():
                if verbose:
                    print_success(f"{filepath}")
            else:
                print_error(f"{filepath} MANQUANT")
                category_ok = False
                all_ok = False
        
        if not verbose:
            if category_ok:
                print_success(f"{category}: {len(files)} fichiers OK")
            else:
                print_error(f"{category}: Fichiers manquants")
    
    return all_ok


# ============================================================================
# VALIDATION 2 : IMPORTS
# ============================================================================

def validate_imports(verbose=False):
    """Vérifie que tous les imports fonctionnent."""
    print_header("VALIDATION 2 : Imports Python")
    
    imports_to_test = {
        'core': [
            ('core.kernel', 'run_kernel'),
            ('core.state_preparation', 'prepare_state'),
        ],
        'encodings': [
            ('D_encodings', 'create_identity'),
            ('D_encodings', 'create_random_uniform'),
            ('D_encodings', 'create_random_asymmetric'),
            ('D_encodings', 'create_random_rank3'),
        ],
        'modifiers': [
            ('modifiers.noise', 'add_gaussian_noise'),
            ('modifiers.noise', 'add_uniform_noise'),
        ],
        'operators': [
            ('operators', 'PureSaturationGamma'),
            ('operators', 'get_operator_by_id'),
        ],
        'tests': [
            ('tests.utilities', 'test_symmetry_preservation'),
            ('tests.utilities', 'test_norm_evolution'),
            ('tests.utilities', 'test_diversity_preservation'),
            ('tests.utilities', 'test_convergence_to_fixed_point'),
        ],
    }
    
    all_ok = True
    for category, imports in imports_to_test.items():
        if verbose:
            print(f"\n{category.upper()}:")
        
        category_ok = True
        for module_name, obj_name in imports:
            try:
                module = importlib.import_module(module_name)
                obj = getattr(module, obj_name)
                if verbose:
                    print_success(f"from {module_name} import {obj_name}")
            except ImportError as e:
                print_error(f"Import {module_name}.{obj_name}: {str(e)}")
                category_ok = False
                all_ok = False
            except AttributeError as e:
                print_error(f"Attribut {obj_name} manquant dans {module_name}")
                category_ok = False
                all_ok = False
        
        if not verbose:
            if category_ok:
                print_success(f"{category}: {len(imports)} imports OK")
            else:
                print_error(f"{category}: Erreurs d'import")
    
    return all_ok


# ============================================================================
# VALIDATION 3 : OPÉRATEURS
# ============================================================================

def validate_operators(verbose=False):
    """Teste les opérateurs Γ."""
    print_header("VALIDATION 3 : Opérateurs Γ")
    
    try:
        import numpy as np
        from operators import PureSaturationGamma, validate_operator
        
        # Test GAM-001
        if verbose:
            print("\nTest GAM-001 (PureSaturationGamma):")
        
        gamma = PureSaturationGamma(beta=2.0)
        test_state = np.random.randn(10, 10)
        
        try:
            result = gamma(test_state)
            
            # Vérifications
            assert isinstance(result, np.ndarray), "Result must be np.ndarray"
            assert result.shape == test_state.shape, "Shape must be preserved"
            assert not np.any(np.isnan(result)), "No NaN"
            assert not np.any(np.isinf(result)), "No Inf"
            
            print_success("GAM-001 : Opérateur fonctionnel")
            if verbose:
                print_info(f"  Input shape: {test_state.shape}")
                print_info(f"  Output shape: {result.shape}")
                print_info(f"  Output range: [{np.min(result):.3f}, {np.max(result):.3f}]")
            
            return True
        
        except Exception as e:
            print_error(f"GAM-001 : Erreur d'exécution - {str(e)}")
            return False
    
    except Exception as e:
        print_error(f"Impossible de charger opérateurs: {str(e)}")
        return False


# ============================================================================
# VALIDATION 4 : TESTS MODULAIRES
# ============================================================================

def validate_tests(verbose=False):
    """Teste les tests modulaires."""
    print_header("VALIDATION 4 : Tests Modulaires")
    
    try:
        import numpy as np
        from tests.utilities import (
            test_symmetry_preservation,
            test_norm_evolution,
            test_diversity_preservation,
            test_convergence_to_fixed_point,
        )
        
        # Créer historique de test
        history = [np.eye(10) + 0.01 * np.random.randn(10, 10) for _ in range(10)]
        
        tests_to_run = [
            ("TEST-SYM-001", test_symmetry_preservation),
            ("TEST-UNIV-001", test_norm_evolution),
            ("TEST-UNIV-002", test_diversity_preservation),
            ("TEST-UNIV-003", test_convergence_to_fixed_point),
        ]
        
        all_ok = True
        for test_name, test_func in tests_to_run:
            try:
                result = test_func(history)
                
                # Vérifier que result a les attributs attendus
                assert hasattr(result, 'status'), f"{test_name}: missing status"
                assert hasattr(result, 'message'), f"{test_name}: missing message"
                
                if verbose:
                    print_success(f"{test_name}: {result.status} - {result.message}")
                else:
                    print_success(f"{test_name}: Fonctionnel")
            
            except Exception as e:
                print_error(f"{test_name}: Erreur - {str(e)}")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print_error(f"Impossible de charger tests: {str(e)}")
        return False


# ============================================================================
# VALIDATION 5 : ENCODINGS D
# ============================================================================

def validate_encodings(verbose=False):
    """Teste les générateurs D^(base)."""
    print_header("VALIDATION 5 : Encodings D^(base)")
    
    try:
        from D_encodings import (
            create_identity,
            create_random_uniform,
            create_random_asymmetric,
            run_verification_suite,
        )
        
        generators_to_test = [
            ("SYM-001", create_identity, {'n_dof': 20}),
            ("SYM-002", create_random_uniform, {'n_dof': 20, 'seed': 42}),
            ("ASY-001", create_random_asymmetric, {'n_dof': 20, 'seed': 42}),
        ]
        
        all_ok = True
        for gen_id, gen_func, params in generators_to_test:
            try:
                # Générer
                D = gen_func(**params)
                
                # Vérifications basiques
                assert D is not None, "Generator returned None"
                assert D.shape[0] == params['n_dof'], "Wrong shape"
                
                if verbose:
                    print_success(f"{gen_id}: shape={D.shape}, range=[{D.min():.2f}, {D.max():.2f}]")
                else:
                    print_success(f"{gen_id}: OK")
            
            except Exception as e:
                print_error(f"{gen_id}: Erreur - {str(e)}")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print_error(f"Impossible de charger encodings: {str(e)}")
        return False


# ============================================================================
# VALIDATION 6 : TM-GAM-001
# ============================================================================

def validate_tm_gam001(verbose=False, quick=False):
    """Teste TM-GAM-001 (exécution minimale)."""
    print_header("VALIDATION 6 : TM-GAM-001")
    
    if quick:
        print_warning("Mode quick : Validation import uniquement")
        try:
            # Juste importer
            sys.path.insert(0, str(Path('tests')))
            import TM_GAM_001
            print_success("TM-GAM-001 : Import OK")
            return True
        except Exception as e:
            print_error(f"TM-GAM-001 : Import échoué - {str(e)}")
            return False
    
    print_info("Exécution run de test (peut prendre 10-20 secondes)...")
    
    try:
        # Exécuter 1 run minimal
        import numpy as np
        from core.kernel import run_kernel
        from core.state_preparation import prepare_state
        from operators import PureSaturationGamma
        from D_encodings import create_identity
        from tests.utilities import test_norm_evolution, test_diversity_preservation
        
        # Config minimale
        D_base = create_identity(n_dof=20)  # Petit pour rapidité
        D = prepare_state(D_base, [])
        gamma = PureSaturationGamma(beta=2.0)
        
        # Exécuter 50 itérations
        history = []
        for i, state in run_kernel(D, gamma, max_iterations=50):
            if i % 5 == 0:
                history.append(state.copy())
        
        # Tests rapides
        result_norm = test_norm_evolution(history)
        result_div = test_diversity_preservation(history)
        
        print_success("TM-GAM-001 : Exécution complète réussie")
        if verbose:
            print_info(f"  Iterations: 50")
            print_info(f"  Snapshots: {len(history)}")
            print_info(f"  UNIV-001: {result_norm.status}")
            print_info(f"  UNIV-002: {result_div.status}")
        
        return True
    
    except Exception as e:
        print_error(f"TM-GAM-001 : Exécution échouée - {str(e)}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation Phase 0")
    parser.add_argument('--verbose', action='store_true', help='Affichage détaillé')
    parser.add_argument('--quick', action='store_true', help='Validation rapide (sans TM run)')
    parser.add_argument('--skip-tm', action='store_true', help='Sauter validation TM-GAM-001')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'#'*70}")
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "VALIDATION PHASE 0" + " "*30 + "#")
    print("#" + " "*68 + "#")
    print(f"{'#'*70}{Colors.RESET}\n")
    
    # Exécuter validations
    results = {}
    
    results['structure'] = validate_file_structure(verbose=args.verbose)
    results['imports'] = validate_imports(verbose=args.verbose)
    results['operators'] = validate_operators(verbose=args.verbose)
    results['tests'] = validate_tests(verbose=args.verbose)
    results['encodings'] = validate_encodings(verbose=args.verbose)
    
    if not args.skip_tm:
        results['tm'] = validate_tm_gam001(verbose=args.verbose, quick=args.quick)
    
    # Rapport final
    print_header("RAPPORT FINAL")
    
    for category, success in results.items():
        if success:
            print_success(f"{category.upper()}: OK")
        else:
            print_error(f"{category.upper()}: ÉCHEC")
    
    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)
    
    print(f"\n{Colors.BOLD}{'─'*70}")
    print(f"Résultat: {n_success}/{n_total} validations réussies")
    print(f"{'─'*70}{Colors.RESET}\n")
    
    if n_success == n_total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ PHASE 0 PRÊTE POUR EXÉCUTION{Colors.RESET}")
        print("\nCommandes suivantes:")
        print(f"  {Colors.BLUE}python tests/TM-GAM-001.py --single{Colors.RESET}  # Test rapide 1 run")
        print(f"  {Colors.BLUE}python tests/TM-GAM-001.py --phase 0{Colors.RESET}  # Phase 0 complète (36 runs)")
        print()
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ VALIDATION ÉCHOUÉE{Colors.RESET}")
        print("\nCorrections nécessaires avant exécution Phase 0")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())