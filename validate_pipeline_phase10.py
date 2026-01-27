#!/usr/bin/env python3
"""
validate_pipeline_phase10.py

Mini batch runner de validation SANS écriture DB.
Teste le pipeline Phase 10 étape par étape avec logging détaillé.

Usage:
    python validate_pipeline_phase10.py --mode full
    python validate_pipeline_phase10.py --mode quick --combinations 5
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# ============================================================================
# LOGGING COLORÉ
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_section(title: str):
    """Affiche titre section."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def log_step(step: str):
    """Affiche étape."""
    print(f"{Colors.OKCYAN}▶ {step}{Colors.ENDC}")

def log_success(msg: str):
    """Affiche succès."""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def log_warning(msg: str):
    """Affiche warning."""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def log_error(msg: str):
    """Affiche erreur."""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def log_info(msg: str):
    """Affiche info."""
    print(f"  {msg}")


# ============================================================================
# ÉTAPE 1 : DISCOVERY ENTITÉS
# ============================================================================

def test_discovery(phase: str = 'R0') -> Dict[str, List[Dict]]:
    """
    Teste découverte toutes entités.
    
    Returns:
        {
            'tests': [...],
            'gammas': [...],
            'encodings': [...],
            'modifiers': [...]
        }
    """
    log_section("ÉTAPE 1 : DISCOVERY ENTITÉS")
    
    # Import discovery
    log_step("Import data_loading.discover_entities")
    try:
        from tests.utilities.utils.data_loading import discover_entities, CriticalDiscoveryError
        log_success("Module importé")
    except Exception as e:
        log_error(f"Import failed: {e}")
        sys.exit(1)
    
    entities = {}
    
    # Tests
    log_step(f"Découverte tests (phase={phase})")
    try:
        entities['tests'] = discover_entities('test', phase=phase)
        log_success(f"{len(entities['tests'])} tests découverts")
        for test in entities['tests'][:3]:
            log_info(f"  {test['id']}: {test['module_path']}")
        if len(entities['tests']) > 3:
            log_info(f"  ... et {len(entities['tests']) - 3} autres")
    except CriticalDiscoveryError as e:
        log_error(f"CriticalDiscoveryError: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Erreur discovery tests: {e}")
        sys.exit(1)
    
    # Gammas
    log_step(f"Découverte gammas (phase={phase})")
    try:
        entities['gammas'] = discover_entities('gamma', phase=phase)
        log_success(f"{len(entities['gammas'])} gammas découverts")
        
        # Vérifier d_applicability présent
        missing_applicability = []
        for gamma in entities['gammas']:
            if 'd_applicability' not in gamma['metadata']:
                missing_applicability.append(gamma['id'])
        
        if missing_applicability:
            log_warning(f"d_applicability manquant: {missing_applicability}")
        else:
            log_success("Tous gammas ont d_applicability")
        
        for gamma in entities['gammas'][:3]:
            d_app = gamma['metadata'].get('d_applicability', [])
            log_info(f"  {gamma['id']}: {gamma['module_path']} (applicabilité: {d_app})")
        if len(entities['gammas']) > 3:
            log_info(f"  ... et {len(entities['gammas']) - 3} autres")
    except CriticalDiscoveryError as e:
        log_error(f"CriticalDiscoveryError: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Erreur discovery gammas: {e}")
        sys.exit(1)
    
    # Encodings
    log_step(f"Découverte encodings (phase={phase})")
    try:
        entities['encodings'] = discover_entities('encoding', phase=phase)
        log_success(f"{len(entities['encodings'])} encodings découverts")
        
        # Grouper par type
        by_type = {}
        for enc in entities['encodings']:
            enc_type = enc['id'].split('-')[0]
            by_type.setdefault(enc_type, []).append(enc['id'])
        
        for enc_type, ids in by_type.items():
            log_info(f"  {enc_type}: {len(ids)} encodings")
    except CriticalDiscoveryError as e:
        log_error(f"CriticalDiscoveryError: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Erreur discovery encodings: {e}")
        sys.exit(1)
    
    # Modifiers
    log_step(f"Découverte modifiers (phase={phase})")
    try:
        entities['modifiers'] = discover_entities('modifier', phase=phase)
        log_success(f"{len(entities['modifiers'])} modifiers découverts")
        for mod in entities['modifiers']:
            log_info(f"  {mod['id']}: {mod['module_path']}")
    except CriticalDiscoveryError as e:
        log_error(f"CriticalDiscoveryError: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Erreur discovery modifiers: {e}")
        sys.exit(1)
    
    return entities


# ============================================================================
# ÉTAPE 2 : VALIDATION APPLICABILITÉ
# ============================================================================

def test_applicability(entities: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Teste validation applicabilité gammas × encodings.
    
    Returns:
        {
            'total_combinations': int,
            'valid_combinations': int,
            'invalid_combinations': int,
            'incompatible_pairs': [(gamma_id, encoding_id), ...]
        }
    """
    log_section("ÉTAPE 2 : VALIDATION APPLICABILITÉ")
    
    gammas = entities['gammas']
    encodings = entities['encodings']
    
    log_step(f"Test applicabilité {len(gammas)} gammas × {len(encodings)} encodings")
    
    total = 0
    valid = 0
    invalid = 0
    incompatible_pairs = []
    
    for gamma in gammas:
        gamma_id = gamma['id']
        gamma_metadata = gamma.get('metadata', {})
        d_applicability = gamma_metadata.get('d_applicability', [])
        
        if not d_applicability:
            log_warning(f"{gamma_id}: d_applicability vide (accepte tous)")
        
        for encoding in encodings:
            encoding_id = encoding['id']
            encoding_prefix = encoding_id.split('-')[0]
            
            total += 1
            
            # Vérifier applicabilité
            if d_applicability and encoding_prefix not in d_applicability:
                invalid += 1
                incompatible_pairs.append((gamma_id, encoding_id))
            else:
                valid += 1
    
    log_success(f"Total combinaisons: {total}")
    log_success(f"Valides: {valid} ({100*valid/total:.1f}%)")
    log_info(f"Invalides (skippées): {invalid} ({100*invalid/total:.1f}%)")
    
    if incompatible_pairs[:5]:
        log_info("Exemples paires incompatibles:")
        for gamma_id, encoding_id in incompatible_pairs[:5]:
            log_info(f"  {gamma_id} × {encoding_id}")
        if len(incompatible_pairs) > 5:
            log_info(f"  ... et {len(incompatible_pairs) - 5} autres")
    
    return {
        'total_combinations': total,
        'valid_combinations': valid,
        'invalid_combinations': invalid,
        'incompatible_pairs': incompatible_pairs
    }


# ============================================================================
# ÉTAPE 3 : TEST PREPARE_STATE
# ============================================================================

def test_prepare_state(entities: Dict[str, List[Dict]], n_tests: int = 3) -> Dict[str, Any]:
    """
    Teste prepare_state() avec différentes combinaisons.
    
    Args:
        n_tests: Nombre de combinaisons à tester
    
    Returns:
        {
            'tested': int,
            'succeeded': int,
            'failed': int,
            'errors': [...]
        }
    """
    log_section("ÉTAPE 3 : TEST PREPARE_STATE")
    
    # Import
    log_step("Import prepare_state")
    try:
        from core.state_preparation import prepare_state
        log_success("Fonction importée")
    except Exception as e:
        log_error(f"Import failed: {e}")
        return {'tested': 0, 'succeeded': 0, 'failed': 1, 'errors': [str(e)]}
    
    encodings = entities['encodings']
    modifiers = entities['modifiers']
    
    log_step(f"Test {n_tests} combinaisons encoding × modifier")
    
    tested = 0
    succeeded = 0
    failed = 0
    errors = []
    
    # Sélectionner combinaisons diverses
    test_combinations = [
        (encodings[0], modifiers[0]),  # Premier encoding + M0
        (encodings[5] if len(encodings) > 5 else encodings[0], modifiers[1] if len(modifiers) > 1 else modifiers[0]),
        (encodings[-1], modifiers[-1])  # Dernier encoding + dernier modifier
    ][:n_tests]
    
    for encoding, modifier in test_combinations:
        encoding_id = encoding['id']
        modifier_id = modifier['id']
        
        log_info(f"\nTest {encoding_id} + {modifier_id}")
        tested += 1
        
        try:
            # Extraire fonctions
            encoding_func = getattr(encoding['module'], encoding['function_name'])
            modifier_func = getattr(modifier['module'], modifier['function_name'])
            
            # Préparer params
            encoding_params = {'n_dof': 10}
            modifiers_list = [modifier_func]
            modifier_configs = {modifier_func: {}}
            
            # Appeler prepare_state
            seed = 42
            D_final = prepare_state(
                encoding_func=encoding_func,
                encoding_params=encoding_params,
                modifiers=modifiers_list,
                modifier_configs=modifier_configs,
                seed=seed
            )
            
            # Vérifications
            assert isinstance(D_final, np.ndarray), f"Type incorrect: {type(D_final)}"
            assert D_final.shape[0] == 10, f"Shape incorrecte: {D_final.shape}"
            assert np.all(np.isfinite(D_final)), "Valeurs non finies détectées"
            
            log_success(f"  Shape: {D_final.shape}, dtype: {D_final.dtype}")
            log_info(f"  Min: {np.min(D_final):.4f}, Max: {np.max(D_final):.4f}, Mean: {np.mean(D_final):.4f}")
            
            # Test reproductibilité
            D_final2 = prepare_state(
                encoding_func=encoding_func,
                encoding_params=encoding_params,
                modifiers=modifiers_list,
                modifier_configs=modifier_configs,
                seed=seed
            )
            
            if np.allclose(D_final, D_final2):
                log_success("  Reproductibilité: OK")
            else:
                log_warning("  Reproductibilité: ÉCHEC (seed non fixé ?)")
            
            succeeded += 1
        
        except Exception as e:
            log_error(f"  Erreur: {e}")
            errors.append(f"{encoding_id}+{modifier_id}: {e}")
            failed += 1
    
    log_success(f"\nRésultat: {succeeded}/{tested} succès")
    
    return {
        'tested': tested,
        'succeeded': succeeded,
        'failed': failed,
        'errors': errors
    }


# ============================================================================
# ÉTAPE 4 : TEST GAMMA FACTORIES
# ============================================================================

def test_gamma_factories(entities: Dict[str, List[Dict]], n_tests: int = 5) -> Dict[str, Any]:
    """
    Teste création gammas avec seed param.
    
    Returns:
        {
            'tested': int,
            'succeeded': int,
            'failed': int,
            'missing_seed': [...],
            'errors': [...]
        }
    """
    log_section("ÉTAPE 4 : TEST GAMMA FACTORIES")
    
    gammas = entities['gammas']
    
    log_step(f"Test {min(n_tests, len(gammas))} factories gamma avec seed")
    
    tested = 0
    succeeded = 0
    failed = 0
    missing_seed = []
    errors = []
    
    for gamma in gammas[:n_tests]:
        gamma_id = gamma['id']
        log_info(f"\nTest factory {gamma_id}")
        tested += 1
        
        try:
            # Extraire factory
            factory_name = gamma['function_name']
            factory = getattr(gamma['module'], factory_name)
            
            # Tester avec seed
            try:
                gamma_instance = factory(seed=42)
                log_success(f"  factory(seed=42): OK")
                
                # Vérifier callable
                assert callable(gamma_instance), "Gamma non callable"
                log_success(f"  Instance callable: OK")
                
                # Tester __call__ sur état test
                test_state = np.random.rand(10, 10)
                result = gamma_instance(test_state)
                
                assert isinstance(result, np.ndarray), f"Résultat non ndarray: {type(result)}"
                assert result.shape == test_state.shape, f"Shape changée: {result.shape}"
                assert np.all(np.isfinite(result)), "Valeurs non finies"
                
                log_success(f"  __call__: OK (shape={result.shape})")
                
                succeeded += 1
            
            except TypeError as e:
                if 'seed' in str(e):
                    log_warning(f"  Param seed manquant: {e}")
                    missing_seed.append(gamma_id)
                    failed += 1
                else:
                    raise
        
        except Exception as e:
            log_error(f"  Erreur: {e}")
            errors.append(f"{gamma_id}: {e}")
            failed += 1
    
    log_success(f"\nRésultat: {succeeded}/{tested} succès")
    
    if missing_seed:
        log_warning(f"Gammas SANS seed param: {missing_seed}")
    
    return {
        'tested': tested,
        'succeeded': succeeded,
        'failed': failed,
        'missing_seed': missing_seed,
        'errors': errors
    }


# ============================================================================
# ÉTAPE 5 : TEST RUN_KERNEL
# ============================================================================

def test_run_kernel(entities: Dict[str, List[Dict]], n_tests: int = 2) -> Dict[str, Any]:
    """
    Teste run_kernel() avec combinaisons complètes.
    
    Returns:
        {
            'tested': int,
            'succeeded': int,
            'failed': int,
            'errors': [...]
        }
    """
    log_section("ÉTAPE 5 : TEST RUN_KERNEL (PIPELINE COMPLET)")
    
    # Import
    log_step("Import run_kernel")
    try:
        from core.kernel import run_kernel
        from core.state_preparation import prepare_state
        log_success("Fonctions importées")
    except Exception as e:
        log_error(f"Import failed: {e}")
        return {'tested': 0, 'succeeded': 0, 'failed': 1, 'errors': [str(e)]}
    
    gammas = entities['gammas']
    encodings = entities['encodings']
    modifiers = entities['modifiers']
    
    log_step(f"Test {n_tests} pipelines complets gamma × encoding × modifier")
    
    tested = 0
    succeeded = 0
    failed = 0
    errors = []
    
    # Sélectionner combinaisons valides
    test_combos = []
    for gamma in gammas[:n_tests]:
        gamma_metadata = gamma.get('metadata', {})
        d_applicability = gamma_metadata.get('d_applicability', [])
        
        # Trouver encoding compatible
        for encoding in encodings:
            encoding_prefix = encoding['id'].split('-')[0]
            if not d_applicability or encoding_prefix in d_applicability:
                test_combos.append((gamma, encoding, modifiers[0]))
                break
    
    for gamma, encoding, modifier in test_combos[:n_tests]:
        gamma_id = gamma['id']
        encoding_id = encoding['id']
        modifier_id = modifier['id']
        
        log_info(f"\nTest {gamma_id} × {encoding_id} × {modifier_id}")
        tested += 1
        
        try:
            # 1. Créer D_final
            encoding_func = getattr(encoding['module'], encoding['function_name'])
            modifier_func = getattr(modifier['module'], modifier['function_name'])
            
            D_final = prepare_state(
                encoding_func=encoding_func,
                encoding_params={'n_dof': 10},
                modifiers=[modifier_func],
                modifier_configs={modifier_func: {}},
                seed=42
            )
            log_success(f"  D_final créé: {D_final.shape}")
            
            # 2. Créer gamma
            factory = getattr(gamma['module'], gamma['function_name'])
            
            try:
                gamma_instance = factory(seed=42)
            except TypeError:
                # Fallback si seed manquant
                gamma_instance = factory()
            
            log_success(f"  Gamma créé: {gamma_instance}")
            
            # Reset mémoire si non-markovien
            if hasattr(gamma_instance, 'reset'):
                gamma_instance.reset()
            
            # 3. Run kernel (limité à 50 iterations pour vitesse)
            max_iterations = 50
            snapshots = []
            
            for iteration, state in run_kernel(
                D_final, gamma_instance,
                max_iterations=max_iterations,
                record_history=False
            ):
                if iteration % 10 == 0:
                    snapshots.append({
                        'iteration': iteration,
                        'norm': float(np.linalg.norm(state.flatten())),
                        'min': float(np.min(state)),
                        'max': float(np.max(state))
                    })
                
                # Détection explosion
                if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                    log_warning(f"  Explosion détectée iter={iteration}")
                    break
            
            log_success(f"  Kernel exécuté: {len(snapshots)} snapshots")
            log_info(f"  Norm finale: {snapshots[-1]['norm']:.4f}")
            log_info(f"  Range finale: [{snapshots[-1]['min']:.4f}, {snapshots[-1]['max']:.4f}]")
            
            succeeded += 1
        
        except Exception as e:
            log_error(f"  Erreur: {e}")
            errors.append(f"{gamma_id}×{encoding_id}×{modifier_id}: {e}")
            failed += 1
    
    log_success(f"\nRésultat: {succeeded}/{tested} succès")
    
    return {
        'tested': tested,
        'succeeded': succeeded,
        'failed': failed,
        'errors': errors
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation pipeline Phase 10 (dry-run)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help="Mode validation (quick: 3 tests par étape, full: tous)")
    
    parser.add_argument('--combinations', type=int, default=3,
                       help="Nombre combinaisons à tester en mode quick")
    
    parser.add_argument('--phase', default='R0',
                       help="Phase cible (défaut: R0)")
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}VALIDATION PIPELINE PHASE 10{Colors.ENDC}")
    print(f"Mode: {args.mode}")
    print(f"Phase: {args.phase}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_tests = args.combinations if args.mode == 'quick' else 999
    
    results = {}
    
    # Étape 1 : Discovery
    try:
        entities = test_discovery(phase=args.phase)
        results['discovery'] = 'SUCCESS'
    except Exception as e:
        log_error(f"Discovery échouée: {e}")
        results['discovery'] = 'FAILED'
        sys.exit(1)
    
    # Étape 2 : Applicabilité
    try:
        applicability_results = test_applicability(entities)
        results['applicability'] = 'SUCCESS'
    except Exception as e:
        log_error(f"Applicabilité échouée: {e}")
        results['applicability'] = 'FAILED'
        sys.exit(1)
    
    # Étape 3 : prepare_state
    try:
        prepare_results = test_prepare_state(entities, n_tests=n_tests)
        if prepare_results['succeeded'] == prepare_results['tested']:
            results['prepare_state'] = 'SUCCESS'
        else:
            results['prepare_state'] = 'PARTIAL'
    except Exception as e:
        log_error(f"prepare_state échoué: {e}")
        results['prepare_state'] = 'FAILED'
    
    # Étape 4 : Gamma factories
    try:
        gamma_results = test_gamma_factories(entities, n_tests=n_tests)
        if gamma_results['missing_seed']:
            results['gamma_factories'] = 'PARTIAL'
        elif gamma_results['succeeded'] == gamma_results['tested']:
            results['gamma_factories'] = 'SUCCESS'
        else:
            results['gamma_factories'] = 'FAILED'
    except Exception as e:
        log_error(f"Gamma factories échoués: {e}")
        results['gamma_factories'] = 'FAILED'
    
    # Étape 5 : run_kernel
    try:
        kernel_results = test_run_kernel(entities, n_tests=n_tests)
        if kernel_results['succeeded'] == kernel_results['tested']:
            results['kernel'] = 'SUCCESS'
        else:
            results['kernel'] = 'PARTIAL'
    except Exception as e:
        log_error(f"run_kernel échoué: {e}")
        results['kernel'] = 'FAILED'
    
    # Résumé final
    log_section("RÉSUMÉ VALIDATION")
    
    all_success = all(status == 'SUCCESS' for status in results.values())
    
    for step, status in results.items():
        if status == 'SUCCESS':
            log_success(f"{step}: {status}")
        elif status == 'PARTIAL':
            log_warning(f"{step}: {status}")
        else:
            log_error(f"{step}: {status}")
    
    print()
    if all_success:
        log_success("VALIDATION COMPLÈTE : TOUS TESTS PASSÉS ✓")
        log_info("Pipeline prêt pour exécution batch_runner réelle")
        return 0
    else:
        log_warning("VALIDATION PARTIELLE : Certains tests ont échoué")
        log_info("Vérifier logs ci-dessus pour détails")
        return 1


if __name__ == "__main__":
    sys.exit(main())
