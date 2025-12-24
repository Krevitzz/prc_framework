"""
tests/TM-GAM-001.py

TOY MODEL: TM-GAM-001
HYPOTHESIS: HYP-GAM-001 (Saturation pure pointwise)

CONFIGURATION:
- Γ: PureSaturationGamma(β)
- β ∈ {0.5, 1.0, 2.0, 5.0} (grille complète Phase 2)
- D_bases: Tous (SYM, ASY, R3) = 13 bases
- Modifiers: M0, M1, M2, M3
- Seeds: 1..5

Total runs = 4 params × 13 D × 4 M × 5 seeds = 1040 exécutions

Pour Phase 0 (calibration), on exécute sous-grille:
- β ∈ {1.0, 2.0}
- D: SYM-001, SYM-002, ASY-001
- M: M0, M1
- Seeds: 1..3
Total: 2 × 3 × 2 × 3 = 36 runs

EXPECTED:
- Convergence monotone vers sign(D_ij)
- Possible trivialité (convergence vers signes)
- Test de pipeline complet
"""

import numpy as np
import sys
from pathlib import Path

# Ajouter parent au path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kernel import run_kernel
from core.state_preparation import prepare_state
from operators.gamma_hyp_001 import PureSaturationGamma
from modifiers.noise import add_gaussian_noise, add_uniform_noise

# D^(base) generators
from D_encodings.rank2_symmetric import (
    create_identity,
    create_random_uniform,
    create_random_gaussian,
    create_correlation_matrix,
    create_banded,
    create_block_hierarchical
)
from D_encodings.rank2_asymmetric import (
    create_random_asymmetric,
    create_lower_triangular,
    create_antisymmetric,
    create_directional_gradient
)
from D_encodings.rank3_correlations import (
    create_random_rank3,
    create_partial_symmetric_rank3,
    create_local_coupling_rank3
)

# Tests
from utilities.test_symmetry import (
    test_symmetry_preservation,
    test_symmetry_creation,
    test_asymmetry_evolution
)
from utilities.test_norm import (
    test_norm_evolution,
    test_bounds_preservation,
    test_spectral_evolution
)
from utilities.test_diversity import (
    test_diversity_preservation,
    test_entropy_evolution,
    test_uniformity
)
from utilities.test_convergence import (
    test_convergence_to_fixed_point,
    test_lyapunov_exponent
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Phase 0 (calibration)
PHASE0_CONFIG = {
    'beta_values': [1.0, 2.0],
    'd_bases': {
        'SYM-001': (create_identity, {'n_dof': 50}),
        'SYM-002': (create_random_uniform, {'n_dof': 50}),
        'ASY-001': (create_random_asymmetric, {'n_dof': 50}),
    },
    'modifiers': {
        'M0': None,  # Base seule
        'M1': lambda seed: add_gaussian_noise(sigma=0.05, seed=seed),
    },
    'seeds': [1, 2, 3],
    'max_iterations': 200,  # Court pour calibration
}

# Phase 2 (exploration complète)
PHASE2_CONFIG = {
    'beta_values': [0.5, 1.0, 2.0, 5.0],
    'd_bases': {
        # Symétriques
        'SYM-001': (create_identity, {'n_dof': 50}),
        'SYM-002': (create_random_uniform, {'n_dof': 50}),
        'SYM-003': (create_random_gaussian, {'n_dof': 50, 'sigma': 0.3}),
        'SYM-004': (create_correlation_matrix, {'n_dof': 50}),
        'SYM-005': (create_banded, {'n_dof': 50, 'bandwidth': 3}),
        'SYM-006': (create_block_hierarchical, {'n_dof': 50, 'n_blocks': 10}),
        
        # Asymétriques
        'ASY-001': (create_random_asymmetric, {'n_dof': 50}),
        'ASY-002': (create_lower_triangular, {'n_dof': 50}),
        'ASY-003': (create_antisymmetric, {'n_dof': 50}),
        'ASY-004': (create_directional_gradient, {'n_dof': 50}),
        
        # Rang 3
        'R3-001': (create_random_rank3, {'n_dof': 20}),
        'R3-002': (create_partial_symmetric_rank3, {'n_dof': 20}),
        'R3-003': (create_local_coupling_rank3, {'n_dof': 20, 'radius': 2}),
    },
    'modifiers': {
        'M0': None,
        'M1': lambda seed: add_gaussian_noise(sigma=0.05, seed=seed),
        'M2': lambda seed: add_uniform_noise(amplitude=0.1, seed=seed),
        'M3': None,  # TODO: sparsification
    },
    'seeds': [1, 2, 3, 4, 5],
    'max_iterations': 2000,
}


# ============================================================================
# EXÉCUTION SINGLE RUN
# ============================================================================

def execute_single_run(beta, d_base_id, d_base_generator, d_base_params,
                      modifier_id, modifier_factory, seed, max_iterations):
    """
    Exécute 1 cellule de l'hypercube.
    
    Returns:
        dict avec résultats et status
    """
    run_id = f"GAM-001_beta{beta}_D{d_base_id}_M{modifier_id}_seed{seed}"
    
    print(f"\n{'='*70}")
    print(f"RUN: {run_id}")
    print(f"{'='*70}")
    
    try:
        # 1. Générer D^(base)
        # Essayer d'abord avec seed, puis sans si ça échoue
        try:
            params_with_seed = {**d_base_params, 'seed': seed}
            D_base = d_base_generator(**params_with_seed)
        except TypeError:
            # Le générateur ne prend pas de seed (ex: create_identity)
            D_base = d_base_generator(**d_base_params)
        
        print(f"✓ D^(base) généré: {d_base_id}, shape={D_base.shape}")
        
        # 2. Appliquer modifiers
        if modifier_factory is None:
            D_final = prepare_state(D_base, [])
            print(f"✓ Modifier: aucun (M0)")
        else:
            modifier = modifier_factory(seed)
            D_final = prepare_state(D_base, [modifier])
            print(f"✓ Modifier appliqué: {modifier_id}")
        
        # 3. Créer Γ
        gamma = PureSaturationGamma(beta=beta)
        print(f"✓ Γ créé: {gamma}")
        
        # 4. Exécuter kernel
        print(f"✓ Exécution kernel (max_iter={max_iterations})...")
        history = []
        final_iteration = 0
        
        for i, state in run_kernel(D_final, gamma, 
                                   max_iterations=max_iterations,
                                   record_history=False):
            # Enregistrer snapshot tous les 10 itérations
            if i % 10 == 0:
                history.append(state.copy())
            
            final_iteration = i
            
            # Détection explosion early (sécurité)
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"  ⚠ Explosion détectée à iteration {i}")
                break
        
        # Ajouter état final si pas déjà dans history
        if final_iteration % 10 != 0:
            history.append(state.copy())
        
        print(f"✓ Kernel terminé: {final_iteration} iterations, {len(history)} snapshots")
        
        # 5. Appliquer tests
        print(f"✓ Application des tests...")
        results = run_tests(history, D_base, d_base_id)
        
        # 6. Résumé
        print(f"\n{'─'*70}")
        print(f"RÉSUMÉ DES TESTS:")
        print(f"{'─'*70}")
        for test_result in results.values():
            print(f"  {test_result}")
        
        # 7. Verdict global
        n_pass = sum(1 for r in results.values() if r.status == "PASS")
        n_fail = sum(1 for r in results.values() if r.status == "FAIL")
        n_neutral = sum(1 for r in results.values() if r.status == "NEUTRAL")
        n_total = len(results)
        
        # Vérifier blockers
        blockers = [r for r in results.values() if r.blocking and r.status == "FAIL"]
        
        if blockers:
            global_status = "REJECTED"
            print(f"\n❌ VERDICT: REJECTED ({len(blockers)} tests bloquants échoués)")
        elif n_fail > n_pass:
            global_status = "POOR"
            print(f"\n⚠ VERDICT: POOR ({n_fail} FAIL vs {n_pass} PASS)")
        elif n_pass > n_total / 2:
            global_status = "PASS"
            print(f"\n✓ VERDICT: PASS ({n_pass}/{n_total} tests réussis)")
        else:
            global_status = "NEUTRAL"
            print(f"\n○ VERDICT: NEUTRAL ({n_pass} PASS, {n_fail} FAIL, {n_neutral} NEUTRAL)")
        
        return {
            'run_id': run_id,
            'status': 'COMPLETED',
            'global_status': global_status,
            'params': {
                'beta': beta,
                'd_base_id': d_base_id,
                'modifier_id': modifier_id,
                'seed': seed,
            },
            'iterations': final_iteration,
            'tests': results,
            'summary': {
                'n_pass': n_pass,
                'n_fail': n_fail,
                'n_neutral': n_neutral,
                'blockers': len(blockers),
            }
        }
    
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'run_id': run_id,
            'status': 'ERROR',
            'error': str(e),
            'params': {
                'beta': beta,
                'd_base_id': d_base_id,
                'modifier_id': modifier_id,
                'seed': seed,
            }
        }


def run_tests(history, D_base, d_base_id):
    """
    Applique tous les tests applicables selon le type de D.
    
    Returns:
        dict {test_name: result}
    """
    results = {}
    
    # Tests universels (toujours applicables)
    results['UNIV-001'] = test_norm_evolution(history, name="UNIV-001")
    results['UNIV-002'] = test_diversity_preservation(history, name="UNIV-002")
    results['UNIV-003'] = test_convergence_to_fixed_point(history, name="UNIV-003")
    
    # Tests symétrie (si rang 2)
    if D_base.ndim == 2:
        if d_base_id.startswith('SYM'):
            # Base symétrique → tester préservation
            results['SYM-001'] = test_symmetry_preservation(history, name="SYM-001")
        elif d_base_id.startswith('ASY'):
            # Base asymétrique → tester création symétrie
            results['SYM-002'] = test_symmetry_creation(history, name="SYM-002")
        
        # Évolution asymétrie (toutes matrices)
        results['SYM-003'] = test_asymmetry_evolution(history, name="SYM-003")
        
        # Tests structure rang 2
        results['STR-002'] = test_spectral_evolution(history, name="STR-002")
    
    # Tests bornes
    results['BND-001'] = test_bounds_preservation(
        history, 
        initial_bounds=(-1.0, 1.0),
        name="BND-001"
    )
    
    # Tests complémentaires
    results['DIV-ENTROPY'] = test_entropy_evolution(history, name="DIV-ENTROPY")
    results['DIV-UNIFORM'] = test_uniformity(history, name="DIV-UNIFORM")
    results['CONV-LYAPUNOV'] = test_lyapunov_exponent(history, name="CONV-LYAPUNOV")
    
    return results


# ============================================================================
# EXÉCUTION BATCH
# ============================================================================

def run_phase0():
    """Exécute Phase 0 (calibration) : 36 runs."""
    print("\n" + "="*70)
    print(" TM-GAM-001 - PHASE 0 (CALIBRATION)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  β values: {PHASE0_CONFIG['beta_values']}")
    print(f"  D bases: {list(PHASE0_CONFIG['d_bases'].keys())}")
    print(f"  Modifiers: {list(PHASE0_CONFIG['modifiers'].keys())}")
    print(f"  Seeds: {PHASE0_CONFIG['seeds']}")
    
    n_total = (len(PHASE0_CONFIG['beta_values']) * 
               len(PHASE0_CONFIG['d_bases']) * 
               len(PHASE0_CONFIG['modifiers']) * 
               len(PHASE0_CONFIG['seeds']))
    
    print(f"\nTotal runs: {n_total}")
    print("\n" + "="*70 + "\n")
    
    all_results = []
    completed = 0
    
    for beta in PHASE0_CONFIG['beta_values']:
        for d_base_id, (generator, params) in PHASE0_CONFIG['d_bases'].items():
            for modifier_id, modifier_factory in PHASE0_CONFIG['modifiers'].items():
                for seed in PHASE0_CONFIG['seeds']:
                    
                    result = execute_single_run(
                        beta, d_base_id, generator, params,
                        modifier_id, modifier_factory, seed,
                        PHASE0_CONFIG['max_iterations']
                    )
                    
                    all_results.append(result)
                    completed += 1
                    
                    print(f"\n{'='*70}")
                    print(f"PROGRÈS: {completed}/{n_total} ({100*completed/n_total:.1f}%)")
                    print(f"{'='*70}\n")
    
    # Rapport final
    print_final_report(all_results, "PHASE 0")
    
    return all_results


def print_final_report(results, phase_name):
    """Génère rapport final agrégé."""
    print("\n" + "#"*70)
    print(f"# RAPPORT FINAL - {phase_name}")
    print("#"*70)
    
    # Statistiques globales
    n_total = len(results)
    n_completed = sum(1 for r in results if r['status'] == 'COMPLETED')
    n_errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nStatistiques globales:")
    print(f"  Total runs: {n_total}")
    print(f"  Complétés: {n_completed} ({100*n_completed/n_total:.1f}%)")
    print(f"  Erreurs: {n_errors} ({100*n_errors/n_total:.1f}%)")
    
    # Verdicts globaux
    if n_completed > 0:
        completed_results = [r for r in results if r['status'] == 'COMPLETED']
        
        n_pass = sum(1 for r in completed_results if r['global_status'] == 'PASS')
        n_poor = sum(1 for r in completed_results if r['global_status'] == 'POOR')
        n_rejected = sum(1 for r in completed_results if r['global_status'] == 'REJECTED')
        n_neutral = sum(1 for r in completed_results if r['global_status'] == 'NEUTRAL')
        
        print(f"\nVerdicts globaux:")
        print(f"  PASS: {n_pass} ({100*n_pass/n_completed:.1f}%)")
        print(f"  POOR: {n_poor} ({100*n_poor/n_completed:.1f}%)")
        print(f"  REJECTED: {n_rejected} ({100*n_rejected/n_completed:.1f}%)")
        print(f"  NEUTRAL: {n_neutral} ({100*n_neutral/n_completed:.1f}%)")
        
        # Matrice β × D
        print(f"\nMatrice β × D_base:")
        print(f"{'β':<10}", end="")
        d_bases = sorted(set(r['params']['d_base_id'] for r in completed_results))
        for d_base in d_bases:
            print(f"{d_base:<15}", end="")
        print()
        
        betas = sorted(set(r['params']['beta'] for r in completed_results))
        for beta in betas:
            print(f"{beta:<10.1f}", end="")
            for d_base in d_bases:
                # Moyenne des statuts pour cette combinaison
                subset = [r for r in completed_results 
                         if r['params']['beta'] == beta 
                         and r['params']['d_base_id'] == d_base]
                
                if subset:
                    n_pass_subset = sum(1 for r in subset if r['global_status'] == 'PASS')
                    score = n_pass_subset / len(subset)
                    print(f"{score:.0%} ({len(subset)})", end="    ")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
    
    print("\n" + "#"*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TM-GAM-001: Saturation pure")
    parser.add_argument('--phase', type=int, choices=[0, 2], default=0,
                       help='Phase à exécuter (0=calibration, 2=complet)')
    parser.add_argument('--single', action='store_true',
                       help='Exécuter un seul run pour test rapide')
    
    args = parser.parse_args()
    
    if args.single:
        # Test rapide : 1 run
        print("\nMODE TEST RAPIDE (1 run)")
        result = execute_single_run(
            beta=2.0,
            d_base_id='SYM-001',
            d_base_generator=create_identity,
            d_base_params={'n_dof': 50},
            modifier_id='M0',
            modifier_factory=None,
            seed=42,
            max_iterations=200
        )
        print(f"\nRésultat: {result['status']}")
    
    elif args.phase == 0:
        results = run_phase0()
        print(f"\n✓ Phase 0 terminée: {len(results)} runs")
    
    elif args.phase == 2:
        print("\nPhase 2 non implémentée (TODO: adapter config)")
    
    print("\n✓ TM-GAM-001 terminé\n")