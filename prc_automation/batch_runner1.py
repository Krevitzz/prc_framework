#!/usr/bin/env python3
"""
prc_automation/batch_runner.py

Orchestration exhaustive exécutions R0 - Phase 1.

Usage:
    python batch_runner.py --phase 1 --gamma GAM-001
    python batch_runner.py --phase 1 --all
    python batch_runner.py --resume
"""

import argparse
import sqlite3
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np

# Imports PRC
from core.kernel import run_kernel
from core.state_preparation import prepare_state
from operators import get_operator_by_id, OPERATOR_REGISTRY
from modifiers.noise import add_gaussian_noise, add_uniform_noise

# D^(base) generators
from D_encodings import (
    create_identity, create_random_uniform, create_random_gaussian,
    create_correlation_matrix, create_banded, create_block_hierarchical,
    create_random_asymmetric, create_lower_triangular,
    create_antisymmetric, create_directional_gradient,
    create_random_rank3, create_partial_symmetric_rank3,
    create_local_coupling_rank3
)

# Tests
from tests.utilities import run_all_applicable_tests


# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

DB_PATH = Path("prc_database/prc_r0_results.db")

# Catalogue D^(base) complet
D_BASE_CATALOG = {
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
}

# Modificateurs standards
MODIFIERS = {
    'M0': None,  # Base seule
    'M1': lambda seed: add_gaussian_noise(sigma=0.05, seed=seed),
    'M2': lambda seed: add_uniform_noise(amplitude=0.1, seed=seed),
    'M3': None,  # TODO: sparsification
}

SEEDS = [1, 2, 3, 4, 5]

MAX_ITERATIONS_DEFAULT = 2000


# ============================================================================
# BASE DE DONNÉES
# ============================================================================

def init_database():
    """Initialise la base de données avec le schéma."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Lire et exécuter schema.sql
    schema_path = Path("prc_database/schema.sql")
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = f.read()
        conn.executescript(schema)
    
    conn.commit()
    conn.close()
    
    print(f"✓ Base de données initialisée: {DB_PATH}")


def execution_exists(conn, gamma_id, gamma_params, d_base_id, modifier_id, seed):
    """Vérifie si une exécution existe déjà."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM Executions
        WHERE gamma_id = ? AND gamma_params = ? 
          AND d_base_id = ? AND modifier_id = ? AND seed = ?
    """, (gamma_id, json.dumps(gamma_params), d_base_id, modifier_id, seed))
    
    return cursor.fetchone() is not None


def insert_execution(conn, gamma_id, gamma_params, d_base_id, modifier_id, seed,
                    status, error_message, final_iteration, converged,
                    global_verdict, execution_time):
    """Insère une exécution dans la DB."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO Executions (
            gamma_id, gamma_params, d_base_id, modifier_id, seed,
            max_iterations, status, error_message,
            final_iteration, converged, global_verdict, execution_time_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gamma_id, json.dumps(gamma_params), d_base_id, modifier_id, seed,
        MAX_ITERATIONS_DEFAULT, status, error_message,
        final_iteration, converged, global_verdict, execution_time
    ))
    
    exec_id = cursor.lastrowid
    conn.commit()
    return exec_id


def insert_test_results(conn, exec_id, test_results: Dict):
    """Insère les résultats de tests."""
    cursor = conn.cursor()
    
    for test_name, result in test_results.items():
        # Extraire catégorie du test (ex: "TEST-UNIV-001" → "UNIV")
        category = test_name.split('-')[0] if '-' in test_name else 'OTHER'
        
        cursor.execute("""
            INSERT INTO TestResults (
                exec_id, test_name, test_category, status, blocking,
                value, message, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id, test_name, category, result.status,
            getattr(result, 'blocking', False),
            getattr(result, 'final_diversity', None) or getattr(result, 'max_norm', None),
            result.message,
            json.dumps({})  # Metadata optionnelles
        ))
    
    conn.commit()


# ============================================================================
# EXÉCUTION SINGLE RUN
# ============================================================================

def execute_single_run(gamma_id: str, gamma_params: Dict,
                      d_base_id: str, modifier_id: str, seed: int,
                      verbose: bool = False) -> Dict:
    """
    Exécute 1 cellule de l'hypercube.
    
    Returns:
        dict avec résultats
    """
    run_id = f"{gamma_id}_{json.dumps(gamma_params)}_{d_base_id}_{modifier_id}_s{seed}"
    
    start_time = time.time()
    
    try:
        # 1. Générer D^(base)
        generator, params = D_BASE_CATALOG[d_base_id]
        
        # Ajouter seed si générateur le supporte
        try:
            D_base = generator(**{**params, 'seed': seed})
        except TypeError:
            D_base = generator(**params)
        
        if verbose:
            print(f"  D^(base): {d_base_id}, shape={D_base.shape}")
        
        # 2. Appliquer modifiers
        modifier_factory = MODIFIERS[modifier_id]
        if modifier_factory is None:
            D_final = prepare_state(D_base, [])
        else:
            modifier = modifier_factory(seed)
            D_final = prepare_state(D_base, [modifier])
        
        if verbose:
            print(f"  Modifier: {modifier_id}")
        
        # 3. Créer Γ
        gamma = get_operator_by_id(gamma_id, **gamma_params)
        
        # Réinitialiser mémoire si non-markovien
        if hasattr(gamma, 'reset'):
            gamma.reset()
        
        if verbose:
            print(f"  Γ: {gamma}")
        
        # 4. Exécuter kernel
        history = []
        final_iteration = 0
        converged = False
        
        for i, state in run_kernel(D_final, gamma,
                                   max_iterations=MAX_ITERATIONS_DEFAULT,
                                   record_history=False):
            if i % 10 == 0:
                history.append(state.copy())
            
            final_iteration = i
            
            # Détection explosion
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                if verbose:
                    print(f"  ⚠ Explosion détectée à iter {i}")
                break
        
        # Ajouter état final
        if final_iteration % 10 != 0:
            history.append(state.copy())
        
        if verbose:
            print(f"  Kernel: {final_iteration} iterations, {len(history)} snapshots")
        
        # 5. Appliquer tests
        test_results = run_all_applicable_tests(history, D_base, d_base_id, gamma_id)
        
        # 6. Verdict global
        n_pass = sum(1 for r in test_results.values() if r.status == "PASS")
        n_fail = sum(1 for r in test_results.values() if r.status == "FAIL")
        n_total = len(test_results)
        
        blockers = [r for r in test_results.values() 
                   if getattr(r, 'blocking', False) and r.status == "FAIL"]
        
        if blockers:
            global_verdict = "REJECTED"
        elif n_fail > n_pass:
            global_verdict = "POOR"
        elif n_pass > n_total / 2:
            global_verdict = "PASS"
        else:
            global_verdict = "NEUTRAL"
        
        execution_time = time.time() - start_time
        
        return {
            'run_id': run_id,
            'status': 'COMPLETED',
            'gamma_id': gamma_id,
            'gamma_params': gamma_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'final_iteration': final_iteration,
            'converged': converged,
            'global_verdict': global_verdict,
            'test_results': test_results,
            'execution_time': execution_time,
            'error_message': None,
        }
    
    except Exception as e:
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"  ❌ Erreur: {str(e)}")
            traceback.print_exc()
        
        return {
            'run_id': run_id,
            'status': 'ERROR',
            'gamma_id': gamma_id,
            'gamma_params': gamma_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'final_iteration': 0,
            'converged': False,
            'global_verdict': None,
            'test_results': {},
            'execution_time': execution_time,
            'error_message': str(e),
        }


# ============================================================================
# ORCHESTRATION BATCH
# ============================================================================

def run_phase1_gamma(gamma_id: str, verbose: bool = True):
    """
    Exécute Phase 1 pour un Γ donné.
    
    Phase 1: Paramètres nominaux uniquement, tous D, tous M, tous seeds.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1 - {gamma_id}")
    print(f"{'='*70}\n")
    
    # Vérifier que Γ existe et est implémenté
    if gamma_id not in OPERATOR_REGISTRY:
        print(f"❌ Γ inconnu: {gamma_id}")
        return
    
    info = OPERATOR_REGISTRY[gamma_id]
    if not info['implemented']:
        print(f"⏳ {gamma_id} non implémenté")
        return
    
    # Récupérer paramètres nominaux (Phase 1)
    # Importer dynamiquement le module
    module_name = f"operators.gamma_hyp_{gamma_id.split('-')[1]}"
    try:
        module = __import__(module_name, fromlist=['PARAM_GRID_PHASE1'])
        param_grid = module.PARAM_GRID_PHASE1
    except (ImportError, AttributeError):
        print(f"⚠ Pas de grille Phase 1 pour {gamma_id}, utilisation paramètres par défaut")
        param_grid = {'nominal': {}}
    
    # Récupérer D applicables
    d_applicability = info.get('metadata', {}).get('d_applicability', ['SYM', 'ASY', 'R3'])
    
    # Filtrer D_BASE_CATALOG selon applicabilité
    d_bases = {
        d_id: d_info
        for d_id, d_info in D_BASE_CATALOG.items()
        if any(d_id.startswith(prefix) for prefix in d_applicability)
    }
    
    # Calculer nombre de runs
    n_params = len(param_grid)
    n_d = len(d_bases)
    n_m = len(MODIFIERS)
    n_seeds = len(SEEDS)
    n_total = n_params * n_d * n_m * n_seeds
    
    print(f"Configuration:")
    print(f"  Paramètres: {list(param_grid.keys())}")
    print(f"  D bases: {len(d_bases)}")
    print(f"  Modifiers: {len(MODIFIERS)}")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total runs: {n_total}\n")
    
    # Connexion DB
    conn = sqlite3.connect(DB_PATH)
    
    completed = 0
    skipped = 0
    errors = 0
    
    # Boucles d'exécution
    for param_name, params in param_grid.items():
        for d_base_id in sorted(d_bases.keys()):
            for modifier_id in sorted(MODIFIERS.keys()):
                for seed in SEEDS:
                    
                    # Vérifier si déjà exécuté
                    if execution_exists(conn, gamma_id, params, d_base_id, modifier_id, seed):
                        if verbose:
                            print(f"⏭  Skip (existant): {gamma_id}_{param_name}_{d_base_id}_{modifier_id}_s{seed}")
                        skipped += 1
                        continue
                    
                    # Exécuter
                    if verbose:
                        print(f"\n▶ Run: {gamma_id}_{param_name}_{d_base_id}_{modifier_id}_s{seed}")
                    
                    result = execute_single_run(
                        gamma_id, params, d_base_id, modifier_id, seed,
                        verbose=verbose
                    )
                    
                    # Insérer dans DB
                    exec_id = insert_execution(
                        conn,
                        gamma_id, params, d_base_id, modifier_id, seed,
                        result['status'], result['error_message'],
                        result['final_iteration'], result['converged'],
                        result['global_verdict'], result['execution_time']
                    )
                    
                    if result['status'] == 'COMPLETED':
                        insert_test_results(conn, exec_id, result['test_results'])
                        completed += 1
                        
                        if verbose:
                            print(f"  ✓ Verdict: {result['global_verdict']}")
                    else:
                        errors += 1
                    
                    # Progression
                    total_processed = completed + skipped + errors
                    print(f"  Progrès: {total_processed}/{n_total} ({100*total_processed/n_total:.1f}%)")
    
    conn.close()
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ {gamma_id}")
    print(f"{'='*70}")
    print(f"Total: {n_total} runs")
    print(f"  Complétés: {completed}")
    print(f"  Sautés (existants): {skipped}")
    print(f"  Erreurs: {errors}")
    print(f"{'='*70}\n")


def run_phase1_all():
    """Exécute Phase 1 pour tous les Γ implémentés."""
    print("\n" + "#"*70)
    print("# PHASE 1 - BALAYAGE NOMINAL")
    print("# Tous Γ avec paramètres nominaux")
    print("#"*70 + "\n")
    
    implemented = [
        gamma_id for gamma_id, info in OPERATOR_REGISTRY.items()
        if info['implemented']
    ]
    
    print(f"Γ implémentés: {len(implemented)}")
    for gamma_id in implemented:
        print(f"  - {gamma_id}")
    
    print()
    
    for gamma_id in implemented:
        run_phase1_gamma(gamma_id, verbose=False)
    
    print("\n" + "#"*70)
    print("# PHASE 1 TERMINÉE")
    print("#"*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Runner Phase 1")
    parser.add_argument('--phase', type=int, default=1, help='Phase (1 ou 2)')
    parser.add_argument('--gamma', type=str, help='Γ spécifique (ex: GAM-001)')
    parser.add_argument('--all', action='store_true', help='Tous les Γ')
    parser.add_argument('--init-db', action='store_true', help='Initialiser DB')
    parser.add_argument('--verbose', action='store_true', help='Mode verbeux')
    
    args = parser.parse_args()
    
    # Initialiser DB si demandé
    if args.init_db or not DB_PATH.exists():
        init_database()
    
    # Exécuter
    if args.phase == 1:
        if args.all:
            run_phase1_all()
        elif args.gamma:
            run_phase1_gamma(args.gamma, verbose=args.verbose)
        else:
            print("❌ Spécifier --all ou --gamma GAM-XXX")
    else:
        print("❌ Phase 2 non implémentée")


if __name__ == "__main__":
    main()