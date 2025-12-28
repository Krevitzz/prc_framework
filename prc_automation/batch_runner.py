#!/usr/bin/env python3
"""
prc_automation/batch_runner.py

Orchestration exhaustive exploration R0 - Pipeline complet (Section 14.2).

PRINCIPE FONDAMENTAL (Section 14.2) :
  - UN SEUL point d'entrée
  - TROIS modes : --brut, --test, --verdict
  - Chaînage AUTOMATIQUE des dépendances

MODES :
  --brut    : Collecte données runs (écrit db_raw)
  --test    : Application tests + scoring (écrit db_results)
  --verdict : Calcul verdicts agrégés (écrit db_results)

RÈGLE CRITIQUE (Section 14.1) :
  Modifications de scoring/verdict SANS reruns du kernel.

Usage:
    batch_runner.py --brut --gamma GAM-001
    batch_runner.py --test --gamma GAM-001 --config weights_default
    batch_runner.py --verdict --gamma GAM-001 --config weights_default --thresholds thresholds_default
    batch_runner.py --all --gamma GAM-001 --config weights_default --thresholds thresholds_default
"""

import sys
import argparse
import sqlite3
import json
import time
import gzip
import pickle
import yaml
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Ajouter project root au path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

# Tests et scoring
from tests.utilities import (
    run_all_applicable_tests,
    score_all_observations,
    load_weights_config,
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")

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
SNAPSHOT_INTERVAL = 10


# =============================================================================
# MODE 1 : --brut (Collecte données runs)
# =============================================================================

def mode_brut(gamma_id: str, verbose: bool = True) -> int:
    """
    MODE --brut : Collecte données runs (Section 14.2).
    
    Responsabilité :
      - Exécute kernel pour tous (D, modifier, seed) applicables
      - Stocke dans db_raw (Executions, Snapshots, Metrics)
      - PAS de tests appliqués
      - PAS d'interprétation
    
    Args:
        gamma_id: ID du Γ (ex: "GAM-001")
        verbose: Mode verbeux
    
    Returns:
        Nombre de runs complétés
    """
    print(f"\n{'='*70}")
    print(f"MODE --brut : {gamma_id}")
    print(f"{'='*70}\n")
    
    # Vérifier Γ existe et est implémenté
    if gamma_id not in OPERATOR_REGISTRY:
        print(f"❌ Γ inconnu: {gamma_id}")
        return 0
    
    info = OPERATOR_REGISTRY[gamma_id]
    if not info['implemented']:
        print(f"⏳ {gamma_id} non implémenté")
        return 0
    
    # Charger grille paramètres Phase 1
    module_name = f"operators.gamma_hyp_{gamma_id.split('-')[1]}"
    try:
        module = __import__(module_name, fromlist=['PARAM_GRID_PHASE1'])
        param_grid = module.PARAM_GRID_PHASE1
    except (ImportError, AttributeError):
        print(f"⚠ Pas de PARAM_GRID_PHASE1, utilisation paramètres vides")
        param_grid = {'nominal': {}}
    
    # Filtrer D applicables
    d_applicability = info.get('metadata', {}).get('d_applicability', ['SYM', 'ASY', 'R3'])
    d_bases = {
        d_id: d_info
        for d_id, d_info in D_BASE_CATALOG.items()
        if any(d_id.startswith(prefix) for prefix in d_applicability)
    }
    
    # Calculer nombre total
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
    
    # Connexion db_raw
    if not DB_RAW_PATH.exists():
        print("❌ db_raw non initialisée. Lancer: python init_databases.py")
        return 0
    
    conn_raw = sqlite3.connect(DB_RAW_PATH)
    
    completed = 0
    skipped = 0
    errors = 0
    
    # Boucles d'exécution
    for param_name, params in param_grid.items():
        for d_base_id in sorted(d_bases.keys()):
            for modifier_id in sorted(MODIFIERS.keys()):
                for seed in SEEDS:
                    
                    # Vérifier si run existe déjà
                    if execution_exists_raw(conn_raw, gamma_id, params, d_base_id, modifier_id, seed):
                        if verbose:
                            print(f"⏭  Skip (existant): {gamma_id}_{param_name}_{d_base_id}_{modifier_id}_s{seed}")
                        skipped += 1
                        continue
                    
                    # Exécuter run
                    if verbose:
                        print(f"\n▶ Run: {gamma_id}_{param_name}_{d_base_id}_{modifier_id}_s{seed}")
                    
                    result = execute_run_brut(
                        gamma_id, params, d_base_id, modifier_id, seed,
                        verbose=verbose
                    )
                    
                    # Insérer dans db_raw
                    if result['status'] == 'COMPLETED':
                        insert_execution_raw(conn_raw, result)
                        completed += 1
                    else:
                        insert_execution_raw(conn_raw, result)
                        errors += 1
                    
                    # Progression
                    total_proc = completed + skipped + errors
                    pct = 100 * total_proc / n_total
                    print(f"  Progrès: {total_proc}/{n_total} ({pct:.1f}%) "
                          f"[OK={completed}, Skip={skipped}, Err={errors}]")
    
    conn_raw.close()
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ MODE --brut : {gamma_id}")
    print(f"{'='*70}")
    print(f"Total: {n_total} runs")
    print(f"  Complétés: {completed}")
    print(f"  Sautés (existants): {skipped}")
    print(f"  Erreurs: {errors}")
    print(f"{'='*70}\n")
    
    return completed


def execute_run_brut(gamma_id: str, gamma_params: Dict,
                    d_base_id: str, modifier_id: str, seed: int,
                    verbose: bool = False) -> Dict:
    """
    Exécute 1 run et collecte données brutes.
    
    Returns:
        dict avec toutes les données collectées
    """
    start_time = time.time()
    
    try:
        # 1. Préparer D
        generator, d_params = D_BASE_CATALOG[d_base_id]
        
        try:
            D_base = generator(**{**d_params, 'seed': seed})
        except TypeError:
            D_base = generator(**d_params)
        
        # Appliquer modifier
        modifier_factory = MODIFIERS[modifier_id]
        if modifier_factory is None:
            D_final = prepare_state(D_base, [])
        else:
            modifier = modifier_factory(seed)
            D_final = prepare_state(D_base, [modifier])
        
        if verbose:
            print(f"  D: {d_base_id}, shape={D_base.shape}, modifier={modifier_id}")
        
        # 2. Créer Γ
        gamma = get_operator_by_id(gamma_id, **gamma_params)
        
        # Reset mémoire si non-markovien
        if hasattr(gamma, 'reset'):
            gamma.reset()
        
        if verbose:
            print(f"  Γ: {gamma}")
        
        # 3. Exécuter kernel et collecter métriques
        metrics_list = []
        snapshots = []
        previous_state = None
        converged = False
        convergence_iter = None
        
        for iteration, state in run_kernel(
            D_final, gamma,
            max_iterations=MAX_ITERATIONS_DEFAULT,
            record_history=False
        ):
            # Calculer métriques (fonction à implémenter)
            metrics = compute_metrics(state, previous_state)
            metrics['iteration'] = iteration
            metrics_list.append(metrics)
            
            # Sauvegarder snapshot
            if iteration % SNAPSHOT_INTERVAL == 0:
                snap_metrics = compute_metrics(state)
                snap_metrics['iteration'] = iteration
                snap_metrics['state'] = state.copy()
                snapshots.append(snap_metrics)
            
            # Détection explosion
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                if verbose:
                    print(f"  ⚠ Explosion détectée: iter={iteration}")
                break
            
            previous_state = state
        
        final_iteration = iteration
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Terminé: {final_iteration} iterations, {execution_time:.1f}s")
        
        return {
            'status': 'COMPLETED',
            'gamma_id': gamma_id,
            'gamma_params': gamma_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'final_iteration': final_iteration,
            'execution_time': execution_time,
            'metrics': metrics_list,
            'snapshots': snapshots,
        }
    
    except Exception as e:
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"  ❌ Erreur: {str(e)}")
            traceback.print_exc()
        
        return {
            'status': 'ERROR',
            'gamma_id': gamma_id,
            'gamma_params': gamma_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'final_iteration': 0,
            'execution_time': execution_time,
            'error_message': str(e),
            'metrics': [],
            'snapshots': [],
        }


def compute_metrics(state: np.ndarray, previous_state: Optional[np.ndarray] = None) -> Dict:
    """Calcule métriques observables brutes (même logique que batch_runner_raw.py)."""
    metrics = {}
    
    # Normes
    if state.ndim == 2:
        metrics['norm_frobenius'] = float(np.linalg.norm(state, 'fro'))
    else:
        metrics['norm_frobenius'] = float(np.linalg.norm(state.flatten()))
    
    metrics['norm_max'] = float(np.max(np.abs(state)))
    
    # Spectre (si rang 2 carré)
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        try:
            eigs = np.linalg.eigvalsh(state)
            metrics['norm_spectral'] = float(np.max(np.abs(eigs)))
        except np.linalg.LinAlgError:
            metrics['norm_spectral'] = None
    else:
        metrics['norm_spectral'] = None
    
    # Statistiques éléments
    flat = state.flatten()
    metrics['min_value'] = float(np.min(flat))
    metrics['max_value'] = float(np.max(flat))
    metrics['mean_value'] = float(np.mean(flat))
    metrics['std_value'] = float(np.std(flat))
    
    # Distance à l'état précédent
    if previous_state is not None:
        diff = state - previous_state
        if diff.ndim == 2:
            metrics['distance_to_previous'] = float(np.linalg.norm(diff, 'fro'))
        else:
            metrics['distance_to_previous'] = float(np.linalg.norm(diff.flatten()))
    else:
        metrics['distance_to_previous'] = None
    
    # Asymétrie (si rang 2 carré)
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        metrics['asymmetry_norm'] = float(np.linalg.norm(state - state.T, 'fro'))
    else:
        metrics['asymmetry_norm'] = None
    
    return metrics


# =============================================================================
# HELPERS DB_RAW
# =============================================================================

def execution_exists_raw(conn, gamma_id, gamma_params, d_base_id, modifier_id, seed):
    """Vérifie si run existe dans db_raw."""
    cursor = conn.cursor()
    params_json = json.dumps(gamma_params, sort_keys=True)
    cursor.execute("""
        SELECT id FROM Executions
        WHERE gamma_id = ? AND d_base_id = ? AND modifier_id = ? AND seed = ?
    """, (gamma_id, d_base_id, modifier_id, seed))
    return cursor.fetchone() is not None


def insert_execution_raw(conn, result: Dict) -> int:
    """Insère run dans db_raw."""
    cursor = conn.cursor()
    
    # Extraire tous les paramètres Γ
    all_params = extract_all_gamma_params(result['gamma_params'])
    
    cursor.execute("""
        INSERT INTO Executions (
            run_id, timestamp,
            gamma_id,
            alpha, beta, gamma_param, omega,
            memory_weight, window_size, epsilon,
            sigma, lambda_param,
            eta, subspace_dim,
            d_base_id, modifier_id, seed,
            max_iterations, snapshot_interval,
            status, error_message, final_iteration, execution_time_seconds,
            converged, convergence_iteration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        format_run_id(result),
        datetime.now().isoformat(),
        result['gamma_id'],
        all_params.get('alpha'), all_params.get('beta'),
        all_params.get('gamma_param'), all_params.get('omega'),
        all_params.get('memory_weight'), all_params.get('window_size'),
        all_params.get('epsilon'),
        all_params.get('sigma'), all_params.get('lambda_param'),
        all_params.get('eta'), all_params.get('subspace_dim'),
        result['d_base_id'], result['modifier_id'], result['seed'],
        MAX_ITERATIONS_DEFAULT, SNAPSHOT_INTERVAL,
        result['status'], result.get('error_message'),
        result.get('final_iteration'), result.get('execution_time'),
        result.get('converged', False), result.get('convergence_iter')
    ))
    
    exec_id = cursor.lastrowid
    
    # Insérer métriques
    for metrics in result.get('metrics', []):
        cursor.execute("""
            INSERT INTO Metrics (
                exec_id, iteration,
                norm_frobenius, norm_spectral, norm_max,
                min_value, max_value, mean_value, std_value,
                distance_to_previous, asymmetry_norm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id, metrics['iteration'],
            metrics['norm_frobenius'], metrics.get('norm_spectral'),
            metrics['norm_max'],
            metrics['min_value'], metrics['max_value'],
            metrics['mean_value'], metrics['std_value'],
            metrics.get('distance_to_previous'), metrics.get('asymmetry_norm')
        ))
    
    # Insérer snapshots
    for snap in result.get('snapshots', []):
        state_compressed = gzip.compress(pickle.dumps(snap['state']))
        cursor.execute("""
            INSERT INTO Snapshots (
                exec_id, iteration, state_blob,
                norm_frobenius, norm_spectral,
                min_value, max_value, mean_value, std_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id, snap['iteration'], state_compressed,
            snap['norm_frobenius'], snap.get('norm_spectral'),
            snap['min_value'], snap['max_value'],
            snap['mean_value'], snap['std_value']
        ))
    
    conn.commit()
    return exec_id


def extract_all_gamma_params(params: Dict) -> Dict:
    """Extrait tous paramètres Γ possibles (conformité schema_raw.sql)."""
    return {
        'alpha': params.get('alpha'),
        'beta': params.get('beta'),
        'gamma_param': params.get('gamma'),
        'omega': params.get('omega'),
        'memory_weight': params.get('memory_weight'),
        'window_size': params.get('window_size'),
        'epsilon': params.get('epsilon'),
        'sigma': params.get('sigma'),
        'lambda_param': params.get('lambda'),
        'eta': params.get('eta'),
        'subspace_dim': params.get('subspace_dim'),
    }


def format_run_id(result: Dict) -> str:
    """Formate run_id unique."""
    params_str = "_".join(f"{k}{v}" for k, v in result['gamma_params'].items())
    return f"{result['gamma_id']}_{params_str}_{result['d_base_id']}_{result['modifier_id']}_s{result['seed']}"


# =============================================================================
# MODE 2 : --test (Application tests + scoring)
# =============================================================================

def mode_test(gamma_id: str, config_id: str, verbose: bool = True) -> int:
    """
    MODE --test : Application tests + scoring (Section 14.2).
    
    Responsabilité :
      - Vérifie que db_raw contient les runs nécessaires
      - Si runs manquants : exécute --brut automatiquement
      - Lit runs depuis db_raw
      - Applique tests applicables
      - Calcule scores avec config spécifiée
      - Stocke dans db_results (TestObservations, TestScores)
    
    Args:
        gamma_id: ID du Γ
        config_id: ID config pondérations
        verbose: Mode verbeux
    
    Returns:
        Nombre de tests appliqués
    """
    print(f"\n{'='*70}")
    print(f"MODE --test : {gamma_id}, config={config_id}")
    print(f"{'='*70}\n")
    
    # 1. Vérifier que db_raw existe
    if not DB_RAW_PATH.exists():
        print("❌ db_raw non trouvée. Initialiser d'abord.")
        return 0
    
    # 2. Vérifier que db_results existe
    if not DB_RESULTS_PATH.exists():
        print("❌ db_results non trouvée. Initialiser d'abord.")
        return 0
    
    # 3. Vérifier que runs existent dans db_raw
    conn_raw = sqlite3.connect(DB_RAW_PATH)
    cursor = conn_raw.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM Executions WHERE gamma_id = ? AND status = 'COMPLETED'
    """, (gamma_id,))
    n_runs_raw = cursor.fetchone()[0]
    
    if n_runs_raw == 0:
        print(f"⚠ Aucun run trouvé dans db_raw pour {gamma_id}")
        print(f"→ Exécution automatique de --brut...")
        conn_raw.close()
        
        # Chaînage automatique : exécuter --brut d'abord
        n_completed = mode_brut(gamma_id, verbose=verbose)
        
        if n_completed == 0:
            print("❌ Aucun run complété par --brut")
            return 0
        
        # Reconnecter après --brut
        conn_raw = sqlite3.connect(DB_RAW_PATH)
    
    print(f"✓ {n_runs_raw} runs trouvés dans db_raw")
    
    # 4. Charger config pondérations
    try:
        weights = load_weights_config(config_id)
        print(f"✓ Config chargée: {config_id} ({len(weights)} tests)")
    except FileNotFoundError:
        print(f"❌ Config non trouvée: {config_id}")
        conn_raw.close()
        return 0
    
    # 5. Récupérer tous les runs à tester
    cursor.execute("""
        SELECT id, gamma_id, d_base_id, modifier_id, seed
        FROM Executions
        WHERE gamma_id = ? AND status = 'COMPLETED'
    """, (gamma_id,))
    runs_to_test = cursor.fetchall()
    
    print(f"\nTraitement de {len(runs_to_test)} runs...")
    
    # 6. Connexion db_results
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    
    tested = 0
    skipped = 0
    errors = 0
    
    for exec_id, gamma_id_db, d_base_id, modifier_id, seed in runs_to_test:
        # Vérifier si déjà testé pour cette config
        if tests_exist_for_config(conn_results, exec_id, config_id):
            if verbose:
                print(f"⏭  Skip (déjà testé): exec_id={exec_id}, config={config_id}")
            skipped += 0#1
            continue
        
        if verbose:
            print(f"\n▶ Test run: exec_id={exec_id}, D={d_base_id}, M={modifier_id}, s={seed}")
        
        try:
            # Charger historique depuis db_raw
            history = load_history_from_snapshots(conn_raw, exec_id)
            
            if not history:
                print(f"  ⚠ Historique vide, skip")
                errors += 1
                continue
            
            # Charger D_base (pour tests applicabilité)
            D_base = load_d_base(d_base_id, seed)
            
            # Appliquer tous les tests applicables
            observations = run_all_applicable_tests(
                history, D_base, d_base_id, gamma_id_db
            )
            
            if verbose:
                print(f"  ✓ {len(observations)} tests appliqués")
            
            # Calculer scores
            context = {
                'd_base_id': d_base_id,
                'gamma_id': gamma_id_db,
                'modifier_id': modifier_id,
                'seed': seed,
            }
            
            scoring_results = score_all_observations(
                observations, context, config_id
            )
            
            if verbose:
                print(f"  ✓ Score global: {scoring_results['global_score']:.2f}/20")
            
            # Insérer dans db_results
            insert_test_observations(conn_results, exec_id, observations)
            insert_test_scores(conn_results, exec_id, scoring_results, config_id)
            
            tested += 1
        
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")
            if verbose:
                traceback.print_exc()
            errors += 1
            continue
    
    conn_raw.close()
    conn_results.close()
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ MODE --test : {gamma_id}")
    print(f"{'='*70}")
    print(f"Total: {len(runs_to_test)} runs")
    print(f"  Testés: {tested}")
    print(f"  Sautés (existants): {skipped}")
    print(f"  Erreurs: {errors}")
    print(f"{'='*70}\n")
    
    return tested


# =============================================================================
# HELPERS MODE --test
# =============================================================================

def tests_exist_for_config(conn, exec_id: int, config_id: str) -> bool:
    """Vérifie si tests déjà appliqués pour cette config."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM TestScores
        WHERE exec_id = ? AND config_id = ?
    """, (exec_id, config_id))
    return cursor.fetchone()[0] > 0


def load_history_from_snapshots(conn, exec_id: int) -> List[np.ndarray]:
    """Charge historique complet depuis snapshots et état final."""
    cursor = conn.cursor()
    
    # 1. Charger tous les snapshots
    cursor.execute("""
        SELECT iteration, state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    history = []
    snapshots_loaded = 0
    
    for iteration, state_blob in cursor.fetchall():
        try:
            state_bytes = gzip.decompress(state_blob)
            state = pickle.loads(state_bytes)
            history.append(state)
            snapshots_loaded += 1
        except Exception as e:
            print(f"  ⚠ Erreur chargement snapshot iter={iteration}: {str(e)}")
            continue
    
    # 2. Si moins de 2 snapshots, c'est problématique
    if snapshots_loaded < 2:
        print(f"  ⚠ Historique insuffisant: {snapshots_loaded} snapshots")
        return []
    
    # 3. S'assurer qu'on a l'état final (dernière itération)
    cursor.execute("""
        SELECT final_iteration FROM Executions 
        WHERE id = ?
    """, (exec_id,))
    
    final_iter = cursor.fetchone()[0]
    last_snap_iter = history[-1].iteration if hasattr(history[-1], 'iteration') else len(history)-1
    
    # 4. Si le dernier snapshot n'est pas l'état final, essayer de le récupérer
    if last_snap_iter != final_iter:
        print(f"  ⚠ Dernier snapshot ({last_snap_iter}) != final ({final_iter})")
        # On pourrait charger l'état final depuis Metrics si disponible
    
    print(f"  ✓ Historique chargé: {len(history)} états")
    return history


def load_d_base(d_base_id: str, seed: int) -> np.ndarray:
    """Recrée D_base (pour tests applicabilité)."""
    if d_base_id not in D_BASE_CATALOG:
        raise ValueError(f"D_base inconnu: {d_base_id}")
    
    generator, params = D_BASE_CATALOG[d_base_id]
    
    try:
        return generator(**{**params, 'seed': seed})
    except TypeError:
        return generator(**params)


def insert_test_observations(conn, exec_id: int, observations: Dict):
    """Insère observations dans db_results."""
    cursor = conn.cursor()
    
    for test_name, obs in observations.items():
        # Extraire catégorie
        category = test_name.split('-')[0]
        
        # Sérialiser observation complète en JSON
        obs_data = {
            'test_name': obs.test_name,
            'status': obs.status,
            'message': obs.message,
            'blocking': getattr(obs, 'blocking', False),
        }
        
        # Ajouter champs spécifiques selon type
        if hasattr(obs, 'initial_norm'):
            obs_data['initial_value'] = obs.initial_norm
            obs_data['final_value'] = obs.final_norm
        elif hasattr(obs, 'initial_diversity'):
            obs_data['initial_value'] = obs.initial_diversity
            obs_data['final_value'] = obs.final_diversity
        elif hasattr(obs, 'initial_symmetric'):
            obs_data['initial_value'] = 1.0 if obs.initial_symmetric else 0.0
            obs_data['final_value'] = 1.0 if obs.final_symmetric else 0.0
        
        obs_data['transition'] = getattr(obs, 'transition', None) or getattr(obs, 'evolution', 'unknown')
        
        cursor.execute("""
            INSERT OR IGNORE INTO TestObservations (
                exec_id, test_name, test_category,
                applicable, observation_data,
                initial_value, final_value, transition,
                computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id, test_name, category,
            True,  # Applicable (sinon pas dans observations)
            json.dumps(obs_data),
            obs_data.get('initial_value'),
            obs_data.get('final_value'),
            obs_data.get('transition'),
            datetime.now().isoformat()
        ))
    
    conn.commit()


def insert_test_scores(conn, exec_id: int, scoring_results: Dict, config_id: str):
    """Insère scores dans db_results."""
    cursor = conn.cursor()
    
    for test_name, score_data in scoring_results['test_scores'].items():
        cursor.execute("""
            INSERT OR IGNORE INTO TestScores (
                exec_id, test_name, config_id,
                score, weight, weighted_score,
                computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id, test_name, config_id,
            score_data['score'],
            score_data['weight'],
            score_data['weighted_score'],
            datetime.now().isoformat()
        ))
    
    conn.commit()


# =============================================================================
# MODE 3 : --verdict (Calcul verdicts)
# =============================================================================

def mode_verdict(gamma_id: str, config_id: str, threshold_id: str, verbose: bool = True) -> Dict:
    """
    MODE --verdict : Calcul verdicts agrégés (Section 14.2).
    
    [Docstring inchangée...]
    """
    print(f"\n{'='*70}")
    print(f"MODE --verdict : {gamma_id}, config={config_id}, thresholds={threshold_id}")
    print(f"{'='*70}\n")
    
    # 1. Vérifier que db_results existe
    if not DB_RESULTS_PATH.exists():
        print("❌ db_results non trouvée. Initialiser d'abord.")
        return {}
    
    # 2. Vérifier que db_raw existe (nécessaire pour ATTACH)
    if not DB_RAW_PATH.exists():
        print("❌ db_raw non trouvée. Nécessaire pour calcul critères.")
        return {}
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    
    # === FIX CRITIQUE : ATTACH db_raw ===
    # Permet requêtes cross-database sans redondance de données
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # 3. Vérifier que scores existent pour cette config
    cursor.execute("""
        SELECT COUNT(DISTINCT exec_id) 
        FROM TestScores
        WHERE config_id = ?
    """, (config_id,))
    n_scored_runs = cursor.fetchone()[0]
    
    if n_scored_runs == 0:
        print(f"⚠ Aucun score trouvé pour config={config_id}")
        print(f"→ Exécution automatique de --test...")
        
        # Détacher avant de fermer
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        
        # Chaînage automatique : exécuter --test d'abord
        n_tested = mode_test(gamma_id, config_id, verbose=verbose)
        
        if n_tested == 0:
            print("❌ Aucun test complété par --test")
            return {}
        
        # Reconnecter et réattacher après --test
        conn_results = sqlite3.connect(DB_RESULTS_PATH)
        cursor = conn_results.cursor()
        cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    print(f"✓ {n_scored_runs} runs scorés trouvés pour config={config_id}")
    
    # 4. Vérifier si verdict existe déjà
    cursor.execute("""
        SELECT verdict FROM GammaVerdicts
        WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
    """, (gamma_id, config_id, threshold_id))
    
    existing = cursor.fetchone()
    if existing:
        print(f"⚠ Verdict existant: {existing[0]}")
        print(f"  Suppression et recalcul...")
        cursor.execute("""
            DELETE FROM GammaVerdicts
            WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
        """, (gamma_id, config_id, threshold_id))
        conn_results.commit()
    
    # 5. Charger seuils depuis YAML
    try:
        thresholds = load_thresholds_config(threshold_id)
        print(f"✓ Seuils chargés: {threshold_id}")
    except FileNotFoundError:
        print(f"❌ Config seuils non trouvée: {threshold_id}")
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return {}
    
    # 6. Calculer les 3 critères (connexion a maintenant accès à db_raw via ATTACH)
    print("\nCalcul des critères...")
    
    criteria = compute_verdict_criteria(conn_results, gamma_id, config_id)
    
    print(f"  Majorité: {criteria['majority_pct']:.1f}%")
    print(f"  Robustesse: {criteria['robustness_pct']:.1f}%")
    print(f"  Score global: {criteria['score_global']:.2f}/20")
    
    # 7. Appliquer logique verdict
    verdict = apply_verdict_logic(criteria, thresholds)
    
    print(f"\n→ VERDICT: {verdict['verdict']}")
    print(f"  Raison: {verdict['reason']}")
    
    # 8. Insérer verdict dans db_results
    insert_gamma_verdict(
        conn_results,
        gamma_id, config_id, threshold_id,
        criteria, verdict
    )
    
    # === DÉTACHER db_raw avant fermeture ===
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ MODE --verdict : {gamma_id}")
    print(f"{'='*70}")
    print(f"Config: {config_id}")
    print(f"Thresholds: {threshold_id}")
    print(f"Verdict: {verdict['verdict']}")
    print(f"{'='*70}\n")
    
    return verdict


# =============================================================================
# HELPERS MODE --verdict
# =============================================================================

def load_thresholds_config(threshold_id: str) -> Dict:
    """Charge seuils depuis YAML."""
    config_path = Path("config") / f"{threshold_id}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def compute_verdict_criteria(conn, gamma_id: str, config_id: str) -> Dict:
    """
    Calcule les 3 critères selon Feuille de Route Section 7.
    
    IMPORTANT: Connexion doit avoir db_raw attachée via ATTACH DATABASE.
    
    Returns:
        dict {
            'majority_pct': float,
            'robustness_pct': float,
            'score_global': float,
            'n_total_configs': int,
            'n_pass_configs': int,
            'n_viable_d_bases': int,
            'n_total_d_bases': int,
        }
    """
    cursor = conn.cursor()
    
    # ========================================================================
    # CRITÈRE 1 : Score global
    # ========================================================================
    # Score global = moyenne pondérée de tous les tests, ramené sur /20
    
    cursor.execute("""
        SELECT 
            AVG(ts.weighted_score) as avg_weighted,
            AVG(ts.weight) as avg_weight
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
    """, (gamma_id, config_id))
    
    row = cursor.fetchone()
    if row and row[0] is not None and row[1] is not None:
        avg_weighted = row[0]
        avg_weight = row[1]
        score_normalized = avg_weighted / avg_weight if avg_weight > 0 else 0
        score_global = score_normalized * 20.0
    else:
        score_global = 0.0
    
    # ========================================================================
    # CRITÈRE 2 : Majorité (% configs PASS)
    # ========================================================================
    # Une config = (exec_id) qui correspond à (gamma_params, d_base, modifier, seed)
    # PASS si score global de cette config ≥ 0.6 (12/20 comme seuil)
    
    # Total configs
    cursor.execute("""
        SELECT COUNT(DISTINCT ts.exec_id)
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
    """, (gamma_id, config_id))
    n_total_configs = cursor.fetchone()[0] or 0
    
    # Configs PASS (score moyen ≥ 0.6)
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT ts.exec_id
            FROM TestScores ts
            JOIN db_raw.Executions e ON ts.exec_id = e.id
            WHERE e.gamma_id = ? AND ts.config_id = ?
            GROUP BY ts.exec_id
            HAVING AVG(ts.weighted_score / ts.weight) >= 0.6
        )
    """, (gamma_id, config_id))
    n_pass_configs = cursor.fetchone()[0] or 0
    
    majority_pct = (n_pass_configs / n_total_configs * 100) if n_total_configs > 0 else 0.0
    
    # ========================================================================
    # CRITÈRE 3 : Robustesse (% D avec ≥1 config viable)
    # ========================================================================
    # Viable = score global de la config ≥ 0.6
    
    # Total D bases testés
    cursor.execute("""
        SELECT COUNT(DISTINCT e.d_base_id)
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        WHERE e.gamma_id = ? AND ts.config_id = ?
    """, (gamma_id, config_id))
    n_total_d_bases = cursor.fetchone()[0] or 0
    
    # D bases avec ≥1 config viable
    cursor.execute("""
        SELECT COUNT(DISTINCT d_base_id) FROM (
            SELECT e.d_base_id
            FROM db_raw.Executions e
            JOIN TestScores ts ON ts.exec_id = e.id
            WHERE e.gamma_id = ? AND ts.config_id = ?
            GROUP BY ts.exec_id, e.d_base_id
            HAVING AVG(ts.weighted_score / ts.weight) >= 0.6
        )
    """, (gamma_id, config_id))
    n_viable_d_bases = cursor.fetchone()[0] or 0
    
    robustness_pct = (n_viable_d_bases / n_total_d_bases * 100) if n_total_d_bases > 0 else 0.0
    
    # ========================================================================
    # RETOUR
    # ========================================================================
    
    return {
        'majority_pct': majority_pct,
        'robustness_pct': robustness_pct,
        'score_global': score_global,
        'n_total_configs': n_total_configs,
        'n_pass_configs': n_pass_configs,
        'n_viable_d_bases': n_viable_d_bases,
        'n_total_d_bases': n_total_d_bases,
    }




def apply_verdict_logic(criteria: Dict, thresholds: Dict) -> Dict:
    """
    Applique logique verdict selon Section 14.6.
    
    SURVIVES[R0] (OU logique) :
      score_global ≥ score_min OU
      robustness_pct ≥ robustness_min OU
      majority_pct ≥ majority_min
    
    FLAGGED_FOR_REVIEW (ET logique) :
      score_global < score_max ET
      robustness_pct < robustness_max ET
      majority_pct < majority_max
    
    WIP[R0-closed] (par défaut) :
      Ni SURVIVES ni FLAGGED
    
    Returns:
        dict {'verdict': str, 'reason': str}
    """
    survives_thresholds = thresholds['survives']
    flagged_thresholds = thresholds['flagged']
    
    score_global = criteria['score_global']
    robustness_pct = criteria['robustness_pct']
    majority_pct = criteria['majority_pct']
    
    # Vérifier SURVIVES (OU logique)
    survives_conditions = [
        score_global >= survives_thresholds['score_min'],
        robustness_pct >= survives_thresholds['robustness_min'],
        majority_pct >= survives_thresholds['majority_min'],
    ]
    
    if any(survives_conditions):
        reasons = []
        if survives_conditions[0]:
            reasons.append(f"score={score_global:.1f}≥{survives_thresholds['score_min']}")
        if survives_conditions[1]:
            reasons.append(f"robustness={robustness_pct:.1f}%≥{survives_thresholds['robustness_min']}%")
        if survives_conditions[2]:
            reasons.append(f"majority={majority_pct:.1f}%≥{survives_thresholds['majority_min']}%")
        
        return {
            'verdict': 'SURVIVES[R0]',
            'reason': f"OU logique satisfait: {', '.join(reasons)}"
        }
    
    # Vérifier FLAGGED (ET logique)
    flagged_conditions = [
        score_global < flagged_thresholds['score_max'],
        robustness_pct < flagged_thresholds['robustness_max'],
        majority_pct < flagged_thresholds['majority_max'],
    ]
    
    if all(flagged_conditions):
        return {
            'verdict': 'FLAGGED_FOR_REVIEW',
            'reason': f"ET logique: score={score_global:.1f}<{flagged_thresholds['score_max']}, "
                     f"robustness={robustness_pct:.1f}%<{flagged_thresholds['robustness_max']}%, "
                     f"majority={majority_pct:.1f}%<{flagged_thresholds['majority_max']}%"
        }
    
    # Par défaut : WIP[R0-closed]
    return {
        'verdict': 'WIP[R0-closed]',
        'reason': f"Ni SURVIVES (score={score_global:.1f}, robustness={robustness_pct:.1f}%, "
                 f"majority={majority_pct:.1f}%) ni FLAGGED"
    }


def insert_gamma_verdict(conn, gamma_id: str, config_id: str, threshold_id: str,
                        criteria: Dict, verdict: Dict):
    """Insère verdict dans GammaVerdicts."""
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO GammaVerdicts (
            gamma_id, config_id, threshold_id,
            majority_pct, robustness_pct, score_global,
            verdict, verdict_reason,
            n_total_configs, n_pass_configs,
            n_viable_d_bases, n_total_d_bases,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gamma_id, config_id, threshold_id,
        criteria['majority_pct'],
        criteria['robustness_pct'],
        criteria['score_global'],
        verdict['verdict'],
        verdict['reason'],
        criteria['n_total_configs'],
        criteria['n_pass_configs'],
        criteria['n_viable_d_bases'],
        criteria['n_total_d_bases'],
        datetime.now().isoformat()
    ))
    
    conn.commit()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Runner R0 - Pipeline complet")
    
    # Modes (mutuellement exclusifs)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--brut', action='store_true',
                           help='Mode collecte données runs')
    mode_group.add_argument('--test', action='store_true',
                           help='Mode application tests + scoring')
    mode_group.add_argument('--verdict', action='store_true',
                           help='Mode calcul verdicts')
    mode_group.add_argument('--all', action='store_true',
                           help='Mode complet (brut + test + verdict)')
    
    # Arguments communs
    parser.add_argument('--gamma', type=str, required=True,
                       help='ID Γ (ex: GAM-001)')
    parser.add_argument('--config', type=str, default='weights_default',
                       help='ID config pondérations (défaut: weights_default)')
    parser.add_argument('--thresholds', type=str, default='thresholds_default',
                       help='ID config seuils (défaut: thresholds_default)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    # Exécution selon mode
    if args.brut:
        mode_brut(args.gamma, verbose=args.verbose)
    
    elif args.test:
        mode_test(args.gamma, args.config, verbose=args.verbose)
    
    elif args.verdict:
        mode_verdict(args.gamma, args.config, args.thresholds, verbose=args.verbose)
    
    elif args.all:
        # Chaîne complète
        print("\n" + "="*70)
        print("MODE --all : Chaîne complète")
        print("="*70 + "\n")
        
        mode_brut(args.gamma, verbose=args.verbose)
        mode_test(args.gamma, args.config, verbose=args.verbose)
        mode_verdict(args.gamma, args.config, args.thresholds, verbose=args.verbose)


if __name__ == "__main__":
    main()