#!/usr/bin/env python3
"""
prc_automation/batch_runner_raw.py

Collecte de données brutes pour exploration R0.
Pas de tests, pas de verdicts - juste les observables.

Usage:
    python batch_runner_raw.py --gamma GAM-001
    python batch_runner_raw.py --gamma GAM-002 --save-snapshots
    python batch_runner_raw.py --init-db
"""

import sys
import argparse
import sqlite3
import json
import time
import gzip
import pickle
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Imports PRC
from core.kernel import run_kernel
from core.state_preparation import prepare_state
from operators import get_operator_by_id, OPERATOR_REGISTRY

# Modificateurs
from modifiers.noise import add_gaussian_noise, add_uniform_noise

# D^(base) generators
from D_encodings.rank2_symmetric import (
    create_identity, create_random_uniform, create_random_gaussian,
    create_correlation_matrix, create_banded, create_block_hierarchical
)
from D_encodings.rank2_asymmetric import (
    create_random_asymmetric, create_lower_triangular,
    create_antisymmetric, create_directional_gradient
)
from D_encodings.rank3_correlations import (
    create_random_rank3, create_partial_symmetric_rank3,
    create_local_coupling_rank3
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = Path("prc_database/prc_r0_raw.db")

# Catalogue D^(base)
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

# Modificateurs
MODIFIERS = {
    'M0': None,  # Base seule
    'M1': lambda seed: add_gaussian_noise(sigma=0.05, seed=seed),
    'M2': lambda seed: add_uniform_noise(amplitude=0.1, seed=seed),
    'M3': None,  # TODO: sparsification
}

SEEDS = [1, 2, 3, 4, 5]

MAX_ITERATIONS_DEFAULT = 2000
SNAPSHOT_INTERVAL = 10
CONVERGENCE_THRESHOLD = 1e-6
CONVERGENCE_WINDOW = 10


# ============================================================================
# BASE DE DONNÉES
# ============================================================================

def init_database():
    """Initialise la base de données."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Lire schema_raw.sql
    schema_path = Path("prc_database/schema_raw.sql")
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = f.read()
        conn.executescript(schema)
    else:
        print("❌ schema_raw.sql non trouvé")
        return
    
    conn.commit()
    conn.close()
    
    print(f"✓ Base de données initialisée: {DB_PATH}")


def execution_exists(conn, run_id: str) -> bool:
    """Vérifie si une exécution existe déjà."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Executions WHERE run_id = ?", (run_id,))
    return cursor.fetchone() is not None


def insert_execution(conn, exec_data: Dict) -> int:
    """
    Insère une exécution dans la DB.
    
    Tous les paramètres sont présents (NULL si non applicable).
    """
    cursor = conn.cursor()
    
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
        ) VALUES (
            ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?
        )
    """, (
        exec_data['run_id'], exec_data['timestamp'],
        exec_data['gamma_id'],
        exec_data.get('alpha'), exec_data.get('beta'), 
        exec_data.get('gamma_param'), exec_data.get('omega'),
        exec_data.get('memory_weight'), exec_data.get('window_size'), 
        exec_data.get('epsilon'),
        exec_data.get('sigma'), exec_data.get('lambda_param'),
        exec_data.get('eta'), exec_data.get('subspace_dim'),
        exec_data['d_base_id'], exec_data['modifier_id'], exec_data['seed'],
        exec_data['max_iterations'], exec_data['snapshot_interval'],
        exec_data['status'], exec_data.get('error_message'), 
        exec_data.get('final_iteration'), exec_data.get('execution_time'),
        exec_data.get('converged'), exec_data.get('convergence_iteration')
    ))
    
    exec_id = cursor.lastrowid
    conn.commit()
    return exec_id


def insert_metrics(conn, exec_id: int, metrics_list: List[Dict]):
    """Insère les métriques par itération."""
    cursor = conn.cursor()
    
    for metrics in metrics_list:
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
    
    conn.commit()


def insert_snapshots(conn, exec_id: int, snapshots: List[Dict]):
    """Insère les snapshots compressés."""
    cursor = conn.cursor()
    
    for snap in snapshots:
        # Comprimer l'état
        state_bytes = pickle.dumps(snap['state'])
        state_compressed = gzip.compress(state_bytes)
        
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


# ============================================================================
# CALCUL MÉTRIQUES
# ============================================================================

def compute_metrics(state: np.ndarray, previous_state: Optional[np.ndarray] = None) -> Dict:
    """
    Calcule métriques observables brutes pour un état.
    
    Returns:
        Dict avec toutes les métriques calculables
    """
    metrics = {}
    
    # Normes
    # Note: 'fro' uniquement pour rang 2, sinon norme euclidienne du flatten
    if state.ndim == 2:
        metrics['norm_frobenius'] = float(np.linalg.norm(state, 'fro'))
    else:
        # Pour rang ≠ 2 : norme euclidienne du vecteur aplati
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
        # Même logique : 'fro' pour rang 2, sinon norme euclidienne
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


def detect_convergence(distances: List[float], threshold: float, window: int) -> Optional[int]:
    """
    Détecte convergence simple : distance < threshold sur window itérations.
    
    Returns:
        Iteration de convergence ou None
    """
    if len(distances) < window:
        return None
    
    for i in range(len(distances) - window + 1):
        if all(d < threshold for d in distances[i:i+window]):
            return i
    
    return None


# ============================================================================
# EXTRACTION PARAMÈTRES
# ============================================================================

def extract_all_params(gamma_params: Dict) -> Dict:
    """
    Extrait TOUS les paramètres possibles, NULL si absent.
    
    Assure cohérence du schéma DB.
    """
    return {
        # Markoviens
        'alpha': gamma_params.get('alpha'),
        'beta': gamma_params.get('beta'),
        'gamma_param': gamma_params.get('gamma'),  # Évite conflit avec gamma_id
        'omega': gamma_params.get('omega'),
        
        # Non-markoviens
        'memory_weight': gamma_params.get('memory_weight'),
        'window_size': gamma_params.get('window_size'),
        'epsilon': gamma_params.get('epsilon'),
        
        # Stochastiques
        'sigma': gamma_params.get('sigma'),
        'lambda_param': gamma_params.get('lambda'),
        
        # Structurels
        'eta': gamma_params.get('eta'),
        'subspace_dim': gamma_params.get('subspace_dim'),
    }


def format_params_str(params: Dict) -> str:
    """Formate paramètres pour run_id."""
    # Prendre seulement paramètres non-None
    relevant = {k: v for k, v in params.items() if v is not None}
    
    if not relevant:
        return "default"
    
    # Format compact
    parts = []
    for k, v in sorted(relevant.items()):
        if isinstance(v, float):
            parts.append(f"{k}{v:.2f}")
        else:
            parts.append(f"{k}{v}")
    
    return "_".join(parts)


# ============================================================================
# EXÉCUTION SINGLE RUN
# ============================================================================

def execute_single_run(
    gamma_id: str,
    gamma_params: Dict,
    d_base_id: str,
    modifier_id: str,
    seed: int,
    save_snapshots: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Exécute 1 run et collecte données brutes.
    
    Returns:
        Dict avec toutes les données collectées
    """
    # Générer run_id
    params_str = format_params_str(gamma_params)
    run_id = f"{gamma_id}_{params_str}_{d_base_id}_{modifier_id}_s{seed}"
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"▶ RUN: {run_id}")
        print(f"{'─'*70}")
    
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
        distances = []
        previous_state = None
        converged = False
        convergence_iter = None
        
        for iteration, state in run_kernel(
            D_final, gamma,
            max_iterations=MAX_ITERATIONS_DEFAULT,
            record_history=False
        ):
            # Calculer métriques
            metrics = compute_metrics(state, previous_state)
            metrics['iteration'] = iteration
            metrics_list.append(metrics)
            
            # Collecter distances pour convergence
            if metrics['distance_to_previous'] is not None:
                distances.append(metrics['distance_to_previous'])
            
            # Sauvegarder snapshot si demandé
            if save_snapshots and iteration % SNAPSHOT_INTERVAL == 0:
                snap_metrics = compute_metrics(state)
                snap_metrics['iteration'] = iteration
                snap_metrics['state'] = state.copy()
                snapshots.append(snap_metrics)
            
            # Détection explosion
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                if verbose:
                    print(f"  ⚠ Explosion détectée: iter={iteration}")
                break
            
            # Détection convergence
            if not converged and len(distances) >= CONVERGENCE_WINDOW:
                conv_iter = detect_convergence(
                    distances, CONVERGENCE_THRESHOLD, CONVERGENCE_WINDOW
                )
                if conv_iter is not None:
                    converged = True
                    convergence_iter = conv_iter
                    if verbose:
                        print(f"  ✓ Convergence détectée: iter={convergence_iter}")
            
            previous_state = state
            
            # Progression périodique
            if verbose and iteration % 100 == 0:
                dist = metrics.get('distance_to_previous')
                dist_str = f"{dist:.2e}" if dist is not None else "None"
                print(f"  Iter {iteration}: norm={metrics['norm_frobenius']:.4f}, dist={dist_str}")
        
        final_iteration = iteration
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Terminé: {final_iteration} iterations, {execution_time:.1f}s")
            print(f"  Convergé: {converged}")
        
        # Préparer données pour DB
        all_params = extract_all_params(gamma_params)
        
        exec_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'gamma_id': gamma_id,
            **all_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'max_iterations': MAX_ITERATIONS_DEFAULT,
            'snapshot_interval': SNAPSHOT_INTERVAL,
            'status': 'COMPLETED',
            'error_message': None,
            'final_iteration': final_iteration,
            'execution_time': execution_time,
            'converged': converged,
            'convergence_iteration': convergence_iter,
        }
        
        return {
            'exec_data': exec_data,
            'metrics': metrics_list,
            'snapshots': snapshots if save_snapshots else [],
            'success': True,
        }
    
    except Exception as e:
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"  ❌ Erreur: {str(e)}")
            traceback.print_exc()
        
        all_params = extract_all_params(gamma_params)
        
        exec_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'gamma_id': gamma_id,
            **all_params,
            'd_base_id': d_base_id,
            'modifier_id': modifier_id,
            'seed': seed,
            'max_iterations': MAX_ITERATIONS_DEFAULT,
            'snapshot_interval': SNAPSHOT_INTERVAL,
            'status': 'ERROR',
            'error_message': str(e),
            'final_iteration': None,
            'execution_time': execution_time,
            'converged': False,
            'convergence_iteration': None,
        }
        
        return {
            'exec_data': exec_data,
            'metrics': [],
            'snapshots': [],
            'success': False,
        }


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_gamma(
    gamma_id: str,
    save_snapshots: bool = False,
    verbose: bool = True
):
    """
    Exécute tous les runs pour un Γ donné avec paramètres Phase 1.
    """
    print(f"\n{'='*70}")
    print(f"COLLECTE DONNÉES: {gamma_id}")
    print(f"{'='*70}\n")
    
    # Vérifier Γ
    if gamma_id not in OPERATOR_REGISTRY:
        print(f"❌ Γ inconnu: {gamma_id}")
        return
    
    info = OPERATOR_REGISTRY[gamma_id]
    if not info['implemented']:
        print(f"⏳ {gamma_id} non implémenté")
        return
    
    # Charger grille paramètres
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
    n_m = len([m for m in MODIFIERS.values() if m is not None or True])  # Tous
    n_seeds = len(SEEDS)
    n_total = n_params * n_d * n_m * n_seeds
    
    print(f"Configuration:")
    print(f"  Paramètres: {list(param_grid.keys())}")
    print(f"  D bases: {len(d_bases)} {list(d_bases.keys())}")
    print(f"  Modifiers: {list(MODIFIERS.keys())}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {n_total}")
    print(f"  Snapshots: {'OUI' if save_snapshots else 'NON'}")
    print()
    
    # Connexion DB
    if not DB_PATH.exists():
        print("❌ DB non initialisée. Lancer avec --init-db")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    completed = 0
    skipped = 0
    errors = 0
    
    # Boucles exécution
    for param_name, params in param_grid.items():
        for d_base_id in sorted(d_bases.keys()):
            for modifier_id in sorted(MODIFIERS.keys()):
                for seed in SEEDS:
                    
                    # Générer run_id pour vérifier
                    params_str = format_params_str(params)
                    run_id = f"{gamma_id}_{params_str}_{d_base_id}_{modifier_id}_s{seed}"
                    
                    # Skip si existe
                    if execution_exists(conn, run_id):
                        if verbose:
                            print(f"⏭  Skip (existant): {run_id}")
                        skipped += 1
                        continue
                    
                    # Exécuter
                    result = execute_single_run(
                        gamma_id, params, d_base_id, modifier_id, seed,
                        save_snapshots=save_snapshots,
                        verbose=verbose
                    )
                    
                    # Insérer dans DB
                    exec_id = insert_execution(conn, result['exec_data'])
                    
                    if result['success']:
                        insert_metrics(conn, exec_id, result['metrics'])
                        if save_snapshots:
                            insert_snapshots(conn, exec_id, result['snapshots'])
                        completed += 1
                    else:
                        errors += 1
                    
                    # Progression
                    total_proc = completed + skipped + errors
                    pct = 100 * total_proc / n_total
                    print(f"\n  Progrès: {total_proc}/{n_total} ({pct:.1f}%) "
                          f"[OK={completed}, Skip={skipped}, Err={errors}]")
    
    conn.close()
    
    # Résumé final
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ {gamma_id}")
    print(f"{'='*70}")
    print(f"Total runs: {n_total}")
    print(f"  Complétés: {completed}")
    print(f"  Sautés: {skipped}")
    print(f"  Erreurs: {errors}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Runner - Collecte données brutes")
    parser.add_argument('--gamma', type=str, required=False,
                       help='ID Γ (ex: GAM-001)')
    parser.add_argument('--init-db', action='store_true',
                       help='Initialiser base de données')
    parser.add_argument('--save-snapshots', action='store_true',
                       help='Sauvegarder snapshots états (consomme espace)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    # Init DB si demandé
    if args.init_db:
        init_database()
        return
    
    # Exécuter
    if not args.gamma:
        print("❌ Spécifier --gamma GAM-XXX")
        print("\nExemple:")
        print("  python batch_runner_raw.py --gamma GAM-001")
        print("  python batch_runner_raw.py --gamma GAM-002 --save-snapshots")
        return
    
    run_gamma(args.gamma, save_snapshots=args.save_snapshots, verbose=args.verbose)


if __name__ == "__main__":
    main()