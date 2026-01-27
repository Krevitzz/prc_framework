#!/usr/bin/env python3
"""
Batch Runner Charter PHASE 10 - Pipeline intelligent append-only.

Architecture:
1. Load registry
2. Discover entities (tous types)
3. Detect missing combinations
4. Classify nouveauté (branch A/B)
5. Execute (différentiel)
6. Update registry
7. Generate reports (toujours)

Usage:
    python -m prc_automation.batch_runner --phase R0
"""

import argparse
import sys
import sqlite3
import json
import gzip
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Literal

# Imports PRC
from tests.utilities.data_loading import (
    discover_entities,
    check_applicability,
    CriticalDiscoveryError
)
from tests.utilities.HUB.test_engine import TestEngine
from tests.utilities.HUB.verdict_reporter import generate_verdict_report
from prc_framework.core.kernel import run_kernel
from prc_framework.core.state_preparation import prepare_state


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]  # Grille seeds R0
MAX_ITERATIONS = 2000
SNAPSHOT_INTERVAL = 10

DB_DIR = Path("./prc_automation/prc_database")


def get_db_paths(phase: str = 'R0') -> Dict[str, Path]:
    """Retourne chemins bases pour phase."""
    return {
        'raw': DB_DIR / f"prc_{phase.lower()}_raw.db",
        'results': DB_DIR / f"prc_{phase.lower()}_results.db"
    }


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CriticalBatchError(Exception):
    """Exception batch nécessitant arrêt."""
    pass


# =============================================================================
# REGISTRY MANAGEMENT (voir ÉTAPE 4)
# =============================================================================

def load_execution_registry(phase: str = 'R0') -> Dict[str, Any]:
    """Charge registry JSON ou crée vide si absent."""
    registry_path = Path(f"./prc_automation/prc_database/execution_registry_{phase.lower()}.json")
    
    if not registry_path.exists():
        return {
            "version": "1.0",
            "phase": phase,
            "last_update": None,
            "loaded_files": {
                "gammas": [],
                "encodings": [],
                "modifiers": [],
                "tests": [],
                "configs": []
            },
            "counts": {
                "gammas": 0,
                "encodings": 0,
                "modifiers": 0,
                "tests": 0,
                "total_combinations": 0
            }
        }
    
    with open(registry_path, 'r') as f:
        return json.load(f)

def update_execution_registry(
    registry: Dict[str, Any],
    new_files: Dict[str, List[str]],
    phase: str = 'R0'
) -> None:
    """Met à jour registry avec nouveaux fichiers (merge conservatif)."""
    # Merge conservatif
    for entity_type, file_list in new_files.items():
        existing = set(registry['loaded_files'].get(entity_type, []))
        new_set = set(file_list)
        
        # Ajout nouveaux, conservation anciens (traçabilité)
        merged = existing.union(new_set)
        registry['loaded_files'][entity_type] = sorted(merged)
    
    # Update counts
    registry['counts'] = {
        'gammas': len(registry['loaded_files'].get('gammas', [])),
        'encodings': len(registry['loaded_files'].get('encodings', [])),
        'modifiers': len(registry['loaded_files'].get('modifiers', [])),
        'tests': len(registry['loaded_files'].get('tests', [])),
        'total_combinations': (
            len(registry['loaded_files'].get('gammas', [])) *
            len(registry['loaded_files'].get('encodings', [])) *
            len(registry['loaded_files'].get('modifiers', [])) *
            len(SEEDS)  # Constante globale
        )
    }
    
    # Update timestamp
    registry['last_update'] = datetime.now().isoformat()
    
    # Sauvegarder
    registry_path = Path(f"./prc_automation/prc_database/execution_registry_{phase.lower()}.json")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def classify_new_files(
    new_files: Dict[str, List],
    registry: Dict[str, Any]
) -> Literal['tests_only', 'causal_elements', 'none']:
    """
    Détermine branche exécution (A: tests seuls, B: éléments causaux).
    
    Args:
        new_files: Fichiers découverts par discover_entities()
        registry: Registry courant
    
    Returns:
        'tests_only': Nouveaux tests uniquement → skip brut
        'causal_elements': Nouveaux gamma/encoding/modifier → run brut
        'none': Aucun nouveau fichier
    """
    has_new_tests = len(new_files.get('tests', [])) > 0
    has_new_causal = (
        len(new_files.get('gammas', [])) > 0 or
        len(new_files.get('encodings', [])) > 0 or
        len(new_files.get('modifiers', [])) > 0
    )
    
    if has_new_causal:
        return 'causal_elements'  # Branch B (prioritaire)
    elif has_new_tests:
        return 'tests_only'       # Branch A
    else:
        return 'none'             # Skip brut/test
        
# =============================================================================
# DETECTION MISSING COMBINATIONS
# =============================================================================

def detect_missing_combinations(
    registry: Dict[str, Any],
    active_entities: Dict[str, List[Dict]],
    phase: str = 'R0'
) -> List[Dict[str, Any]]:
    """
    Détecte combinaisons non encore exécutées.
    
    Args:
        registry: Registry courant
        active_entities: Entités découvertes
        phase: Phase cible
    
    Returns:
        [
            {
                'gamma_id': 'GAM-001',
                'd_encoding_id': 'SYM-003',
                'modifier_id': 'M1',
                'seed': 42
            },
            ...
        ]
    """
    db_path = get_db_paths(phase)['raw']
    
    if not db_path.exists():
        raise FileNotFoundError(f"Base manquante: {db_path}")
    
    # Charger combinaisons existantes
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT gamma_id, d_encoding_id, modifier_id, seed
        FROM executions
        WHERE phase = ?
    """, (phase,))
    
    existing = set()
    for row in cursor.fetchall():
        existing.add((row[0], row[1], row[2], row[3]))
    
    conn.close()
    
    # Générer cartésien attendu
    expected = set()
    for gamma in active_entities['gammas']:
        for encoding in active_entities['encodings']:
            for modifier in active_entities['modifiers']:
                for seed in SEEDS:
                    expected.add((
                        gamma['id'],
                        encoding['id'],
                        modifier['id'],
                        seed
                    ))
    
    # Différence
    missing = expected - existing
    
    # Convertir en liste dicts
    missing_combinations = [
        {
            'gamma_id': combo[0],
            'd_encoding_id': combo[1],
            'modifier_id': combo[2],
            'seed': combo[3]
        }
        for combo in missing
    ]
    
    return missing_combinations


# =============================================================================
# EXECUTION KERNEL (MODE BRUT DIFFÉRENTIEL)
# =============================================================================

def run_batch_brut(
    missing_combinations: List[Dict],
    active_entities: Dict[str, List[Dict]],
    phase: str = 'R0'
) -> List[str]:
    """
    Execute kernel pour combinaisons manquantes (différentiel).
    
    Args:
        missing_combinations: Combinaisons à exécuter
        active_entities: Entités découvertes (pour résolution factories)
        phase: Phase cible
    
    Returns:
        Liste exec_ids créés
    """
    print(f"\n{'='*70}")
    print(f"MODE BRUT - Exécution kernel ({len(missing_combinations)} runs)")
    print(f"{'='*70}\n")
    
    db_path = get_db_paths(phase)['raw']
    conn = sqlite3.connect(db_path)
    
    # Index entités par ID
    gammas_by_id = {g['id']: g for g in active_entities['gammas']}
    encodings_by_id = {e['id']: e for e in active_entities['encodings']}
    modifiers_by_id = {m['id']: m for m in active_entities['modifiers']}
    
    exec_ids_created = []
    completed = 0
    errors = 0
    
    for i, combo in enumerate(missing_combinations, 1):
        gamma_id = combo['gamma_id']
        d_encoding_id = combo['d_encoding_id']
        modifier_id = combo['modifier_id']
        seed = combo['seed']
        
        print(f"[{i}/{len(missing_combinations)}] {gamma_id}_{d_encoding_id}_{modifier_id}_s{seed}")
        
        try:
            # Résoudre entités
            gamma_info = gammas_by_id[gamma_id]
            encoding_info = encodings_by_id[d_encoding_id]
            modifier_info = modifiers_by_id.get(modifier_id)
            
            # Créer D_base
            encoding_module = encoding_info['module']
            encoding_func_name = encoding_info['function_name']
            encoding_func = getattr(encoding_module, encoding_func_name)
            
            try:
                D_base = encoding_func(seed=seed)
            except TypeError:
                D_base = encoding_func()  # Pas de seed param
            
            # Appliquer modifier
            if modifier_id == 'M0':
                D_final = prepare_state(D_base, [])
            else:
                modifier_module = modifier_info['module']
                modifier_func_name = modifier_info['function_name']
                modifier_func = getattr(modifier_module, modifier_func_name)
                modifier = modifier_func(seed=seed)
                D_final = prepare_state(D_base, [modifier])
            
            # Créer gamma
            gamma_module = gamma_info['module']
            factory_name = f"create_gamma_hyp_{gamma_id.split('-')[1]}"
            factory = getattr(gamma_module, factory_name)
            gamma = factory()  # Params default pour R0
            
            # Reset mémoire si non-markovien
            if hasattr(gamma, 'reset'):
                gamma.reset()
            
            # Exécuter kernel
            snapshots = []
            previous_state = None
            
            for iteration, state in run_kernel(
                D_final, gamma,
                max_iterations=MAX_ITERATIONS,
                record_history=False
            ):
                # Sauvegarder snapshot
                if iteration % SNAPSHOT_INTERVAL == 0:
                    snap = {
                        'iteration': iteration,
                        'state': state.copy(),
                        'norm_frobenius': float(np.linalg.norm(state.flatten())),
                        'min_value': float(np.min(state)),
                        'max_value': float(np.max(state)),
                        'mean_value': float(np.mean(state)),
                        'std_value': float(np.std(state)),
                    }
                    
                    # Norme spectrale si rang 2 carré
                    if state.ndim == 2 and state.shape[0] == state.shape[1]:
                        try:
                            eigs = np.linalg.eigvalsh(state)
                            snap['norm_spectral'] = float(np.max(np.abs(eigs)))
                        except:
                            snap['norm_spectral'] = None
                    else:
                        snap['norm_spectral'] = None
                    
                    snapshots.append(snap)
                
                # Détection explosion
                if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                    print(f"  ⚠ Explosion détectée: iter={iteration}")
                    break
            
            final_iteration = iteration
            status = 'SUCCESS'
            
            # Insérer dans DB
            exec_id = insert_execution(
                conn, phase,
                gamma_id, d_encoding_id, modifier_id, seed,
                D_final.shape, final_iteration, status,
                snapshots
            )
            
            exec_ids_created.append(exec_id)
            completed += 1
            print(f"  ✓ exec_id={exec_id}, {final_iteration} iterations")
        
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            errors += 1
            
            # Insérer erreur
            exec_id = insert_execution(
                conn, phase,
                gamma_id, d_encoding_id, modifier_id, seed,
                (0,), 0, 'ERROR',
                [], error_message=str(e)
            )
            exec_ids_created.append(exec_id)
    
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ MODE BRUT")
    print(f"{'='*70}")
    print(f"Complétés: {completed}")
    print(f"Erreurs:   {errors}")
    print(f"{'='*70}\n")
    
    return exec_ids_created


def insert_execution(
    conn: sqlite3.Connection,
    phase: str,
    gamma_id: str,
    d_encoding_id: str,
    modifier_id: str,
    seed: int,
    state_shape: tuple,
    n_iterations: int,
    status: str,
    snapshots: List[Dict],
    error_message: str = None
) -> str:
    """
    Insère exécution dans db_raw.
    
    Returns:
        exec_id (UUID)
    """
    import uuid
    
    exec_id = str(uuid.uuid4())
    cursor = conn.cursor()
    
    # Insertion executions
    cursor.execute("""
        INSERT INTO executions (
            gamma_id, d_encoding_id, modifier_id, seed, phase,
            exec_id, timestamp, state_shape, n_iterations, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gamma_id, d_encoding_id, modifier_id, seed, phase,
        exec_id,
        datetime.now().isoformat(),
        json.dumps(list(state_shape)),
        n_iterations,
        status
    ))
    
    # Insertion snapshots
    for snap in snapshots:
        state_compressed = gzip.compress(pickle.dumps(snap['state']))
        
        cursor.execute("""
            INSERT INTO snapshots (
                exec_id, iteration, state_blob,
                norm_frobenius, norm_spectral,
                min_value, max_value, mean_value, std_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exec_id,
            snap['iteration'],
            state_compressed,
            snap['norm_frobenius'],
            snap.get('norm_spectral'),
            snap['min_value'],
            snap['max_value'],
            snap['mean_value'],
            snap['std_value']
        ))
    
    conn.commit()
    return exec_id


# =============================================================================
# APPLICATION TESTS (DIFFÉRENTIEL)
# =============================================================================

def run_batch_test(
    exec_ids: List[str],
    active_tests: List[Dict],
    params_config_id: str = 'params_default_v1',
    phase: str = 'R0'
) -> None:
    """
    Applique tests sur exec_ids ciblés (différentiel).
    
    Args:
        exec_ids: Exécutions à tester
        active_tests: Tests découverts
        params_config_id: Config params
        phase: Phase cible
    """
    print(f"\n{'='*70}")
    print(f"MODE TEST - Application tests ({len(exec_ids)} exec_ids)")
    print(f"{'='*70}\n")
    
    db_raw_path = get_db_paths(phase)['raw']
    db_results_path = get_db_paths(phase)['results']
    
    engine = TestEngine()
    
    total_observations = 0
    errors = 0
    
    for i, exec_id in enumerate(exec_ids, 1):
        print(f"[{i}/{len(exec_ids)}] exec_id={exec_id}")
        
        try:
            # Charger contexte
            context = load_execution_context(exec_id, db_raw_path)
            
            # Charger premier snapshot (state_shape)
            first_snapshot = load_first_snapshot(exec_id, db_raw_path)
            context['state_shape'] = first_snapshot.shape
            context['exec_id'] = exec_id
            
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
            
            # Charger history
            history = load_execution_history(exec_id, db_raw_path)
            
            # Appliquer tests
            for test_info in applicable_tests:
                test_module = test_info['module']
                
                try:
                    observation = engine.execute_test(
                        test_module, context, history, params_config_id
                    )
                    
                    # Stocker observation
                    store_test_observation(
                        db_results_path, phase,
                        context['gamma_id'],
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
    print(f"RÉSUMÉ MODE TEST")
    print(f"{'='*70}")
    print(f"Observations: {total_observations}")
    print(f"Erreurs:      {errors}")
    print(f"{'='*70}\n")


def load_execution_context(exec_id: str, db_path: Path) -> Dict:
    """Charge contexte depuis db_raw."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gamma_id, d_encoding_id, modifier_id, seed
        FROM executions
        WHERE exec_id = ?
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"exec_id={exec_id} non trouvé")
    
    return {
        'gamma_id': row['gamma_id'],
        'd_encoding_id': row['d_encoding_id'],
        'modifier_id': row['modifier_id'],
        'seed': row['seed']
    }


def load_first_snapshot(exec_id: str, db_path: Path):
    """Charge premier snapshot."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT state_blob
        FROM snapshots
        WHERE exec_id = ?
        ORDER BY iteration
        LIMIT 1
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"Aucun snapshot pour exec_id={exec_id}")
    
    state = pickle.loads(gzip.decompress(row[0]))
    return state


def load_execution_history(exec_id: str, db_path: Path) -> List:
    """Charge history complète."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT iteration, state_blob
        FROM snapshots
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    history = []
    for row in cursor.fetchall():
        state = pickle.loads(gzip.decompress(row[1]))
        history.append(state)
    
    conn.close()
    return history


def store_test_observation(
    db_path: Path,
    phase: str,
    gamma_id: str,
    d_encoding_id: str,
    modifier_id: str,
    seed: int,
    observation: Dict
) -> None:
    """Stocke observation dans db_results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extraire projections rapides (première métrique)
    stats = observation.get('statistics', {})
    first_metric = list(stats.keys())[0] if stats else None
    
    if first_metric:
        stat_data = stats[first_metric]
        stat_initial = stat_data.get('initial')
        stat_final = stat_data.get('final')
        stat_mean = stat_data.get('mean')
        stat_std = stat_data.get('std')
    else:
        stat_initial = stat_final = stat_mean = stat_std = None
    
    evol = observation.get('evolution', {})
    first_evol = list(evol.keys())[0] if evol else None
    
    if first_evol:
        evol_data = evol[first_evol]
        evolution_slope = evol_data.get('slope')
        evolution_relative_change = evol_data.get('relative_change')
    else:
        evolution_slope = evolution_relative_change = None
    
    cursor.execute("""
        INSERT OR REPLACE INTO observations (
            test_name, gamma_id, d_encoding_id, modifier_id, seed, phase,
            exec_id, timestamp, test_category, params_config_id,
            status, message, observation_data,
            stat_initial, stat_final, stat_mean, stat_std,
            evolution_slope, evolution_relative_change
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        observation['test_name'],
        gamma_id,
        d_encoding_id,
        modifier_id,
        seed,
        phase,
        observation.get('exec_id', ''),
        datetime.now().isoformat(),
        observation['test_category'],
        observation['config_params_id'],
        observation['status'],
        observation.get('message', ''),
        json.dumps(observation),
        stat_initial, stat_final, stat_mean, stat_std,
        evolution_slope, evolution_relative_change
    ))
    
    conn.commit()
    conn.close()


# =============================================================================
# GÉNÉRATION RAPPORTS
# =============================================================================

def run_batch_verdict(
    params_config_id: str = 'params_default_v1',
    verdict_config_id: str = 'verdict_default_v1',
    phase: str = 'R0'
) -> None:
    """
    Génère rapports (TOUJOURS exécuté).
    
    Args:
        params_config_id: Config params
        verdict_config_id: Config verdict
        phase: Phase cible
    """
    print(f"\n{'='*70}")
    print(f"MODE VERDICT - Génération rapports")
    print(f"{'='*70}\n")
    
    # Vérifier observations existent
    db_results_path = get_db_paths(phase)['results']
    conn = sqlite3.connect(db_results_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM observations
        WHERE params_config_id = ? AND phase = ? AND status = 'SUCCESS'
    """, (params_config_id, phase))
    
    n_observations = cursor.fetchone()[0]
    conn.close()
    
    if n_observations == 0:
        print(f"⚠ Aucune observation SUCCESS trouvée")
        print(f"   Skip génération rapports")
        return
    
    print(f"✓ {n_observations} observations trouvées\n")
    
    # Exécuter pipeline verdict
    try:
        results = generate_verdict_report(
            params_config_id=params_config_id,
            verdict_config_id=verdict_config_id
        )
        
        print(f"{'='*70}")
        print(f"RAPPORTS GÉNÉRÉS")
        print(f"{'='*70}")
        print(f"Répertoire: {Path(results['report_paths']['summary_global']).parent}")
        print(f"Fichiers:   {len(results['report_paths'])}")
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"✗ Erreur génération rapports: {e}")
        raise


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Runner PHASE 10 - Pipeline intelligent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Exécution R0 (détection automatique)
  python -m prc_automation.batch_runner --phase R0
  
  # Avec configs custom
  python -m prc_automation.batch_runner --phase R0 \\
      --params params_custom_v1 --verdict verdict_strict_v1
        """
    )
    
    parser.add_argument('--phase', default='R0',
                       help="Phase cible (défaut: R0)")
    
    parser.add_argument('--params', default='params_default_v1',
                       help="Params config ID")
    
    parser.add_argument('--verdict', default='verdict_default_v1',
                       help="Verdict config ID")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# BATCH RUNNER- {args.phase}")
    print(f"{'#'*70}\n")
    
    try:
        # 1. Load registry
        print("1. Chargement registry...")
        registry = load_execution_registry(args.phase)
        print(f"   ✓ Registry chargé ({registry['counts']['total_combinations']} combinaisons)")
        
        # 2. Discover entities
        print("\n2. Découverte entités...")
        active_entities = {
            'tests': discover_entities('test', args.phase),
            'gammas': discover_entities('gamma', args.phase),
            'encodings': discover_entities('encoding', args.phase),
            'modifiers': discover_entities('modifier', args.phase),
        }
        print(f"   ✓ Tests:     {len(active_entities['tests'])}")
        print(f"   ✓ Gammas:    {len(active_entities['gammas'])}")
        print(f"   ✓ Encodings: {len(active_entities['encodings'])}")
        print(f"   ✓ Modifiers: {len(active_entities['modifiers'])}")
        
        # 3. Detect missing
        print("\n3. Détection combinaisons manquantes...")
        missing = detect_missing_combinations(registry, active_entities, args.phase)
        print(f"   ✓ Manquantes: {len(missing)}")
        
        # 4. Classify
        print("\n4. Classification nouveauté...")
        
        # Extraire fichiers nouveaux
        new_files = {
            'tests': [t['module_path'] for t in active_entities['tests']
                     if t['module_path'] not in registry['loaded_files'].get('tests', [])],
            'gammas': [g['module_path'] for g in active_entities['gammas']
                      if g['module_path'] not in registry['loaded_files'].get('gammas', [])],
            'encodings': [e['module_path'] for e in active_entities['encodings']
                         if e['module_path'] not in registry['loaded_files'].get('encodings', [])],
            'modifiers': [m['module_path'] for m in active_entities['modifiers']
                         if m['module_path'] not in registry['loaded_files'].get('modifiers', [])],
        }
        
        classification = classify_new_files(new_files, registry)
        print(f"   ✓ Classification: {classification}")
        
        # 5. Execute
        if classification == 'none':
            print("\n5. Exécution: SKIP (aucun nouveau fichier)")
            exec_ids_all = []
        
        elif classification == 'tests_only':
            print("\n5. Exécution: BRANCH A (tests seuls)")
            # Charger tous exec_ids existants
            db_path = get_db_paths(args.phase)['raw']
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor
            
