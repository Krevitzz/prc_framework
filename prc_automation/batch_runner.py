#!/usr/bin/env python3
"""
Batch Runner V3 - Pipeline unifié sans db_raw.

ARCHITECTURE:
1. Discovery entities (inchangé)
2. Detect missing combinations (query db_results)
3. Execute unified (kernel + tests mémoire)
4. Generate reports (inchangé)

SUPPRESSIONS vs V2:
- Seeds (variance négligeable prouvée - analyze_seed_impact.py)
- db_raw (stockage intermédiaire 120 Go → 0 Go)
- BRANCH A/B (workflow unique)
- load_execution_context/history (mémoire directe)
- compress/decompress snapshots
- classify_new_files (plus de branches)
- get_exec_ids_for_gamma (plus de query db_raw)

CHANGEMENTS MAJEURS:
- Structure boucle : gamma × encoding × modifier × test (sans seeds)
- History complète en RAM (~60 Mo R0, ~500 Mo R1 par combo)
- Garbage collection agressive post-tests
- Détection manquants depuis db_results.observations

COMPATIBILITÉ:
- test_engine.py : INCHANGÉ (interface compatible)
- Schema db_results : INCHANGÉ (seed=42 legacy)
- Rapports/profiling : INCHANGÉS (consomment db_results)

Version: 3.0
Charter: 6.1 compliant
Author: PRC Framework
Date: 2025-02
"""

import argparse
import sys
import sqlite3
import json
import uuid  # Déjà importé (pas de changement)
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Imports PRC
from tests.utilities.utils.data_loading import discover_entities
from tests.utilities.HUB.test_engine import TestEngine
from tests.utilities.HUB.verdict_reporter import generate_verdict_report
from core.kernel import run_kernel
from core.state_preparation import prepare_state


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

MAX_ITERATIONS = 2000
SNAPSHOT_INTERVAL = 10

# Seed fixe (legacy schema SQL - clé primaire composite)
# Variance inter-seeds négligeable prouvée (analyze_seed_impact.py)
# TODO R2: Migration schema (suppression colonne seed)
SEED_FIXED = 42

# Dimension encodings (source: batch_runner.py:368)
# Compatible avec analyses existantes
# Configurable via CLI --n-dof
N_DOF_DEFAULT = 50

DB_DIR = Path("./prc_automation/prc_database")
CONFIG_DIR = Path("./tests/config/global") 

def get_db_path(phase: str) -> Path:
    """
    Retourne chemin db_results (db_raw supprimé).
    
    Args:
        phase: Phase cible ('R0', 'R1', etc.)
    
    Returns:
        Path db_results
    
    Examples:
        >>> get_db_path('R0')
        PosixPath('prc_automation/prc_database/prc_r0_results.db')
    """
    return DB_DIR / f"prc_{phase.lower()}_results.db"

# =============================================================================
# YAML PHASE DEFINITIONS
# =============================================================================

def load_phase_definition(phase: str) -> dict:
    """
    Charge définition phase depuis YAML.
    
    Args:
        phase: Phase cible ('R0', 'R1', etc.)
    
    Returns:
        dict: Définition phase
        {
            'name': str,
            'description': str,
            'entities': {
                'gamma': {'phase_filter': str},
                'encoding': {'phase_filter': str},
                'modifier': {'phase_filter': str},
                'test': {'phase_filter': str}
            },
            'database': str,
            'expected_combinations': int
        }
    
    Raises:
        FileNotFoundError: Si phase_definitions.yaml absent
        ValueError: Si phase non définie dans YAML
        yaml.YAMLError: Si YAML invalide
    
    Examples:
        >>> load_phase_definition('R1')
        {'name': 'Gamma Compositions', 'entities': {...}, ...}
    """
    config_path = CONFIG_DIR / 'phase_definitions.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration manquante: {config_path}\n"
            f"→ Créer phase_definitions.yaml dans {CONFIG_DIR}"
        )
    
    try:
        with open(config_path, encoding='utf-8') as f:
            definitions = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Erreur parsing YAML: {config_path}\n"
            f"Détails: {e}"
        )
    
    if phase not in definitions:
        available = list(definitions.keys())
        raise ValueError(
            f"Phase '{phase}' non définie dans {config_path}\n"
            f"Phases disponibles: {available}"
        )
    
    return definitions[phase]


def discover_for_phase(entity_type: str, phase_def: dict) -> List[Dict]:
    """
    Découvre entités selon définition phase.
    
    Args:
        entity_type: 'gamma', 'encoding', 'modifier', 'test'
        phase_def: Définition phase chargée depuis YAML
    
    Returns:
        List[Dict]: Entités découvertes
    
    Raises:
        KeyError: Si entity_type absent de phase_def
        RuntimeError: Si discovery retourne liste vide
    
    Examples:
        >>> phase_def = load_phase_definition('R1')
        >>> gammas = discover_for_phase('gamma', phase_def)
        >>> len(gammas)
        132  # Compositions
    """
    if entity_type not in phase_def['entities']:
        raise KeyError(
            f"Entity type '{entity_type}' absent de phase_def.\n"
            f"Disponibles: {list(phase_def['entities'].keys())}"
        )
    
    phase_filter = phase_def['entities'][entity_type]['phase_filter']
    
    entities = discover_entities(entity_type, phase=phase_filter)
    
    if len(entities) == 0:
        raise RuntimeError(
            f"Discovery retourne 0 entités pour {entity_type} "
            f"avec phase_filter='{phase_filter}'.\n"
            f"→ Vérifier métadonnées PHASE dans modules."
        )
    
    return entities

# =============================================================================
# DÉTECTION COMBINAISONS MANQUANTES
# =============================================================================

def detect_missing_combinations_v3(
    active_entities: Dict[str, List[Dict]],
    phase: str,
    db_results_path: Path = None
) -> List[Dict[str, str]]:
    """
    Détecte combinaisons (gamma × encoding × modifier) manquantes.
    
    CHANGEMENT V3:
    - Query db_results.observations (pas db_raw.executions)
    - Suppression seeds (1 seul combo par (gamma, encoding, modifier))
    - Validation applicabilité d_applicability
    
    LOGIQUE:
    1. Charger combos existantes depuis db_results
    2. Générer cartésien attendu avec validation applicabilité
    3. Calculer missing = attendu - existant
    
    Args:
        active_entities: Entités découvertes
            {
                'gammas': [{'id': 'GAM-001', 'metadata': {...}}, ...],
                'encodings': [{'id': 'SYM-003', ...}, ...],
                'modifiers': [{'id': 'M1', ...}, ...]
            }
        
        phase: Phase cible ('R0', 'R1', etc.)
        
        db_results_path: Chemin DB results (optionnel, déduit si None)
    
    Returns:
        Liste combinaisons manquantes:
        [
            {
                'gamma_id': 'GAM-001',
                'd_encoding_id': 'SYM-003',
                'modifier_id': 'M1'
            },
            ...
        ]
    
    Raises:
        FileNotFoundError: Si db_results absente
        sqlite3.Error: Si erreur query
    
    Notes:
        - Clé unique : (gamma_id, d_encoding_id, modifier_id, phase)
        - Combo partiellement exécutée (kernel OK, certains tests ERROR)
          sera considérée "présente" (limitation acceptable)
    
    Examples:
        >>> entities = {
        ...     'gammas': [{'id': 'GAM-001', 'metadata': {'d_applicability': ['SYM']}}],
        ...     'encodings': [{'id': 'SYM-003'}, {'id': 'R3-001'}],
        ...     'modifiers': [{'id': 'M0'}]
        ... }
        >>> missing = detect_missing_combinations_v3(entities, 'R0')
        >>> missing
        [{'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-003', 'modifier_id': 'M0'}]
        # R3-001 skippé (applicabilité)
    """
    # Déduire chemin DB si non fourni
    if db_results_path is None:
        db_results_path = get_db_path(phase)
    
    if not db_results_path.exists():
        raise FileNotFoundError(
            f"DB results manquante: {db_results_path}\n"
            f"→ Initialiser: python -m prc_automation.init_databases --phase {phase}"
        )
    
    # 1. Charger combinaisons existantes
    conn = sqlite3.connect(db_results_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT gamma_id, d_encoding_id, modifier_id
        FROM observations
        WHERE phase = ?
    """, (phase,))
    
    existing = set(cursor.fetchall())
    conn.close()
    
    # 2. Générer cartésien attendu (avec validation applicabilité)
    expected = set()
    skipped_incompatible = 0
    
    for gamma in active_entities['gammas']:
        gamma_id = gamma['id']
        gamma_metadata = gamma.get('metadata', {})
        d_applicability = gamma_metadata.get('d_applicability', [])
        
        for encoding in active_entities['encodings']:
            encoding_id = encoding['id']
            encoding_prefix = encoding_id.split('-')[0]  # 'SYM', 'ASY', 'R3'
            
            # Validation applicabilité
            if d_applicability and encoding_prefix not in d_applicability:
                skipped_incompatible += 1
                continue
            
            for modifier in active_entities['modifiers']:
                modifier_id = modifier['id']
                expected.add((gamma_id, encoding_id, modifier_id))
    
    # 3. Calculer missing
    missing_tuples = expected - existing
    
    # Logs
    print(f"   Attendues:           {len(expected)}")
    print(f"   Skippées (incomp.) : {skipped_incompatible}")
    print(f"   Existantes:          {len(existing)}")
    print(f"   Manquantes:          {len(missing_tuples)}")
    
    # Convertir tuples → list[dict]
    missing = [
        {
            'gamma_id': t[0],
            'd_encoding_id': t[1],
            'modifier_id': t[2]
        }
        for t in missing_tuples
    ]
    
    return missing


# =============================================================================
# EXÉCUTION KERNEL + CAPTURE HISTORY
# =============================================================================

def execute_kernel_with_history(
    combo: Dict[str, str],
    active_entities: Dict[str, List[Dict]],
    max_iterations: int = MAX_ITERATIONS,
    snapshot_interval: int = SNAPSHOT_INTERVAL,
    seed: int = SEED_FIXED,
    n_dof: int = N_DOF_DEFAULT
) -> Tuple[List[np.ndarray], str]:
    """
    Exécute kernel et capture history complète en mémoire.
    
    WORKFLOW:
    1. Résoudre modules gamma/encoding/modifier depuis active_entities
    2. Créer encoding (avec seed fixe et n_dof)
    3. Appliquer modifier
    4. Créer instance gamma (factory avec seed)
    5. Préparer état via prepare_state()
    6. Exécuter kernel avec capture snapshots à intervalle régulier
    7. Retourner history + exec_id UUID
    
    Args:
        combo: Combinaison à exécuter
            {'gamma_id': str, 'd_encoding_id': str, 'modifier_id': str}
        
        active_entities: Entités découvertes (pour résolution modules)
        
        max_iterations: Nombre max itérations kernel (défaut: 2000)
        
        snapshot_interval: Intervalle snapshots (défaut: 10)
            → 2000 / 10 = 200 snapshots
        
        seed: Seed fixe pour reproducibilité (défaut: 42)
        
        n_dof: Dimension encoding (défaut: 50, configurable)
    
    Returns:
        (history, exec_id):
        - history: List[np.ndarray] (200 snapshots pour max_iterations=2000)
        - exec_id: str (UUID unique pour traçabilité)
    
    Raises:
        ValueError: Si gamma/encoding/modifier introuvable
        ValueError: Si gamma incompatible avec encoding (applicabilité)
        RuntimeError: Si kernel crash
    
    Memory:
        R0 (n_dof=50, rank=2): ~2 Mo (200 × 50×50 × float64)
        R1 (n_dof=50, rank=3): ~20 Mo (200 × 50×50×50 × float64)
    
    Examples:
        >>> combo = {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-003', 'modifier_id': 'M1'}
        >>> history, exec_id = execute_kernel_with_history(combo, entities)
        >>> len(history)
        200
        >>> history[0].shape
        (50, 50)
    """
    # 1. Résolution modules
    try:
        gamma_entity = next(g for g in active_entities['gammas'] 
                           if g['id'] == combo['gamma_id'])
    except StopIteration:
        raise ValueError(f"Gamma introuvable: {combo['gamma_id']}")
    
    try:
        encoding_entity = next(e for e in active_entities['encodings'] 
                              if e['id'] == combo['d_encoding_id'])
    except StopIteration:
        raise ValueError(f"Encoding introuvable: {combo['d_encoding_id']}")
    
    try:
        modifier_entity = next(m for m in active_entities['modifiers'] 
                              if m['id'] == combo['modifier_id'])
    except StopIteration:
        raise ValueError(f"Modifier introuvable: {combo['modifier_id']}")
    
    # 2. Validation applicabilité
    gamma_metadata = gamma_entity.get('metadata', {})
    d_applicability = gamma_metadata.get('d_applicability', [])
    encoding_prefix = combo['d_encoding_id'].split('-')[0]
    
    if d_applicability and encoding_prefix not in d_applicability:
        raise ValueError(
            f"Incompatibilité: {combo['gamma_id']} (applicabilité={d_applicability}) "
            f"ne supporte pas {encoding_prefix}"
        )
    
    # 3. Créer encoding
    encoding_module = encoding_entity['module']
    encoding_func_name = encoding_entity['function_name']
    encoding_func = getattr(encoding_module, encoding_func_name)
    
    encoding_params = {'n_dof': n_dof}
    
    # 4. Créer modifier
    modifier_module = modifier_entity['module']
    modifier_func_name = modifier_entity['function_name']
    modifier_func = getattr(modifier_module, modifier_func_name)
    
    # 5. Préparer état final via prepare_state
    modifiers_list = [modifier_func]
    modifier_configs = {modifier_func: {}}  # Params default
    
    D_final = prepare_state(
        encoding_func=encoding_func,
        encoding_params=encoding_params,
        modifiers=modifiers_list,
        modifier_configs=modifier_configs,
        seed=seed
    )
    
    # Validation dimension encoding (dynamique)
    expected_rank = len(D_final.shape)
    if expected_rank == 2:
        assert D_final.shape == (n_dof, n_dof), \
            f"Shape R0 invalide: {D_final.shape}, attendu ({n_dof}, {n_dof})"
    elif expected_rank == 3:
        assert D_final.shape == (n_dof, n_dof, n_dof), \
            f"Shape R1 invalide: {D_final.shape}, attendu ({n_dof}, {n_dof}, {n_dof})"
    
    # 6. Créer gamma instance
    gamma_module = gamma_entity['module']
    factory_name = gamma_entity['function_name']
    factory = getattr(gamma_module, factory_name)
    gamma = factory(seed=seed)
    
    # Reset mémoire si non-markovien
    if hasattr(gamma, 'reset'):
        gamma.reset()
    
    # 7. Générer exec_id unique
    exec_id = str(uuid.uuid4())
    
    # 8. Exécuter kernel avec capture history
    history = []
    
    try:
        for iteration, state in run_kernel(
            D_final, gamma,
            max_iterations=max_iterations,
            record_history=False  # Ne pas utiliser record interne
        ):
            # Capture snapshot AVANT check explosion (legacy behavior)
            if iteration % snapshot_interval == 0:
                history.append(state.copy())
            
            # Détection explosion APRÈS snapshot (legacy behavior)
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"      ⚠ Explosion détectée iteration {iteration}")
                break  # Arrêt propre, snapshot déjà capté
    
    except Exception as e:
        raise RuntimeError(
            f"Kernel crash iteration {iteration} (snapshots captés: {len(history)}):\n"
            f"  Combo: {combo['gamma_id']} × {combo['d_encoding_id']} × {combo['modifier_id']}\n"
            f"  Erreur: {e}"
        ) from e
    
    # 9. Validation history finale
    # Note: Si explosion → break anticipé → history plus courte (NORMAL)
    # Iterations normales: 0, 10, 20, ..., 1990, 2000 → (2000/10) + 1 = 201 snapshots
    # Si break iteration 131 → ~14 snapshots (VALIDE)
    assert len(history) > 0, "History vide (aucun snapshot capté)"
    
    assert len(history) > 0, "History vide"
    
    # Cohérence interne (toutes shapes identiques)
    reference_shape = history[0].shape
    for i, snapshot in enumerate(history):
        assert snapshot.shape == reference_shape, \
            f"Shape snapshot {i} incohérente: {snapshot.shape} != {reference_shape}"
    
    return history, exec_id


# =============================================================================
# STOCKAGE OBSERVATION
# =============================================================================

def store_observation_v3(
    db_results_path: Path,
    observation: Dict[str, Any],
    phase: str
) -> None:
    """
    Stocke observation dans db_results.
    
    SCHEMA (conformité schema_results.sql):
    - Clé primaire composite : (test_name, gamma_id, d_encoding_id, 
                                 modifier_id, seed, phase)
    - Projections rapides : stat_initial, stat_final, stat_mean, stat_std,
                           evolution_slope, evolution_relative_change
    - Données complètes : observation_data (JSON)
    
    Args:
        db_results_path: Chemin DB results
        
        observation: Dict retourné par test_engine.execute_test()
            {
                'exec_id': str,
                'run_metadata': {
                    'gamma_id': str,
                    'd_encoding_id': str,
                    'modifier_id': str,
                    'seed': int
                },
                'test_name': str,
                'test_category': str,
                'status': 'SUCCESS' | 'ERROR' | 'NOT_APPLICABLE',
                'message': str,
                'statistics': {...},
                'evolution': {...},
                'config_params_id': str,
                ...
            }
        
        phase: Phase cible
    
    Returns:
        None
    
    Raises:
        ValueError: Si observation invalide (clés manquantes)
        sqlite3.Error: Si erreur DB
    
    Side-effects:
        - INSERT OR REPLACE dans observations
        - Commit immédiat
    
    Examples:
        >>> obs = {
        ...     'exec_id': 'abc-123',
        ...     'test_name': 'UNIV-001',
        ...     'run_metadata': {'gamma_id': 'GAM-001', ...},
        ...     'status': 'SUCCESS',
        ...     ...
        ... }
        >>> store_observation_v3(db_path, obs, 'R0')
    """
    # Validation clés obligatoires
    required_keys = ['exec_id', 'run_metadata', 'test_name', 'test_category', 
                     'status', 'config_params_id']
    for key in required_keys:
        if key not in observation:
            raise ValueError(f"Observation invalide: clé manquante '{key}'")
    
    # Validation status
    valid_statuses = {'SUCCESS', 'ERROR', 'NOT_APPLICABLE'}
    if observation['status'] not in valid_statuses:
        raise ValueError(f"Status invalide: {observation['status']}")
    
    # Extraction projections (première métrique si disponible)
    stats = observation.get('statistics', {})
    evolution = observation.get('evolution', {})
    
    first_metric = next(iter(stats.keys()), None)
    
    stat_initial = None
    stat_final = None
    stat_mean = None
    stat_std = None
    evolution_slope = None
    evolution_relative_change = None
    
    if first_metric:
        metric_stats = stats.get(first_metric, {})
        stat_initial = metric_stats.get('initial')
        stat_final = metric_stats.get('final')
        stat_mean = metric_stats.get('mean')
        stat_std = metric_stats.get('std')
        
        metric_evol = evolution.get(first_metric, {})
        evolution_slope = metric_evol.get('slope')
        evolution_relative_change = metric_evol.get('relative_change')
    
    # Connexion DB
    conn = sqlite3.connect(db_results_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO observations (
                test_name, gamma_id, d_encoding_id, modifier_id, seed, phase,
                exec_id, timestamp, test_category, params_config_id, status, message,
                observation_data,
                stat_initial, stat_final, stat_mean, stat_std,
                evolution_slope, evolution_relative_change
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation['test_name'],
            observation['run_metadata']['gamma_id'],
            observation['run_metadata']['d_encoding_id'],
            observation['run_metadata']['modifier_id'],
            observation['run_metadata']['seed'],
            phase,
            observation['exec_id'],
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
    
    except sqlite3.Error as e:
        conn.rollback()
        raise
    
    finally:
        conn.close()


# =============================================================================
# PIPELINE UNIFIÉ
# =============================================================================

def run_batch_unified_v3(
    missing: List[Dict[str, str]],
    active_entities: Dict[str, List[Dict]],
    active_tests: List[Dict],
    params_config_id: str,
    phase: str,
    n_dof: int = N_DOF_DEFAULT,
    verbose: bool = False
) -> None:
    """
    Pipeline unifié kernel + tests en mémoire.
    
    ARCHITECTURE:
    FOR combo IN missing:
        1. Exécuter kernel → history complète (RAM)
        2. FOR test IN active_tests:
            a. Exécuter test avec history
            b. Stocker observation db_results
        3. Garbage collection history (libère RAM)
    
    WORKFLOW:
    1. Pour chaque combo manquante:
       a. Exécuter kernel → history (~60 Mo R0, ~500 Mo R1)
       b. Générer exec_id unique
       c. Construire run_metadata
       d. Pour chaque test actif:
          - Exécuter test avec history complète
          - Stocker observation db_results
       e. Garbage collection history
    
    Args:
        missing: Combinaisons manquantes
            [
                {
                    'gamma_id': 'GAM-001',
                    'd_encoding_id': 'SYM-003',
                    'modifier_id': 'M1'
                },
                ...
            ]
        
        active_entities: Entités découvertes
            {
                'gammas': [...],
                'encodings': [...],
                'modifiers': [...],
                'tests': [...]
            }
        
        active_tests: Tests actifs (sous-ensemble active_entities['tests'])
        
        params_config_id: ID config params (ex: 'params_default_v1')
        
        phase: Phase cible ('R0', 'R1', etc.)
        
        n_dof: Dimension encodings (défaut: 50, configurable)
        
        verbose: Afficher traceback complet en cas d'erreur (défaut: False)
    
    Returns:
        None (side-effect: écrit dans db_results)
    
    Raises:
        ValueError: Si gamma/encoding/modifier introuvable
        RuntimeError: Si kernel crash
        sqlite3.Error: Si erreur DB
    
    Side-effects:
        - Écrit observations dans db_results
        - Alloue temporairement ~60 Mo RAM par combo (R0)
        - Logs progression stdout
    
    Examples:
        >>> missing = [{'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-003', 'modifier_id': 'M1'}]
        >>> run_batch_unified_v3(missing, entities, tests, 'params_default_v1', 'R0')
        [1/1] GAM-001 × SYM-003 × M1
          → Kernel: 2000 iterations... ✓ (200 snapshots)
          → Tests: 9/9 SUCCESS
        ✓ 1 combinaisons exécutées
    """
    if not missing:
        print("   Aucune combinaison manquante")
        return
    
    # Initialiser test engine
    test_engine = TestEngine()
    db_results_path = get_db_path(phase)
    
    print(f"\n   Exécution {len(missing)} combinaisons...\n")
    
    # Compteurs globaux
    total_observations = 0
    total_success = 0
    total_errors = 0
    failed_combos = []  # Track combos avec erreur kernel
    
    # Exécuter combos
    for idx, combo in enumerate(missing, 1):
        print(f"   [{idx}/{len(missing)}] {combo['gamma_id']} × "
              f"{combo['d_encoding_id']} × {combo['modifier_id']}")
        
        try:
            # 1. Exécuter kernel → history
            print(f"      → Kernel: {MAX_ITERATIONS} iterations...", end='', flush=True)
            
            history, exec_id = execute_kernel_with_history(
                combo,
                active_entities,
                max_iterations=MAX_ITERATIONS,
                snapshot_interval=SNAPSHOT_INTERVAL,
                seed=SEED_FIXED,
                n_dof=n_dof
            )
            
            print(f" ✓ ({len(history)} snapshots)")
            
            # 2. Construire run_metadata
            run_metadata = {
                'exec_id': exec_id,
                'gamma_id': combo['gamma_id'],
                'd_encoding_id': combo['d_encoding_id'],
                'modifier_id': combo['modifier_id'],
                'seed': SEED_FIXED,
                'phase': phase,
                'state_shape': history[0].shape  # Déduit de premier snapshot
            }
            
            # 3. Filtrer tests applicables (comportement legacy)
            from tests.utilities.utils.data_loading import check_applicability
            
            applicable_tests = []
            for test_info in active_tests:
                test_module = test_info['module']
                applicable, reason = check_applicability(test_module, run_metadata)
                if applicable:
                    applicable_tests.append(test_info)
            
            if not applicable_tests:
                print(f"      → Tests: 0/{len(active_tests)} applicables (skip)")
                del history  # GC même si skip
                continue
            
            print(f"      → {len(applicable_tests)}/{len(active_tests)} tests applicables")
            
            # 4. Exécuter tests applicables uniquement
            success_count = 0
            error_count = 0
            
            for test_info in applicable_tests:  # ← Seulement applicables
                test_module = test_info['module']
                
                try:
                    observation = test_engine.execute_test(
                        test_module=test_module,
                        run_metadata=run_metadata,
                        history=history,
                        params_config_id=params_config_id
                    )
                    
                    # Stocker observation
                    store_observation_v3(db_results_path, observation, phase)
                    
                    total_observations += 1
                    
                    if observation['status'] == 'SUCCESS':
                        success_count += 1
                        total_success += 1
                    elif observation['status'] == 'ERROR':
                        error_count += 1
                        total_errors += 1
                
                except Exception as e:
                    error_count += 1
                    total_errors += 1
                    print(f"      ✗ {test_info['id']}: {e}")
            
            print(f"      → Tests: {success_count}/{len(applicable_tests)} SUCCESS", end='')
            if error_count > 0:
                print(f", {error_count} ERROR")
            else:
                print()
            
            # 4. Garbage collection explicite
            del history
        
        except Exception as e:
            # Log erreur propre
            print(f"      ✗ Erreur combo: {e}")
            
            # Enregistrer combo échouée
            failed_combos.append({
                'combo': combo,
                'error': str(e)
            })
            
            # Traceback complet si mode verbose
            if verbose:
                import traceback
                traceback.print_exc()
            
            continue
    
    # Résumé final
    print(f"\n   {'='*66}")
    print(f"   RÉSUMÉ EXÉCUTION")
    print(f"   {'='*66}")
    print(f"   Combinaisons:    {len(missing)}")
    print(f"   Observations:    {total_observations}")
    print(f"   SUCCESS:         {total_success}")
    print(f"   ERROR:           {total_errors}")
    
    if failed_combos:
        print(f"   Combos crashées: {len(failed_combos)}")
        print(f"\n   Top 5 combos crashées:")
        for i, failure in enumerate(failed_combos[:5], 1):
            combo = failure['combo']
            error_msg = failure['error'].split('\n')[0][:80]  # Première ligne tronquée
            print(f"     {i}. {combo['gamma_id']} × {combo['d_encoding_id']} × {combo['modifier_id']}")
            print(f"        → {error_msg}...")
    
    print(f"   {'='*66}\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Batch Runner V3 - Pipeline unifié sans db_raw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Exécution R0 standard
  python -m prc_automation.batch_runner_v3 --phase R0
  
  # Avec configs custom
  python -m prc_automation.batch_runner_v3 --phase R0 \\
      --params params_custom_v1 --verdict verdict_strict_v1
  
  # Override dimension encodings
  python -m prc_automation.batch_runner_v3 --phase R0 --n-dof 100

Changements vs V2:
  - Suppression db_raw (économie 120 Go R0)
  - Suppression seeds (variance négligeable)
  - Workflow unifié (plus de BRANCH A/B)
  - History en mémoire (garbage collection agressive)
        """
    )
    
    parser.add_argument('--phase', default='R0',
                       help="Phase cible (défaut: R0)")
    
    parser.add_argument('--params', default='params_default_v1',
                       help="Params config ID (défaut: params_default_v1)")
    
    parser.add_argument('--verdict', default='verdict_default_v1',
                       help="Verdict config ID (défaut: verdict_default_v1)")
    
    parser.add_argument('--n-dof', type=int, default=N_DOF_DEFAULT,
                       help=f"Dimension encodings (défaut: {N_DOF_DEFAULT})")
    
    parser.add_argument('--verbose', action='store_true',
                       help="Afficher traceback complet en cas d'erreur (debugging)")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# BATCH RUNNER V3 - {args.phase}")
    print(f"# n_dof={args.n_dof}, seed={SEED_FIXED} (fixe)")
    print(f"{'#'*70}\n")
    
    try:
        # 1. Load phase definition + Discovery
        print("1. Chargement phase definition...")
        phase_def = load_phase_definition(args.phase)
        print(f"   ✓ Phase: {phase_def['name']}")
        print(f"   ✓ Description: {phase_def['description']}")

        print("\n2. Découverte entités...")
        active_entities = {
            'tests': discover_for_phase('test', phase_def),
            'gammas': discover_for_phase('gamma', phase_def),
            'encodings': discover_for_phase('encoding', phase_def),
            'modifiers': discover_for_phase('modifier', phase_def),
        }
        print(f"   ✓ Tests:     {len(active_entities['tests'])}")
        print(f"   ✓ Gammas:    {len(active_entities['gammas'])}")
        print(f"   ✓ Encodings: {len(active_entities['encodings'])}")
        print(f"   ✓ Modifiers: {len(active_entities['modifiers'])}")
        # 2. Detect missing
        print("\n3. Détection combinaisons manquantes...")
        missing = detect_missing_combinations_v3(active_entities, args.phase)
        
        # 3. Execute unified
        if len(missing) > 0:
            print("\n4. Exécution pipeline unifié...")
            run_batch_unified_v3(
                missing,
                active_entities,
                active_entities['tests'],
                args.params,
                args.phase,
                n_dof=args.n_dof,
                verbose=args.verbose
            )
        else:
            print("\n4. Exécution: SKIP (aucune combinaison manquante)")
        
        # 4. Generate reports
        print("\n5. Génération rapports...")
        
        # Vérifier observations existent
        db_results_path = get_db_path(args.phase)
        conn = sqlite3.connect(db_results_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM observations
            WHERE params_config_id = ? AND phase = ? AND status = 'SUCCESS'
        """, (args.params, args.phase))
        
        n_observations = cursor.fetchone()[0]
        conn.close()
        
        if n_observations == 0:
            print("   ⚠ Aucune observation SUCCESS trouvée")
            print("   → Skip génération rapports")
        else:
            print(f"   ✓ {n_observations} observations SUCCESS trouvées")
            
            results = generate_verdict_report(
                params_config_id=args.params,
                verdict_config_id=args.verdict
            )
            
            print(f"\n   {'='*66}")
            print(f"   RAPPORTS GÉNÉRÉS")
            print(f"   {'='*66}")
            print(f"   Répertoire: {Path(results['report_paths']['summary_global']).parent}")
            print(f"   Fichiers:   {len(results['report_paths'])}")
            print(f"   {'='*66}")
        
        print(f"\n{'#'*70}")
        print(f"# PIPELINE TERMINÉ")
        print(f"{'#'*70}\n")
    
    except FileNotFoundError as e:
        print(f"\n✗ ERREUR: {e}")
        print(f"   → Initialiser bases: python -m prc_automation.init_databases --phase {args.phase}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()