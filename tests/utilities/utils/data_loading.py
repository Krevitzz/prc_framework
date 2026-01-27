# tests/utilities/data_loading.py
"""
Data Loading Utilities - Discovery + I/O Observations.

RESPONSABILITÉS (PHASE 10):
- Discovery unifiée (tests, gammas, encodings, modifiers)
- Validation applicabilité tests
- I/O observations (UNE SEULE connexion DB)
- Conversion observations → DataFrame normalisé
- Cache observations (futur)

ARCHITECTURE UNIFIÉE:
- discover_entities() : Point d'entrée unique tous types
- check_applicability() : Validation contraintes techniques
- load_all_observations() : Connexion unique db_results
- observations_to_dataframe() : Normalisation analyses stats

FUSION:
- discovery.py (tests)
- discovery_entities.py (legacy, supprimé)
- applicability.py (validators)
- data_loading.py (I/O)

Version: 2.0 (PHASE 10)
"""

import importlib
import inspect
import warnings
import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, List, Dict, Any, Tuple, Callable
from contextlib import contextmanager


# =============================================================================
# DISCOVERY UNIFIÉE (FUSION discovery.py + discovery_entities.py)
# =============================================================================

def discover_entities(
    entity_type: Literal['test', 'gamma', 'encoding', 'modifier'],
    phase: str = None
) -> List[Dict[str, Any]]:
    """
    Découvre entités actives d'un type donné.
    
    Args:
        entity_type: Type entité à découvrir
        phase: Filtre phase ('R0', 'R1', None=all)
    
    Returns:
        [
            {
                'id': 'GAM-001',
                'module_path': 'operators/gamma_hyp_001.py',
                'module': <module object>,
                'function_name': 'create_gamma_hyp_001',  # gammas uniquement
                'phase': 'R0',
                'metadata': {...}  # Type-specific
            },
            ...
        ]
    
    Raises:
        CriticalDiscoveryError: Si PHASE absent pour gamma/encoding/modifier
        ValidationError: Si structure module invalide
    
    Examples:
        >>> tests = discover_entities('test', phase='R0')
        >>> gammas = discover_entities('gamma', phase=None)
    """
    if entity_type == 'test':
        return _discover_tests(phase)
    elif entity_type == 'gamma':
        return _discover_gammas(phase)
    elif entity_type == 'encoding':
        return _discover_encodings(phase)
    elif entity_type == 'modifier':
        return _discover_modifiers(phase)
    else:
        raise ValueError(f"Type inconnu: {entity_type}")


def _discover_tests(phase: str = None) -> List[Dict]:
    """
    Découvre tests actifs (architecture 5.5 inchangée).
    
    Notes:
        - TEST_PHASE = None autorisé (applicable toutes phases)
        - Skip *_deprecated_*
        - Validation structure stricte (TEST_VERSION='5.5')
    """
    tests_dir = Path(__file__).parent.parent.parent
    test_files = tests_dir.glob('test_*.py')
    
    entities = []
    
    for test_file in test_files:
        # Skip deprecated
        if '_deprecated' in test_file.stem:
            continue
        
        # Import module
        module_name = f'tests.{test_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed import {module_name}: {e}")
            continue
        
        # Validate structure
        try:
            _validate_test_structure(module)
        except AssertionError as e:
            warnings.warn(f"Invalid structure {module_name}: {e}")
            continue
        
        # Extract phase
        test_phase = getattr(module, 'TEST_PHASE', None)
        metadata = {
            'test_id': module.TEST_ID,
            'category': module.TEST_CATEGORY,
            'version': module.TEST_VERSION,
            'weight': getattr(module, 'TEST_WEIGHT', 1.0),
            'applicability': module.APPLICABILITY_SPEC,
            'computation_specs': module.COMPUTATION_SPECS,
        }
        # Filter by phase
        if phase is not None and test_phase is not None and test_phase != phase:
            continue
        
        entities.append({
            'id': module.TEST_ID,
            'module_path': str(test_file),
            'module': module,
            'phase': test_phase,
            'metadata': metadata
        })
    
    return entities


def _discover_gammas(phase: str = None) -> List[Dict]:
    """
    Découvre gammas actifs (1 fichier = 1 gamma).
    
    Structure attendue:
        - PHASE = "R0" (OBLIGATOIRE)
        - METADATA = {'gamma_id': 'GAM-NNN', ...}
        - create_gamma_hyp_NNN() factory
    """
    operators_dir = Path(__file__).parent.parent.parent.parent / 'operators'
    gamma_files = operators_dir.glob('gamma_hyp_*.py')
    
    entities = []
    
    for gamma_file in gamma_files:
        if '_deprecated' in gamma_file.stem:
            continue
        
        module_name = f'operators.{gamma_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed import {module_name}: {e}")
            continue
        
        # ✅ Validation PHASE obligatoire
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing (OBLIGATOIRE)"
            )
        
        gamma_phase = module.PHASE
        
        # Filter by phase
        if phase is not None and gamma_phase != phase:
            continue
        
        # Extract metadata
        if not hasattr(module, 'METADATA'):
            warnings.warn(f"{module_name}: METADATA missing")
            continue
        
        metadata = module.METADATA
        gamma_id = metadata.get('id')
        
        if not gamma_id:
            warnings.warn(f"{module_name}: METADATA['id'] missing")
            continue
        
        # Find factory function (create_gamma_hyp_NNN)
        factory_name = f"create_{gamma_file.stem}"
        factory_func = getattr(module, factory_name, None)
        
        if factory_func is None:
            warnings.warn(f"{module_name}: {factory_name}() not found")
            continue
        
        entities.append({
            'id': gamma_id,
            'module_path': str(gamma_file),
            'module': module,
            'function_name': factory_name,
            'phase': gamma_phase,
            'metadata': metadata
        })
    
    return entities


def _discover_encodings(phase: str = None) -> List[Dict]:
    """
    Découvre encodings actifs (1 fichier = 1 encoding).
    
    Structure attendue:
        - Fichier: {sym,asy,r3}_NNN_descriptif.py
        - PHASE = "R0" (OBLIGATOIRE)
        - METADATA = {'id': 'XXX-NNN', 'rank': 2, ...}
        - create(n_dof, seed=None, **kwargs) → np.ndarray
    """
    encodings_dir = Path(__file__).parent.parent.parent.parent / 'D_encodings'
    
    # Pattern strict: {sym,asy,r3}_*.py
    encoding_files = list(encodings_dir.glob('sym_*.py'))
    encoding_files += list(encodings_dir.glob('asy_*.py'))
    encoding_files += list(encodings_dir.glob('r3_*.py'))
    
    entities = []
    
    for enc_file in encoding_files:
        if '_deprecated' in enc_file.stem:
            continue
        
        module_name = f'D_encodings.{enc_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed import {module_name}: {e}")
            continue
        
        # ✅ Validation PHASE obligatoire
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing (OBLIGATOIRE)"
            )
        
        enc_phase = module.PHASE
        
        # Filter by phase
        if phase is not None and enc_phase != phase:
            continue
        
        # Extract metadata
        if not hasattr(module, 'METADATA'):
            warnings.warn(f"{module_name}: METADATA missing")
            continue
        
        metadata = module.METADATA
        enc_id = metadata.get('id')
        
        if not enc_id:
            warnings.warn(f"{module_name}: METADATA['id'] missing")
            continue
        
        # Validate create() exists
        create_func = getattr(module, 'create', None)
        
        if create_func is None:
            warnings.warn(f"{module_name}: create() function not found")
            continue
        
        entities.append({
            'id': enc_id,
            'module_path': str(enc_file),
            'module': module,
            'function_name': 'create',
            'phase': enc_phase,
            'metadata': metadata
        })
    
    return entities


def _discover_modifiers(phase: str = None) -> List[Dict]:
    """
    Découvre modifiers actifs (1 fichier = 1 modifier).
    
    Structure attendue:
        - Fichier: m{N}_descriptif.py
        - PHASE = "R0" (OBLIGATOIRE)
        - METADATA = {'id': 'M{N}', 'type': '...', ...}
        - apply(state, seed=None, **kwargs) → np.ndarray
    """
    modifiers_dir = Path(__file__).parent.parent.parent.parent / 'modifiers'
    modifier_files = list(modifiers_dir.glob('m*.py'))
    
    entities = []
    
    for mod_file in modifier_files:
        if '_deprecated' in mod_file.stem or mod_file.stem == '__init__':
            continue
        
        module_name = f'modifiers.{mod_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed import {module_name}: {e}")
            continue
        
        # ✅ Validation PHASE obligatoire
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing (OBLIGATOIRE)"
            )
        
        mod_phase = module.PHASE
        
        # Filter by phase
        if phase is not None and mod_phase != phase:
            continue
        
        # Extract metadata
        if not hasattr(module, 'METADATA'):
            warnings.warn(f"{module_name}: METADATA missing")
            continue
        
        metadata = module.METADATA
        mod_id = metadata.get('id')
        
        if not mod_id:
            warnings.warn(f"{module_name}: METADATA['id'] missing")
            continue
        
        # Validate apply() exists
        apply_func = getattr(module, 'apply', None)
        
        if apply_func is None:
            warnings.warn(f"{module_name}: apply() function not found")
            continue
        
        entities.append({
            'id': mod_id,
            'module_path': str(mod_file),
            'module': module,
            'function_name': 'apply',
            'phase': mod_phase,
            'metadata': metadata
        })
    
    return entities


# =============================================================================
# VALIDATION (HELPERS)
# =============================================================================

def _validate_test_structure(module) -> None:
    """Valide structure test architecture 5.5."""
    REQUIRED_ATTRIBUTES = [
        'TEST_ID', 'TEST_CATEGORY', 'TEST_VERSION',
        'APPLICABILITY_SPEC', 'COMPUTATION_SPECS'
    ]
    
    for attr in REQUIRED_ATTRIBUTES:
        assert hasattr(module, attr), f"Missing: {attr}"
    
    assert isinstance(module.TEST_ID, str)
    assert isinstance(module.TEST_CATEGORY, str)
    assert isinstance(module.TEST_VERSION, str)
    assert module.TEST_VERSION == "5.5", f"TEST_VERSION must be '5.5'"
    
    import re
    assert re.match(r'^[A-Z]{3,4}-\d{3}$', module.TEST_ID), \
        f"TEST_ID invalid: {module.TEST_ID}"
    
    assert isinstance(module.APPLICABILITY_SPEC, dict)
    assert isinstance(module.COMPUTATION_SPECS, dict)
    assert 1 <= len(module.COMPUTATION_SPECS) <= 5
    
    for metric_name, spec in module.COMPUTATION_SPECS.items():
        assert 'registry_key' in spec
        assert 'default_params' in spec
        assert '.' in spec['registry_key']


# =============================================================================
# APPLICABILITY (FUSIONNÉ applicability.py)
# =============================================================================

VALIDATORS: Dict[str, Callable] = {
    'requires_rank': lambda run_metadata, expected: 
        expected is None or len(run_metadata['state_shape']) == expected,
    
    'requires_square': lambda run_metadata, required: 
        not required or (
            len(run_metadata['state_shape']) == 2 and 
            run_metadata['state_shape'][0] == run_metadata['state_shape'][1]
        ),
    
    'allowed_d_types': lambda run_metadata, allowed: 
        'ALL' in allowed or run_metadata['d_encoding_id'].split('-')[0] in allowed,
    
    'requires_even_dimension': lambda run_metadata, required: 
        not required or all(dim % 2 == 0 for dim in run_metadata['state_shape']),
    
    'minimum_dimension': lambda run_metadata, min_dim:
        min_dim is None or all(dim >= min_dim for dim in run_metadata['state_shape']),
}


def check_applicability(test_module, run_metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Vérifie applicabilité test (fusion applicability.py).
    
    Args:
        test_module: Module test
        run_metadata: {
            'gamma_id': str,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int,
            'state_shape': tuple
        }
    
    Returns:
        (applicable: bool, reason: str)
    
    Examples:
        >>> applicable, reason = check_applicability(test_module, run_metadata)
        >>> if not applicable:
        ...     print(f"Skip: {reason}")
    """
    spec = test_module.APPLICABILITY_SPEC
    
    for constraint_name, constraint_value in spec.items():
        if constraint_name not in VALIDATORS:
            return False, f"Contrainte inconnue: {constraint_name}"
        
        validator = VALIDATORS[constraint_name]
        
        try:
            is_valid = validator(run_metadata, constraint_value)
            
            if not is_valid:
                return False, f"{constraint_name}={constraint_value} non satisfait"
        
        except KeyError as e:
            return False, f"Info manquante: {e}"
        
        except Exception as e:
            return False, f"Erreur validation {constraint_name}: {e}"
    
    return True, ""


def add_validator(name: str, validator: Callable) -> None:
    """
    Ajoute validator custom.
    
    Args:
        name: Nom contrainte
        validator: Fonction (run_metadata, constraint_value) -> bool
    
    Raises:
        ValueError: Si validator déjà existant
    
    Examples:
        >>> def my_validator(run_metadata, value):
        ...     return run_metadata['d_encoding_id'].startswith('SYM')
        >>> add_validator('requires_symmetric', my_validator)
    """
    if name in VALIDATORS:
        raise ValueError(f"Validator '{name}' déjà existant")
    VALIDATORS[name] = validator


# =============================================================================
# I/O OBSERVATIONS (SIMPLIFIÉ - UNE SEULE CONNEXION)
# =============================================================================

@contextmanager
def db_connection(db_path: str):
    """
    Gestionnaire contexte DB (context manager).
    
    Args:
        db_path: Chemin base données
    
    Yields:
        Connection SQLite
    
    Examples:
        >>> with db_connection('./prc_r0_results.db') as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT COUNT(*) FROM observations")
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_all_observations(
    params_config_id: str,
    phase: str = 'R0',
    db_results_path: str = './prc_automation/prc_database/prc_r0_results.db'
) -> List[Dict]:
    """
    Charge observations SUCCESS (SIMPLIFIÉ - une seule connexion).
    
    CHANGEMENT PHASE 10:
    - UNE SEULE connexion (db_results uniquement)
    - gamma_id, d_encoding_id présents directement dans observations
    - Pas de JOIN avec db_raw
    
    Args:
        params_config_id: ID config params
        phase: Phase cible ('R0', 'R1', etc.)
        db_results_path: Chemin DB résultats
    
    Returns:
        List[dict]: Observations avec identité complète
        {
            'gamma_id': str,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int,
            'test_name': str,
            'test_category': str,
            'params_config_id': str,
            'observation_data': dict,
            'timestamp': str,
            'exec_id': str  # Traçabilité
        }
    
    Raises:
        ValueError: Si aucune observation SUCCESS
    
    Examples:
        >>> obs = load_all_observations('params_baseline', phase='R0')
        >>> len(obs)
        4320
        >>> obs[0].keys()
        dict_keys(['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', ...])
    """
    with db_connection(db_results_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                gamma_id,
                d_encoding_id,
                modifier_id,
                seed,
                test_name,
                test_category,
                params_config_id,
                observation_data,
                timestamp,
                exec_id,
                status
            FROM observations
            WHERE params_config_id = ?
              AND phase = ?
              AND status = 'SUCCESS'
        """, (params_config_id, phase))
        
        rows = cursor.fetchall()
    
    if not rows:
        raise ValueError(
            f"Aucune observation SUCCESS pour params={params_config_id}, phase={phase}"
        )
    
    observations = []
    for row in rows:
        try:
            obs_data = json.loads(row['observation_data'])
            
            observations.append({
                'gamma_id': row['gamma_id'],
                'd_encoding_id': row['d_encoding_id'],
                'modifier_id': row['modifier_id'],
                'seed': row['seed'],
                'test_name': row['test_name'],
                'test_category': row['test_category'],
                'params_config_id': row['params_config_id'],
                'observation_data': obs_data,
                'timestamp': row['timestamp'],
                'exec_id': row['exec_id'],
            })
        except (json.JSONDecodeError, KeyError) as e:
            warnings.warn(f"Skip observation: {e}")
            continue
    
    return observations


def observations_to_dataframe(observations: List[Dict]) -> pd.DataFrame:
    """
    Convertit observations → DataFrame normalisé pour analyses stats.
    
    PROJECTIONS EXTRAITES:
    - value_final, value_initial, value_mean, value_std, value_min, value_max
    - slope, volatility, relative_change
    - transition, trend (catégorielles)
    
    Args:
        observations: Liste observations (retour load_all_observations)
    
    Returns:
        DataFrame avec colonnes:
        - Identifiants: gamma_id, d_encoding_id, modifier_id, seed, 
                        test_name, params_config_id, metric_name
        - Projections numériques: value_*, slope, volatility, relative_change
        - Catégorielles: transition, trend
    
    Notes:
        - Filtre lignes avec NaN dans TOUTES projections numériques
        - Une ligne par (observation, metric)
    
    Examples:
        >>> df = observations_to_dataframe(obs)
        >>> df.columns
        Index(['gamma_id', 'test_name', 'value_final', 'slope', ...])
        >>> df.shape
        (8640, 17)  # 4320 obs × 2 métriques moyennes
    """
    rows = []
    
    for obs in observations:
        gamma_id = obs['gamma_id']
        d_encoding_id = obs['d_encoding_id']
        modifier_id = obs['modifier_id']
        seed = obs['seed']
        test_name = obs['test_name']
        params_config_id = obs['params_config_id']
        
        obs_data = obs['observation_data']
        
        if 'statistics' not in obs_data or 'evolution' not in obs_data:
            continue
        
        stats = obs_data['statistics']
        evolution = obs_data['evolution']
        
        for metric_name in stats.keys():
            if metric_name not in evolution:
                continue
            
            metric_stats = stats[metric_name]
            metric_evol = evolution[metric_name]
            
            rows.append({
                # Identifiants
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id,
                'seed': seed,
                'test_name': test_name,
                'params_config_id': params_config_id,
                'metric_name': metric_name,
                
                # Projections numériques
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                
                # Catégorielles
                'transition': metric_evol.get('transition', 'unknown'),
                'trend': metric_evol.get('trend', 'unknown'),
            })
    
    df = pd.DataFrame(rows)
    
    # Nettoyer NaN (lignes sans aucune projection valide)
    numeric_cols = [
        'value_final', 'value_initial', 'value_mean', 'value_std',
        'slope', 'volatility', 'relative_change'
    ]
    df = df.dropna(subset=numeric_cols, how='all')
    
    return df


def cache_observations(
    observations: List[Dict],
    cache_path: str = './cache/observations.pkl'
) -> None:
    """
    Cache observations sur disque (pickle).
    
    FUTUR: Optimisation chargement répété.
    
    Args:
        observations: Liste observations
        cache_path: Chemin cache
    
    Examples:
        >>> cache_observations(obs, './cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(observations, f)
    
    print(f"✓ Cache observations: {cache_path}")


def load_cached_observations(cache_path: str) -> List[Dict]:
    """
    Charge observations depuis cache.
    
    Args:
        cache_path: Chemin cache
    
    Returns:
        Liste observations
    
    Raises:
        FileNotFoundError: Si cache absent
    
    Examples:
        >>> obs = load_cached_observations('./cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache non trouvé: {cache_path}")
    
    with open(cache_file, 'rb') as f:
        observations = pickle.load(f)
    
    print(f"✓ Chargé cache: {cache_path} ({len(observations)} observations)")
    return observations


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CriticalDiscoveryError(Exception):
    """Exception découverte nécessitant arrêt."""
    pass


class ValidationError(Exception):
    """Exception validation structure."""
    pass