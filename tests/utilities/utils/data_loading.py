# tests/utilities/utils/data_loading.py
"""
Data Loading Utilities - Discovery + I/O Observations.

RESPONSABILITÉS (PHASE 10 + R1):
- Discovery unifiée (tests, gammas, encodings, modifiers)
- R1 : Compositions gammas (inline, duck typing)
- Validation applicabilité tests
- I/O observations (UNE SEULE connexion DB)
- Conversion observations → DataFrame normalisé
- Cache observations (futur)

ARCHITECTURE UNIFIÉE:
- discover_entities() : Point d'entrée unique tous types
- check_applicability() : Validation contraintes techniques
- load_all_observations() : Connexion unique db_results
- observations_to_dataframe() : Normalisation analyses stats

R1 COMPOSITION:
- ComposedGamma : Classe inline (15 lignes)
- _CompositionModule : Duck typing pour batch_runner
- _generate_compositions_dynamic : Génération en mémoire (pas JSON)

Version: 2.1 (R1 COMPOSITION)
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
# R1 COMPOSITION - CLASSES INLINE
# =============================================================================

class ComposedGamma:
    """Composition séquentielle gammas (inline, 15 lignes)."""
    
    def __init__(self, sequence_gammas: List, seed: int = 42):
        self.sequence = sequence_gammas
        self.seed = seed
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Application séquentielle gamma1 → gamma2."""
        for gamma in self.sequence:
            state = gamma(state)
        return state
    
    def reset(self):
        """Reset mémoire gammas non-markoviens."""
        for gamma in self.sequence:
            if hasattr(gamma, 'reset'):
                gamma.reset()
    
    def __repr__(self):
        return f"ComposedGamma({len(self.sequence)} gammas)"


class _CompositionModule:
    """Module virtuel pour duck typing batch_runner."""
    
    def __init__(self, factory_func: Callable):
        self._factory = factory_func
    
    def __getattr__(self, name: str):
        """Intercepte getattr(module, 'create_...')."""
        if name.startswith('create_'):
            return self._factory
        raise AttributeError(f"Module virtuel: attribut '{name}' inexistant")


# =============================================================================
# DISCOVERY UNIFIÉE
# =============================================================================

def discover_entities(
    entity_type: Literal['test', 'gamma', 'encoding', 'modifier'],
    phase: str = None
) -> List[Dict[str, Any]]:
    """
    Découvre entités actives d'un type donné.
    
    MODIFICATION R1:
    - entity_type='gamma' + phase='R1' → compositions dynamiques
    - Autres types inchangés
    """
    if entity_type == 'test':
        return _discover_tests(phase)
    elif entity_type == 'gamma':
        return _discover_gammas_for_phase(phase)  # ← Nouveau routing
    elif entity_type == 'encoding':
        return _discover_encodings(phase)
    elif entity_type == 'modifier':
        return _discover_modifiers(phase)
    else:
        raise ValueError(f"Type inconnu: {entity_type}")


def _discover_gammas_for_phase(phase: str = None) -> List[Dict]:
    """
    Routing discovery gammas R0/R1.
    
    ARCHITECTURE:
    - R0 ou None: Atomiques uniquement
    - R1+: Compositions dynamiques uniquement
    """
    atomics = _discover_atomic_gammas()
    
    if phase == 'R0' or phase is None:
        return atomics
    else:
        return _generate_compositions_dynamic(atomics, phase)


def _discover_atomic_gammas() -> List[Dict]:
    """Découvre gammas atomiques (gamma_hyp_*.py avec PHASE='R0')."""
    # Chemin operators/ adaptatif (pour tests)
    # Si operators/ existe dans parent immédiat → utiliser
    # Sinon → utiliser chemin projet standard
    operators_dir_test = Path(__file__).parent / 'operators'
    operators_dir_project = Path(__file__).parent.parent.parent.parent / 'operators'
    
    if operators_dir_test.exists():
        operators_dir = operators_dir_test
    else:
        operators_dir = operators_dir_project
    
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
        
        # Validation PHASE obligatoire
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing (OBLIGATOIRE)"
            )
        
        gamma_phase = module.PHASE
        
        # Filter atomiques R0 uniquement
        if gamma_phase != 'R0':
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
        
        # Find factory function
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


def _generate_compositions_dynamic(atomics: List[Dict], phase: str) -> List[Dict]:
    """
    Génère compositions dynamiques EN MÉMOIRE (pas de JSON).
    
    Duck typing:
        batch_runner fait:
            factory = getattr(module, 'create_composed_gamma')
            gamma = factory(seed=seed)
        
        _CompositionModule intercepte getattr → batch_runner aveugle!
    """
    from itertools import permutations
    
    compositions = []
    atomic_ids = [g['id'] for g in atomics]
    
    for gamma1_id, gamma2_id in permutations(atomic_ids, 2):
        comp_id = f"{gamma1_id}x{gamma2_id}"
        
        # Résoudre entities atomiques
        g1_entity = next((g for g in atomics if g['id'] == gamma1_id), None)
        g2_entity = next((g for g in atomics if g['id'] == gamma2_id), None)
        
        if not g1_entity or not g2_entity:
            warnings.warn(f"Composition {comp_id}: gamma manquant")
            continue
        
        # Factory closure (capture e1, e2)
        def make_factory(e1: Dict, e2: Dict) -> Callable:
            def factory(seed: int = 42) -> ComposedGamma:
                """Signature identique gammas atomiques."""
                # Instancier atomiques
                f1 = getattr(e1['module'], e1['function_name'])
                f2 = getattr(e2['module'], e2['function_name'])
                
                gamma1 = f1(seed=seed)
                gamma2 = f2(seed=seed)
                
                return ComposedGamma([gamma1, gamma2], seed=seed)
            
            return factory
        
        comp_factory = make_factory(g1_entity, g2_entity)
        virtual_module = _CompositionModule(comp_factory)

        # Calculer d_applicability par intersection des parents
        # Principe: composition applicable SI tous les gammas de la séquence le sont
        d_app_1 = set(g1_entity['metadata'].get('d_applicability', []))
        d_app_2 = set(g2_entity['metadata'].get('d_applicability', []))

        if not d_app_1 and not d_app_2:
            # Aucun parent n'a de restrictions → composition universelle
            d_app_composed = []
        elif not d_app_1:
            # Parent 1 universel → restrictions de parent 2
            d_app_composed = list(d_app_2)
        elif not d_app_2:
            # Parent 2 universel → restrictions de parent 1
            d_app_composed = list(d_app_1)
        else:
            # Intersection stricte (plus restrictif des deux)
            d_app_composed = list(d_app_1 & d_app_2)

        compositions.append({
            'id': comp_id,
            'module_path': 'dynamic',
            'module': virtual_module,
            'function_name': 'create_composed_gamma',
            'phase': phase,
            'metadata': {
                'composition_id': comp_id,
                'sequence_gammas': [gamma1_id, gamma2_id],
                'type': 'composed',
                'n': 2,
                'd_applicability': d_app_composed  # ← CALCULÉ !
            }
        })
    return compositions


def _discover_tests(phase: str = None) -> List[Dict]:
    """Découvre tests actifs (architecture 5.5 inchangée)."""
    tests_dir = Path(__file__).parent.parent.parent
    test_files = tests_dir.glob('test_*.py')
    
    entities = []
    
    for test_file in test_files:
        if '_deprecated' in test_file.stem:
            continue
        
        module_name = f'tests.{test_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed import {module_name}: {e}")
            continue
        
        try:
            _validate_test_structure(module)
        except AssertionError as e:
            warnings.warn(f"Invalid structure {module_name}: {e}")
            continue
        
        test_phase = getattr(module, 'TEST_PHASE', None)
        metadata = {
            'test_id': module.TEST_ID,
            'category': module.TEST_CATEGORY,
            'version': module.TEST_VERSION,
            'weight': getattr(module, 'TEST_WEIGHT', 1.0),
            'applicability': module.APPLICABILITY_SPEC,
            'computation_specs': module.COMPUTATION_SPECS,
        }
        
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


def _discover_encodings(phase: str = None) -> List[Dict]:
    """Découvre encodings actifs."""
    encodings_dir = Path(__file__).parent.parent.parent.parent / 'D_encodings'
    
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
        
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing"
            )
        
        enc_phase = module.PHASE
        
        if phase is not None and enc_phase != phase:
            continue
        
        if not hasattr(module, 'METADATA'):
            warnings.warn(f"{module_name}: METADATA missing")
            continue
        
        metadata = module.METADATA
        enc_id = metadata.get('id')
        
        if not enc_id:
            warnings.warn(f"{module_name}: METADATA['id'] missing")
            continue
        
        create_func = getattr(module, 'create', None)
        
        if create_func is None:
            warnings.warn(f"{module_name}: create() not found")
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
    """Découvre modifiers actifs."""
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
        
        if not hasattr(module, 'PHASE'):
            raise CriticalDiscoveryError(
                f"{module_name}: PHASE attribute missing"
            )
        
        mod_phase = module.PHASE
        
        if phase is not None and mod_phase != phase:
            continue
        
        if not hasattr(module, 'METADATA'):
            warnings.warn(f"{module_name}: METADATA missing")
            continue
        
        metadata = module.METADATA
        mod_id = metadata.get('id')
        
        if not mod_id:
            warnings.warn(f"{module_name}: METADATA['id'] missing")
            continue
        
        apply_func = getattr(module, 'apply', None)
        
        if apply_func is None:
            warnings.warn(f"{module_name}: apply() not found")
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
# VALIDATION
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
    assert module.TEST_VERSION == "5.5"
    
    import re
    assert re.match(r'^[A-Z]{3,4}-\d{3}$', module.TEST_ID)
    
    assert isinstance(module.APPLICABILITY_SPEC, dict)
    assert isinstance(module.COMPUTATION_SPECS, dict)
    assert 1 <= len(module.COMPUTATION_SPECS) <= 5
    
    for metric_name, spec in module.COMPUTATION_SPECS.items():
        assert 'registry_key' in spec
        assert 'default_params' in spec
        assert '.' in spec['registry_key']


# =============================================================================
# APPLICABILITY
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
    """Vérifie applicabilité test."""
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
    """Ajoute validator custom."""
    if name in VALIDATORS:
        raise ValueError(f"Validator '{name}' déjà existant")
    VALIDATORS[name] = validator


# =============================================================================
# I/O OBSERVATIONS
# =============================================================================

@contextmanager
def db_connection(db_path: str):
    """Gestionnaire contexte DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_all_observations(
    params_config_id: str,
    phase: str = 'R1',
    db_results_path: str = './prc_automation/prc_database/prc_r1_results.db'
) -> List[Dict]:
    """Charge observations SUCCESS (une seule connexion)."""
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
    """Convertit observations → DataFrame normalisé."""
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
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id,
                'seed': seed,
                'test_name': test_name,
                'params_config_id': params_config_id,
                'metric_name': metric_name,
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                'transition': metric_evol.get('transition', 'unknown'),
                'trend': metric_evol.get('trend', 'unknown'),
            })
    
    df = pd.DataFrame(rows)
    
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
    """Cache observations sur disque."""
    import pickle
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(observations, f)
    
    print(f"✓ Cache observations: {cache_path}")


def load_cached_observations(cache_path: str) -> List[Dict]:
    """Charge observations depuis cache."""
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
