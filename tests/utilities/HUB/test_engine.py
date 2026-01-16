# tests/utilities/test_engine.py
"""
Test Engine Charter 5.4 - Génération observations pures.
"""

import numpy as np
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional

from .registries.registry_manager import RegistryManager
from .config_loader import get_loader


# =============================================================================
# NOUVEAU : DÉTECTION ÉVÉNEMENTS DYNAMIQUES
# =============================================================================

def detect_dynamic_events(values: np.ndarray) -> dict:
    """
    Détecte événements dynamiques sur trajectoire métrique.
    
    Événements R0 :
    - deviation_onset : |val - initial| > 0.1 * |initial|
    - instability_onset : |diff| > P90(diffs) * 10
    - oscillatory : nb sign_changes > 10% iterations
    - saturation : std(last_20%) / mean(last_20%) < 0.05
    - collapse : any(|last_10| < 1e-10) and max(|all|) > 1.0
    
    Args:
        values: array (n_iterations,) - trajectoire métrique
    
    Returns:
        {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool
        }
    """
    if len(values) < 2:
        return {
            'deviation_onset': None,
            'instability_onset': None,
            'oscillatory': False,
            'saturation': False,
            'collapse': False
        }
    
    # 1. Deviation onset (>10% initial)
    initial = values[0]
    deviations = np.abs(values - initial) / (np.abs(initial) + 1e-10)
    deviation_idx = np.where(deviations > 0.1)[0]
    deviation_onset = int(deviation_idx[0]) if len(deviation_idx) > 0 else None
    
    # 2. Instability onset (|diff| > P90 * 10)
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)
    threshold_instability = np.percentile(abs_diffs, 90) * 10
    instability_idx = np.where(abs_diffs > threshold_instability)[0]
    instability_onset = int(instability_idx[0]) if len(instability_idx) > 0 else None
    
    # 3. Oscillations (>10% sign changes)
    signs = np.sign(diffs)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    oscillatory = bool(sign_changes > len(values) * 0.1)
    
    # 4. Saturation (std(last_20%) / mean < 5%)
    last_20pct = max(int(len(values) * 0.2), 1)
    final_segment = values[-last_20pct:]
    saturation = bool((np.std(final_segment) / (np.abs(np.mean(final_segment)) + 1e-10)) < 0.05)
    
    # 5. Collapse (retour brutal à ~0)
    last_10 = values[-10:] if len(values) >= 10 else values
    collapse = bool(np.any(np.abs(last_10) < 1e-10) and np.max(np.abs(values)) > 1.0)
    
    return {
        'deviation_onset': deviation_onset,
        'instability_onset': instability_onset,
        'oscillatory': oscillatory,
        'saturation': saturation,
        'collapse': collapse
    }


def compute_event_sequence(
    events: dict,
    n_iterations: int
) -> dict:
    """
    Construit séquence ordonnée depuis événements.
    
    Calcule onsets RELATIFS (fraction durée totale) pour timelines.
    
    Args:
        events: Retour de detect_dynamic_events()
        n_iterations: Nombre total itérations (pour calcul relatif)
    
    Returns:
        {
            'sequence': ['deviation', 'instability'],
            'sequence_timing': [3, 7],
            'sequence_timing_relative': [0.015, 0.035],
            'saturation_onset_estimated': bool
        }
    """
    timed_events = []
    saturation_estimated = False
    
    # Événements avec onset ponctuel
    if events['deviation_onset'] is not None:
        onset_abs = events['deviation_onset']
        onset_rel = onset_abs / n_iterations
        timed_events.append(('deviation', onset_abs, onset_rel))
    
    if events['instability_onset'] is not None:
        onset_abs = events['instability_onset']
        onset_rel = onset_abs / n_iterations
        timed_events.append(('instability', onset_abs, onset_rel))
    
    # Saturation : onset estimé à 80% (heuristique R0)
    if events['saturation']:
        onset_abs = int(0.80 * n_iterations)
        onset_rel = 0.80
        timed_events.append(('saturation', onset_abs, onset_rel))
        saturation_estimated = True
    
    # Collapse : onset estimé à 90% (fin de run)
    if events['collapse']:
        onset_abs = int(0.90 * n_iterations)
        onset_rel = 0.90
        timed_events.append(('collapse', onset_abs, onset_rel))
    
    # Oscillatory : pas d'onset (comportement global)
    # Inséré si présent mais pas dans séquence temporelle
    
    # Trier par timing absolu
    timed_events.sort(key=lambda x: x[1])
    
    return {
        'sequence': [name for name, _, _ in timed_events],
        'sequence_timing': [timing_abs for _, timing_abs, _ in timed_events],
        'sequence_timing_relative': [timing_rel for _, _, timing_rel in timed_events],
        'saturation_onset_estimated': saturation_estimated,
        'oscillatory_global': events['oscillatory']
    }
def patch_execute_test_dynamic_events(
    metric_buffers: Dict[str, List[float]],
    n_iterations: int
) -> Tuple[dict, dict]:
    """
    Calcule dynamic_events + timeseries pour tous metrics.
    
    À insérer dans TestEngine.execute_test() après boucle itérations.
    
    Args:
        metric_buffers: {metric_name: [val_0, ..., val_N]}
        n_iterations: Nombre total itérations (len(history))
    
    Returns:
        (dynamic_events, timeseries)
        - dynamic_events: {metric_name: {events + sequence}}
        - timeseries: {metric_name: [val_0, ..., val_N]}
    """
    dynamic_events = {}
    timeseries = {}
    
    for metric_name, values in metric_buffers.items():
        # Stocker timeseries (optionnel, lourd)
        timeseries[metric_name] = list(values)
        
        if len(values) < 2:
            continue
        
        # Détecter événements
        events = detect_dynamic_events(np.array(values))
        
        # Calculer séquence + onsets relatifs
        seq_info = compute_event_sequence(events, n_iterations)
        
        # Fusionner
        dynamic_events[metric_name] = {
            **events,
            **seq_info
        }
    
    return dynamic_events, timeseries
    
    
class TestEngine:
    """
    Moteur exécution tests PRC 5.4.
    
    Responsabilités :
    1. Valider COMPUTATION_SPECS via RegistryManager
    2. Exécuter formules sur tous snapshots
    3. Appliquer post_processors
    4. Calculer statistics/evolution
    5. Retourner dict standardisé avec exec_id
    """
    
    VERSION = "5.5"
    
    def __init__(self):
        self.registry_manager = RegistryManager()
        self.computation_cache: Dict[str, Dict] = {}
        self.config_loader = get_loader()
    
    def execute_test(
        self,
        test_module,
        run_metadata: Dict[str, Any],
        history: List[np.ndarray],
        params_config_id: str
    ) -> Dict[str, Any]:
        """
        Exécute un test.
        
        Args:
            test_module: Module test importé
            run_metadata: {exec_id, gamma_id, d_encoding_id, modifier_id, seed, state_shape}
            history: Liste complète des snapshots
            params_config_id: ID config params
        
        Returns:
            Dict observation format 5.4 avec exec_id
        """
        result = self._init_result(test_module, run_metadata, params_config_id)
        
        try:
            # Charger params YAML
            params = self.config_loader.load(
                config_type='params',
                config_id=params_config_id,
                test_id=test_module.TEST_ID
            )
            
            if not params or not isinstance(params, dict):
                raise ValueError(f"Config {params_config_id} invalide")
            
            # Extraire params
            common_params = params.get('common', {})
            if not common_params:
                raise ValueError(f"Config {params_config_id} manque section 'common'")
            
            category = test_module.TEST_CATEGORY.lower()
            if category in params:
                common_params = {**common_params, **params[category]}
            
            # Préparer computations
            computations = self._prepare_computations(
                test_module.COMPUTATION_SPECS,
                params_config_id
            )
            
            if not computations:
                result['status'] = 'ERROR'
                result['message'] = 'Aucune spécification valide'
                if result['status'] == 'ERROR':
                    print(result.get('message'))
                    print(result.get('traceback'))
                return result
            
            # Buffers
            metric_buffers = {name: [] for name in computations.keys()}
            skipped_iterations = {}
            
            # Exécution
            start_time = time.time()
            
            for iteration, snapshot in enumerate(history):
                for metric_name, computation in computations.items():
                    try:
                        func = computation['function']
                        func_params = computation['params']
                        
                        raw_value = func(snapshot, **func_params)
                        
                        if computation['post_process']:
                            raw_value = computation['post_process'](raw_value)
                        
                        if not np.isfinite(raw_value):
                            raise ValueError(f"Valeur non finie: {raw_value}")
                        
                        metric_buffers[metric_name].append(float(raw_value))
                    
                    except Exception as e:
                        if metric_name not in skipped_iterations:
                            skipped_iterations[metric_name] = []
                        skipped_iterations[metric_name].append({
                            'iteration': iteration,
                            'error': str(e)
                        })
             # NOUVEAU : Calculer événements dynamiques + séquence
            n_iterations = len(history)           
            execution_time = time.time() - start_time
            dynamic_events, timeseries = patch_execute_test_dynamic_events(
                metric_buffers,
                n_iterations
            )
            

            # Compiler résultats
            return self._compile_results(
                result, metric_buffers, skipped_iterations,
                computations, execution_time, common_params,
                dynamic_events, timeseries  # ← AJOUTER
            )
            
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = f"Erreur: {str(e)}"
            result['traceback'] = traceback.format_exc()
            if result['status'] == 'ERROR':
                print(result.get('message'))
                print(result.get('traceback'))
            return result
            
            
    def _init_result(
        self,
        test_module,
        run_metadata: Dict,
        params_config_id: str
    ) -> Dict:
        """Initialise structure résultat avec exec_id."""
        return {
            # ⚠️ TRAÇABILITÉ COMPLÈTE
            'exec_id': run_metadata.get('exec_id'),
            
            'run_metadata': {
                'gamma_id': run_metadata['gamma_id'],
                'd_encoding_id': run_metadata['d_encoding_id'],
                'modifier_id': run_metadata['modifier_id'],
                'seed': run_metadata['seed'],
            },
            
            'test_name': test_module.TEST_ID,
            'test_category': test_module.TEST_CATEGORY,
            'test_version': test_module.TEST_VERSION,
            'config_params_id': params_config_id,
            
            'status': 'PENDING',
            'message': '',
            
            'statistics': {},
            'evolution': {},
            'dynamic_events': {},
            'timeseries': {},
            
            
            'metadata': {
                'engine_version': self.VERSION,
                'computations': {},
            }
        }
    
    def _prepare_computations(
        self,
        specs: Dict,
        config_id: str
    ) -> Dict:
        """Prépare et valide toutes les spécifications."""
        computations = {}
        
        for metric_name, spec in specs.items():
            cache_key = f"{config_id}_{metric_name}_{hash(str(spec))}"
            
            if cache_key in self.computation_cache:
                computations[metric_name] = self.computation_cache[cache_key]
                continue
            
            try:
                prepared = self.registry_manager.validate_computation_spec(spec)
                computations[metric_name] = prepared
                self.computation_cache[cache_key] = prepared
            
            except Exception as e:
                print(f"[TestEngine] Ignoré '{metric_name}': {e}")
                continue
        
        return computations
    
    def _compile_results(
        self,
        result: Dict,
        buffers: Dict,
        skipped: Dict,
        computations: Dict,
        exec_time: float,
        params: dict,
        dynamic_events: dict,  # ← NOUVEAU
        timeseries: dict       # ← NOUVEAU
        ) -> Dict:
        """Compile résultats finaux."""
        for metric_name, values in buffers.items():
            if len(values) < 2:
                continue
            
            # Statistics
            result['statistics'][metric_name] = {
                'initial': values[0],
                'final': values[-1],
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'q1': float(np.percentile(values, 25)),
                'q3': float(np.percentile(values, 75)),
                'n_valid': len(values),
            }
            
            # Evolution
            result['evolution'][metric_name] = self._analyze_evolution(values, params)

            # Metadata
            result['metadata']['computations'][metric_name] = {
                'registry_key': computations[metric_name]['registry_key'],
                'params_used': computations[metric_name]['params'],
                'has_post_process': computations[metric_name]['post_process'] is not None,
            }
        
        # Ajouter dynamic_events + timeseries
        result['dynamic_events'] = dynamic_events
        result['timeseries'] = timeseries  # Optionnel (lourd en stockage)
            
        result['metadata'].update({
            'execution_time_sec': exec_time,
            'num_iterations_processed': len(next(iter(buffers.values()), [])),
            'total_metrics': len(buffers),
            'successful_metrics': sum(1 for v in buffers.values() if len(v) >= 2),
            'skipped_iterations': skipped,
        })
        
        if not result['statistics']:
            result['status'] = 'ERROR'
            result['message'] = 'Aucune métrique valide'
        else:
            result['status'] = 'SUCCESS'
            total_skipped = sum(len(v) for v in skipped.values())
            if total_skipped > 0:
                result['message'] = f"SUCCESS avec {total_skipped} itérations sautées"
            else:
                result['message'] = 'SUCCESS'
        
        return result
    
    def _analyze_evolution(self, values: List[float], params: dict) -> Dict:
        """Analyse évolution série temporelle."""
        if len(values) < 2:
            return {'transition': 'insufficient_data', 'trend': 'unknown'}
        
        explosion_threshold = params.get('explosion_threshold', 1000.0)
        stability_tolerance = params.get('stability_tolerance', 0.1)
        growth_factor = params.get('growth_factor', 1.5)
        shrink_factor = params.get('shrink_factor', 0.5)
        epsilon = params.get('epsilon', 1e-10)
        
        # Tendance
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])
        
        if abs(slope) < epsilon:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Transition
        initial = values[0]
        final = values[-1]
        max_val = max(values)
        relative_change = abs(final - initial) / (abs(initial) + epsilon)
        
        if max_val > explosion_threshold:
            transition = "explosive"
        elif relative_change < stability_tolerance:
            transition = "stable"
        elif final > initial * growth_factor:
            transition = "growing"
        elif final < initial * shrink_factor:
            transition = "shrinking"
        else:
            transition = "oscillating"
        
        volatility = np.std(np.diff(values)) / (np.mean(np.abs(values)) + epsilon)
        
        return {
            'transition': transition,
            'trend': trend,
            'slope': slope,
            'volatility': float(volatility),
            'relative_change': float(relative_change),
        }
