# tests/utilities/test_engine.py
"""
Test Engine Charter 5.4 - Génération observations pures.
"""

import numpy as np
import time
import traceback
from typing import Dict, Any, List

from .registries.registry_manager import RegistryManager
from .config_loader import get_loader


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
            
            execution_time = time.time() - start_time
            
            # Compiler résultats
            return self._compile_results(
                result, metric_buffers, skipped_iterations,
                computations, execution_time, common_params
            )
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = f"Erreur: {str(e)}"
            result['traceback'] = traceback.format_exc()
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
        params: dict
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