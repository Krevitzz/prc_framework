# prc_framework/utilities/test_engine.py

import numpy as np
import time
import traceback
from typing import Dict, Any, List

from .registries.registry_manager import RegistryManager

class TestEngine:
    """
    Moteur exécution tests PRC 5.4.
    
    Responsabilités :
    1. Valider COMPUTATION_SPECS via RegistryManager
    2. Exécuter formules sur tous snapshots
    3. Appliquer post_processors
    4. Calculer statistics/evolution
    5. Retourner dict standardisé
    """
    
    VERSION = "5.4"
    
    def __init__(self):
        self.registry_manager = RegistryManager()
        self.computation_cache: Dict[str, Dict] = {}
    
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
            run_metadata: {gamma_id, d_base_id, modifier_id, seed, state_shape}
            history: Liste complète des snapshots (~200)
            params_config_id: ID config params
        
        Returns:
            Dict format standardisé 5.4
        """
        result = self._init_result(test_module, run_metadata, params_config_id)
        
        try:
            # Charger params YAML
            yaml_params = self._load_yaml_params(test_module.TEST_ID, params_config_id)
            
            # Préparer computations
            computations = self._prepare_computations(
                test_module.COMPUTATION_SPECS,
                yaml_params
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
                        # Exécuter fonction registre
                        func = computation['function']
                        params = computation['params']
                        
                        raw_value = func(snapshot, **params)
                        
                        # Post-process
                        if computation['post_process']:
                            raw_value = computation['post_process'](raw_value)
                        
                        # Validation
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
                computations, execution_time
            )
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['message'] = f"Erreur: {str(e)}"
            result['traceback'] = traceback.format_exc()
            return result
    
    def _init_result(self, test_module, run_metadata, params_config_id) -> Dict:
        """Initialise structure résultat."""
        return {
            # Traçabilité
            'run_metadata': {
                'gamma_id': run_metadata['gamma_id'],
                'd_base_id': run_metadata['d_base_id'],
                'modifier_id': run_metadata['modifier_id'],
                'seed': run_metadata['seed'],
            },
            
            # Identification test
            'test_name': test_module.TEST_ID,
            'test_category': test_module.TEST_CATEGORY,
            'test_version': test_module.TEST_VERSION,
            'config_params_id': params_config_id,
            
            # Status
            'status': 'PENDING',
            'message': '',
            
            # Résultats
            'statistics': {},
            'evolution': {},
            
            # Metadata
            'metadata': {
                'engine_version': self.VERSION,
                'computations': {},
            }
        }
    
    def _prepare_computations(self, specs, yaml_params) -> Dict:
        """Prépare et valide toutes les spécifications."""
        computations = {}
        
        for metric_name, spec in specs.items():
            cache_key = f"{metric_name}_{hash(str(spec))}"
            
            if cache_key in self.computation_cache:
                computations[metric_name] = self.computation_cache[cache_key]
                continue
            
            # Fusionner avec YAML
            merged_spec = spec.copy()
            if metric_name in yaml_params:
                merged_spec['default_params'] = {
                    **spec.get('default_params', {}),
                    **yaml_params[metric_name]
                }
                
                if 'post_process' in yaml_params[metric_name]:
                    merged_spec['post_process'] = yaml_params[metric_name]['post_process']
            
            # Valider
            try:
                prepared = self.registry_manager.validate_computation_spec(merged_spec)
                computations[metric_name] = prepared
                self.computation_cache[cache_key] = prepared
            
            except Exception as e:
                print(f"[TestEngine] Ignoré '{metric_name}': {e}")
                continue
        
        return computations
    
    def _compile_results(self, result, buffers, skipped, computations, exec_time) -> Dict:
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
            result['evolution'][metric_name] = self._analyze_evolution(values)
            
            # Metadata métrique
            result['metadata']['computations'][metric_name] = {
                'registry_key': computations[metric_name]['registry_key'],
                'params_used': computations[metric_name]['params'],
                'has_post_process': computations[metric_name]['post_process'] is not None,
            }
        
        # Metadata globale
        result['metadata'].update({
            'execution_time_sec': exec_time,
            'num_iterations_processed': len(next(iter(buffers.values()), [])),
            'total_metrics': len(buffers),
            'successful_metrics': sum(1 for v in buffers.values() if len(v) >= 2),
            'skipped_iterations': skipped,
        })
        
        # Status final
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
    
    def _analyze_evolution(self, values: List[float]) -> Dict:
        """Analyse évolution série temporelle."""
        if len(values) < 2:
            return {'transition': 'insufficient_data', 'trend': 'unknown'}
        
        # Tendance
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])
        
        if abs(slope) < 1e-10:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Transition
		#\todo 0.1 1.5 0.5 1000 et 1e-10 chargés par yaml
        initial = values[0]
        final = values[-1]
        max_val = max(values)
        relative_change = abs(final - initial) / (abs(initial) + 1e-10)
        
        if max_val > 1000.0:
            transition = "explosive"
        elif relative_change < 0.1:
            transition = "stable"
        elif final > initial * 1.5:
            transition = "growing"
        elif final < initial * 0.5:
            transition = "shrinking"
        else:
            transition = "oscillating"
        
        volatility = np.std(np.diff(values)) / (np.mean(np.abs(values)) + 1e-10)
        
        return {
            'transition': transition,
            'trend': trend,
            'slope': slope,
            'volatility': float(volatility),
            'relative_change': float(relative_change),
        }
    
    def _load_yaml_params(self, test_id, config_id):
        """Charge params YAML (placeholder)."""
        # TODO: Implémenter chargement YAML
        return {}