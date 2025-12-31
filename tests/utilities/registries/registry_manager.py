# prc_framework/utilities/registries/registry_manager.py

import importlib
import pkgutil
import inspect
from pathlib import Path
from typing import Dict, Any, Callable
import numpy as np

from .base_registry import BaseRegistry
from .post_processors import get_post_processor

class RegistryManager:
    """
    Singleton gérant tous les registres.
    
    Responsabilités :
    1. Charger dynamiquement tous les *_registry.py
    2. Valider registry_key
    3. Fournir fonctions avec paramètres validés
    4. Résoudre post_processors
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registries: Dict[str, BaseRegistry] = {}
        self.function_cache: Dict[str, Callable] = {}
        
        self._load_all_registries()
        self._initialized = True
    
    def _load_all_registries(self) -> None:
        """Charge dynamiquement tous les registres."""
        package_name = "prc_framework.utilities.registries"
        
        try:
            package = importlib.import_module(package_name)
            package_path = Path(package.__file__).parent
            
            for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
                if not is_pkg and module_name.endswith('_registry'):
                    full_name = f"{package_name}.{module_name}"
                    
                    try:
                        module = importlib.import_module(full_name)
                        
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            
                            if (isinstance(attr, type) and 
                                issubclass(attr, BaseRegistry) and 
                                attr != BaseRegistry):
                                
                                instance = attr()
                                
                                if not instance.REGISTRY_KEY:
                                    raise ValueError(f"{attr_name} sans REGISTRY_KEY")
                                
                                if instance.REGISTRY_KEY in self.registries:
                                    raise ValueError(f"REGISTRY_KEY dupliqué: {instance.REGISTRY_KEY}")
                                
                                self.registries[instance.REGISTRY_KEY] = instance
                                print(f"[RegistryManager] Chargé: {instance.REGISTRY_KEY}")
                    
                    except Exception as e:
                        print(f"[RegistryManager] Erreur {module_name}: {e}")
                        continue
        
        except Exception as e:
            print(f"[RegistryManager] Erreur initialisation: {e}")
            raise
    
    def get_function(self, registry_key: str) -> Callable:
        """
        Récupère fonction par clé complète.
        
        Args:
            registry_key: Format "registre.fonction"
        
        Returns:
            Callable
        
        Raises:
            KeyError: Si registre ou fonction non trouvée
        """
        if registry_key in self.function_cache:
            return self.function_cache[registry_key]
        
        if '.' not in registry_key:
            raise KeyError(f"Format invalide: {registry_key}. Attendu: registre.fonction")
        
        registry_name, function_name = registry_key.split('.', 1)
        
        if registry_name not in self.registries:
            available = list(self.registries.keys())
            raise KeyError(f"Registre '{registry_name}' inconnu. Disponibles: {available}")
        
        registry = self.registries[registry_name]
        function = registry.get_function(registry_key)
        
        self.function_cache[registry_key] = function
        
        return function
    
    def validate_computation_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et prépare une spécification de calcul.
        
        Args:
            spec: {
                'registry_key': str,
                'default_params': dict,
                'post_process': str (optionnel),
                'validation': dict (optionnel),
            }
        
        Returns:
            {
                'function': Callable,
                'params': dict (validés),
                'post_process': Callable | None,
                'registry_key': str,
            }
        
        Raises:
            ValueError: Si validation échoue
        """
        # Clés obligatoires
        if 'registry_key' not in spec:
            raise ValueError("Manque 'registry_key'")
        
        if 'default_params' not in spec:
            raise ValueError("Manque 'default_params'")
        
        registry_key = spec['registry_key']
        
        # 1. Récupérer fonction
        function = self.get_function(registry_key)
        
        # 2. Valider paramètres
        validated_params = self._validate_params(function, spec['default_params'])
        
        # 3. Résoudre post_process
        post_process = None
        if 'post_process' in spec and spec['post_process']:
            post_process = get_post_processor(spec['post_process'])
        
        return {
            'function': function,
            'params': validated_params,
            'post_process': post_process,
            'registry_key': registry_key,
        }
    
    def _validate_params(self, function: Callable, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide paramètres contre signature fonction.
        
        Args:
            function: Fonction du registre
            user_params: Paramètres fournis
        
        Returns:
            Dict paramètres validés et complétés
        
        Raises:
            ValueError: Si paramètre invalide
        """
        sig = inspect.signature(function)
        parameters = sig.parameters
        
        if 'state' not in parameters:
            raise ValueError("Fonction doit avoir 'state' comme premier paramètre")
        
        # Valeurs par défaut de la fonction
        defaults = {
            name: param.default
            for name, param in parameters.items()
            if param.default is not inspect.Parameter.empty and name != 'state'
        }
        
        validated = {}
        
        for param_name, param_value in user_params.items():
            if param_name not in parameters:
                raise ValueError(
                    f"Paramètre '{param_name}' invalide. "
                    f"Attendus: {list(parameters.keys())[1:]}"
                )
            
            expected_param = parameters[param_name]
            
            # Validation type (basique)
            if expected_param.annotation is not inspect.Parameter.empty:
                expected_type = expected_param.annotation
                
                if not isinstance(param_value, expected_type):
                    try:
                        param_value = expected_type(param_value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Paramètre '{param_name}' attend {expected_type}, "
                            f"reçu {type(param_value)}"
                        )
            
            validated[param_name] = param_value
        
        # Ajouter defaults manquants
        for param_name, default_value in defaults.items():
            if param_name not in validated:
                validated[param_name] = default_value
        
        return validated
    
    def list_available_functions(self) -> Dict[str, list]:
        """Liste toutes les fonctions par registre."""
        return {
            registry_name: list(registry.list_functions().keys())
            for registry_name, registry in self.registries.items()
        }