# prc_framework/tests/utilities/registries/base_registry.py

from abc import ABC
from typing import Dict, Callable

class BaseRegistry(ABC):
    """
    Classe abstraite pour tous les registres.
    
    Contraintes :
    - REGISTRY_KEY doit être défini et unique
    - Fonctions décorées avec @register_function
    - Signature: func(state, **params) -> float
    """
    
    REGISTRY_KEY: str = None  # À définir dans sous-classe
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._register_all_functions()
    
    def _register_all_functions(self):
        """Découvre et enregistre toutes les fonctions décorées."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_registry_metadata'):
                function_name = attr._registry_metadata['name']
                self._functions[function_name] = attr
    
    def get_function(self, registry_key: str) -> Callable:
        """
        Récupère une fonction par sa clé complète.
        
        Args:
            registry_key: Format "registre.fonction"
        
        Returns:
            Callable
        
        Raises:
            KeyError: Si fonction non trouvée
        """
        _, function_name = registry_key.split('.', 1)
        
        if function_name not in self._functions:
            available = list(self._functions.keys())
            raise KeyError(
                f"Fonction '{function_name}' non trouvée dans {self.REGISTRY_KEY}. "
                f"Disponibles: {available}"
            )
        
        return self._functions[function_name]
    
    def list_functions(self) -> Dict[str, str]:
        """Liste toutes les fonctions avec leur documentation."""
        return {
            name: func.__doc__ or "Pas de documentation"
            for name, func in self._functions.items()
        }


def register_function(name: str):
    """
    Décorateur pour enregistrer une fonction dans un registre.
    
    Usage:
        @register_function("matrix_norm")
        def compute_norm(state, norm_type='fro'):
            return np.linalg.norm(state, norm_type)
    """
    def decorator(func):
        func._registry_metadata = {'name': name}
        return func
    return decorator