# tests/utilities/registries/mydomain_registry.py
"""
Registre [domaine].

Fonctions disponibles :
- fonction1 : Description
- fonction2 : Description
"""

import numpy as np
from .base_registry import BaseRegistry, register_function


class MyDomainRegistry(BaseRegistry):
    """Registre [domaine]."""
    
    REGISTRY_KEY = "mydomain"  # Unique, lowercase
    
    @register_function("my_function")
    def compute_something(
        self,
        state: np.ndarray,
        param1: float = 1.0,
        param2: str = 'default'
    ) -> float:
        """
        Description claire.
        
        Args:
            state: Description tenseur attendu
            param1: Description param1
            param2: Description param2
        
        Returns:
            float: Description valeur
        
        Raises:
            ValueError: Conditions erreur
        
        Notes:
            - Note importante 1
            - Note importante 2
        
        Examples:
            >>> compute_something(np.ones((10,10)), param1=2.0)
            20.0
        """
        # Validation
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Calcul
        result = np.sum(state) * param1
        
        return float(result)