# prc_framework/prc_documentation/templates/template_registry.py
"""
Description du domaine mathématique.

Fonctions disponibles :
- fonction_1 : Description
- fonction_2 : Description
"""

import numpy as np
from .base_registry import BaseRegistry, register_function

class TemplateRegistry(BaseRegistry):
    """Registre template."""
    
    REGISTRY_KEY = "template"
    
    @register_function("fonction_exemple")
    def compute_example(
        self,
        state: np.ndarray,
        param1: float = 1.0,
        param2: str = 'default'
    ) -> float:
        """
        Description claire et concise.
        
        Args:
            state: Description du tenseur attendu
            param1: Description paramètre 1
            param2: Description paramètre 2
        
        Returns:
            float: Description de la valeur retournée
        
        Raises:
            ValueError: Conditions d'erreur
        
        Notes:
            - Note 1
            - Note 2
        
        Examples:
            >>> func(np.ones((10,10)), param1=2.0)
            20.0
        """
        # Validation
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Calcul
        result = np.sum(state) * param1
        
        return float(result)