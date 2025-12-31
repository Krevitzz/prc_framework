# prc_framework/tests/utilities/registries/algebra_registry.py

import numpy as np
from .base_registry import BaseRegistry, register_function

class AlgebraRegistry(BaseRegistry):
    """
    Registre des opérations algébriques sur tenseurs.
    
    Fonctions disponibles :
    - matrix_asymmetry : Norme partie anti-symétrique
    - matrix_norm : Norme tenseur d'ordre N
    - trace_value : Trace de matrice
    - determinant_value : Déterminant
    """
    
    REGISTRY_KEY = "algebra"
    
    @register_function("matrix_asymmetry")
    def compute_asymmetry(
        state: np.ndarray,
        norm_type: str = 'frobenius',
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calcule la norme de la partie anti-symétrique.
        
        Args:
            state: Tenseur rang 2 (matrice carrée)
            norm_type: 'frobenius' | 'spectral' | 'nuclear'
            normalize: Normaliser par norme totale ?
            epsilon: Protection division par zéro
        
        Returns:
            float: Norme de (state - state.T) / 2
        
        Raises:
            ValueError: Si state non 2D ou non carré
        
        Notes:
            - Valeurs élevées indiquent forte asymétrie
            - normalize=True permet comparaison entre matrices tailles différentes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        asymmetry = state - state.T
        
        if norm_type == 'frobenius':
            norm = np.linalg.norm(asymmetry, 'fro')
        elif norm_type == 'spectral':
            norm = np.linalg.norm(asymmetry, 2)
        elif norm_type == 'nuclear':
            norm = np.sum(np.linalg.svd(asymmetry, compute_uv=False))
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")
        
        if normalize:
            total_norm = np.linalg.norm(state, 'fro') + epsilon
            norm = norm / total_norm
        
        return float(norm)
    
    @register_function("matrix_norm")
    def compute_norm(
        state: np.ndarray,
        norm_type: str = 'frobenius'
    ) -> float:
        """
        Norme de tenseur d'ordre quelconque.
        
        Args:
            state: Tenseur de rang N
            norm_type: 'frobenius' | 'spectral' | 'nuclear' | int
        
        Returns:
            float: Norme du tenseur
        
        Raises:
            ValueError: Si norm_type incompatible avec rang
        """
        if norm_type == 'frobenius':
            return float(np.linalg.norm(state, 'fro'))
        elif norm_type == 'spectral':
            if state.ndim != 2:
                raise ValueError("Norme spectrale requiert matrice 2D")
            return float(np.linalg.norm(state, 2))
        elif norm_type == 'nuclear':
            if state.ndim != 2:
                raise ValueError("Norme nucléaire requiert matrice 2D")
            return float(np.sum(np.linalg.svd(state, compute_uv=False)))
        elif isinstance(norm_type, int):
            return float(np.linalg.norm(state.flatten(), norm_type))
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")