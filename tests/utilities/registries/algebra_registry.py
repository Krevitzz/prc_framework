# tests/utilities/registries/algebra_registry.py
"""
Registre des opérations algébriques sur tenseurs.

Fonctions disponibles :
- matrix_asymmetry : Norme partie anti-symétrique
- matrix_norm : Norme tenseur d'ordre N
- trace_value : Trace de matrice
- determinant_value : Déterminant
- frobenius_norm : Norme Frobenius (alias optimisé)
- spectral_norm : Norme spectrale
- condition_number : Conditionnement matrice
- rank_estimate : Estimation rang effectif
"""

import numpy as np
from .base_registry import BaseRegistry, register_function


class AlgebraRegistry(BaseRegistry):
    """Registre des opérations algébriques sur tenseurs."""
    
    REGISTRY_KEY = "algebra"
    
    @register_function("matrix_asymmetry")
    def compute_asymmetry(
        self,
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
            - Symétrie parfaite → 0.0
            - normalize=True → valeurs dans [0, 1]
        
        Examples:
            >>> state = np.array([[1, 2], [2, 1]])  # Symétrique
            >>> compute_asymmetry(state)
            0.0
            
            >>> state = np.array([[1, 2], [3, 1]])  # Asymétrique
            >>> compute_asymmetry(state) > 0
            True
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
        self,
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
        
        Notes:
            - Frobenius : Généralisable à tout rang, O(n²)
            - Spectral : Matrice 2D uniquement, O(n³)
            - Nuclear : Matrice 2D uniquement, O(n³)
            - int : Norme Lp du vecteur aplati
        
        Examples:
            >>> state = np.ones((10, 10))
            >>> compute_norm(state, 'frobenius')
            10.0
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
            # ✅ INT natif (Python direct)
            return float(np.linalg.norm(state.flatten(), norm_type))
        
        else:
            # ✅ FIX: Tentative conversion string → int (YAML/JSON)
            try:
                norm_type_int = int(norm_type)
                return float(np.linalg.norm(state.flatten(), norm_type_int))
            except (ValueError, TypeError):
                raise ValueError(f"Type norme inconnu: {norm_type}")
    
    @register_function("frobenius_norm")
    def compute_frobenius(
        self,
        state: np.ndarray
    ) -> float:
        """
        Norme Frobenius (alias optimisé).
        
        Args:
            state: Tenseur de rang N
        
        Returns:
            float: ||state||_F = sqrt(sum(state**2))
        
        Notes:
            - Généralisation norme euclidienne
            - O(n²) pour matrices n×n
            - Toujours >= 0
        """
        return float(np.linalg.norm(state, 'fro'))
    
    @register_function("trace_value")
    def compute_trace(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Trace de matrice.
        
        Args:
            state: Matrice carrée 2D
            normalize: Diviser par dimension ?
            epsilon: Protection si normalize=True
        
        Returns:
            float: Trace ou trace normalisée
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Trace = somme valeurs propres
            - Trace(AB) = Trace(BA)
            - normalize=True → moyenne diagonale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        trace = np.trace(state)
        
        if normalize:
            trace = trace / (state.shape[0] + epsilon)
        
        return float(trace)
    
    @register_function("determinant_value")
    def compute_determinant(
        self,
        state: np.ndarray,
        log_scale: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Déterminant de matrice.
        
        Args:
            state: Matrice carrée 2D
            log_scale: Retourner log|det| ?
            epsilon: Protection log si det proche de 0
        
        Returns:
            float: det(state) ou log|det(state)|
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Déterminant = produit valeurs propres
            - det = 0 → matrice singulière
            - log_scale utile si det très grand/petit
            - log_scale=True → log(|det| + epsilon)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        det = np.linalg.det(state)
        
        if log_scale:
            return float(np.log(np.abs(det) + epsilon))
        
        return float(det)
    
    @register_function("spectral_norm")
    def compute_spectral_norm(
        self,
        state: np.ndarray
    ) -> float:
        """
        Norme spectrale (plus grande valeur singulière).
        
        Args:
            state: Matrice 2D
        
        Returns:
            float: σ_max (plus grande valeur singulière)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Norme spectrale = ||A||_2
            - σ_max = sqrt(plus grande valeur propre de A^T A)
            - Mesure d'amplification maximale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        return float(np.linalg.norm(state, 2))
    
    @register_function("condition_number")
    def compute_condition(
        self,
        state: np.ndarray,
        norm_type: str = 'spectral',
        epsilon: float = 1e-10
    ) -> float:
        """
        Conditionnement de matrice.
        
        Args:
            state: Matrice carrée 2D
            norm_type: 'spectral' | 'frobenius'
            epsilon: Protection division par zéro
        
        Returns:
            float: κ(A) = ||A|| * ||A⁻¹||
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - κ ≥ 1 toujours
            - κ proche de 1 → bien conditionnée
            - κ >> 1 → mal conditionnée
            - κ = ∞ → singulière
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        if norm_type == 'spectral':
            p = 2
        elif norm_type == 'frobenius':
            p = 'fro'
        else:
            raise ValueError(f"Type norme inconnu: {norm_type}")
        
        try:
            cond = np.linalg.cond(state, p=p)
            if np.isinf(cond) or np.isnan(cond):
                return float(1e10)  # Valeur sentinelle
            return float(cond)
        except np.linalg.LinAlgError:
            return float(1e10)
    
    @register_function("rank_estimate")
    def compute_rank(
        self,
        state: np.ndarray,
        tolerance: float = 1e-10
    ) -> float:
        """
        Estimation rang effectif via SVD.
        
        Args:
            state: Matrice 2D
            tolerance: Seuil valeur singulière (σ > tolerance)
        
        Returns:
            float: Nombre de valeurs singulières > tolerance
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Rang numérique vs rang théorique
            - tolerance détermine sensibilité
            - Rang max = min(n, m) pour matrice n×m
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        singular_values = np.linalg.svd(state, compute_uv=False)
        rank = np.sum(singular_values > tolerance)
        
        return float(rank)