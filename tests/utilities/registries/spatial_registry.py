# tests/utilities/registries/spatial_registry.py
"""
Registre des analyses spatiales.

Fonctions disponibles :
- gradient_magnitude : Magnitude gradient moyen
- laplacian_energy : Énergie laplacien
- local_variance : Variance locale moyenne
- edge_density : Densité contours
- spatial_autocorrelation : Autocorrélation spatiale
- smoothness : Mesure lissage
"""

import numpy as np
from scipy import ndimage
from .base_registry import BaseRegistry, register_function


class SpatialRegistry(BaseRegistry):
    """Registre des analyses spatiales."""
    
    REGISTRY_KEY = "spatial"
    
    @register_function("gradient_magnitude")
    def compute_gradient_magnitude(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Magnitude moyenne du gradient.
        
        Args:
            state: Tenseur rang 2 ou 3
            normalize: Normaliser par amplitude état ?
            epsilon: Protection division
        
        Returns:
            float: Moyenne ||∇state||
        
        Raises:
            ValueError: Si rang < 2 ou > 3
        
        Notes:
            - Mesure variation spatiale
            - Élevé → contours nets, transitions brusques
            - Faible → champ lisse, homogène
        """
        if state.ndim not in [2, 3]:
            raise ValueError(f"Attendu 2D ou 3D, reçu {state.ndim}D")
        
        # Gradient selon chaque axe
        gradients = np.gradient(state)
        
        # Magnitude
        magnitude = np.sqrt(sum(g**2 for g in gradients))
        mean_magnitude = float(np.mean(magnitude))
        
        if normalize:
            amplitude = np.max(np.abs(state)) - np.min(np.abs(state))
            mean_magnitude = mean_magnitude / (amplitude + epsilon)
        
        return mean_magnitude
    
    @register_function("laplacian_energy")
    def compute_laplacian_energy(
        self,
        state: np.ndarray,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Énergie du laplacien (rugosité).
        
        Args:
            state: Matrice 2D
            normalize: Normaliser par énergie totale ?
            epsilon: Protection division
        
        Returns:
            float: Somme |∇²state|²
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Laplacien = ∂²/∂x² + ∂²/∂y²
            - Mesure courbure locale
            - Élevé → surface rugueuse
            - Faible → surface lisse
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Laplacien discret
        laplacian = ndimage.laplace(state)
        
        # Énergie
        energy = float(np.sum(laplacian ** 2))
        
        if normalize:
            total_energy = np.sum(state ** 2) + epsilon
            energy = energy / total_energy
        
        return energy
    
    @register_function("local_variance")
    def compute_local_variance(
        self,
        state: np.ndarray,
        window_size: int = 3,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance locale moyenne.
        
        Args:
            state: Matrice 2D
            window_size: Taille fenêtre (impair)
            normalize: Normaliser par variance globale ?
            epsilon: Protection division
        
        Returns:
            float: Moyenne des variances locales
        
        Raises:
            ValueError: Si non 2D ou window_size pair
        
        Notes:
            - Mesure hétérogénéité texture
            - Élevé → texture complexe, hétérogène
            - Faible → texture uniforme, homogène
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if window_size % 2 == 0:
            raise ValueError(f"window_size doit être impair, reçu {window_size}")
        
        # Moyenne locale
        local_mean = ndimage.uniform_filter(state, size=window_size)
        
        # Variance locale = E[X²] - E[X]²
        local_mean_sq = ndimage.uniform_filter(state**2, size=window_size)
        local_var = local_mean_sq - local_mean**2
        
        # Moyenne des variances locales
        mean_local_var = float(np.mean(local_var))
        
        if normalize:
            global_var = np.var(state) + epsilon
            mean_local_var = mean_local_var / global_var
        
        return mean_local_var
    
    @register_function("edge_density")
    def compute_edge_density(
        self,
        state: np.ndarray,
        threshold: float = 0.1,
        method: str = 'sobel'
    ) -> float:
        """
        Densité de contours détectés.
        
        Args:
            state: Matrice 2D
            threshold: Seuil détection contour (relatif au max)
            method: 'sobel' | 'prewitt' | 'laplace'
        
        Returns:
            float: Fraction pixels > seuil dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou méthode inconnue
        
        Notes:
            - Mesure complexité structure
            - Élevé → nombreux contours, structure complexe
            - Faible → peu de contours, structure simple
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Détection contours
        if method == 'sobel':
            sx = ndimage.sobel(state, axis=0)
            sy = ndimage.sobel(state, axis=1)
            edges = np.hypot(sx, sy)
        elif method == 'prewitt':
            sx = ndimage.prewitt(state, axis=0)
            sy = ndimage.prewitt(state, axis=1)
            edges = np.hypot(sx, sy)
        elif method == 'laplace':
            edges = np.abs(ndimage.laplace(state))
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Normaliser
        max_edge = np.max(edges)
        if max_edge == 0:
            return 0.0
        
        edges_normalized = edges / max_edge
        
        # Densité
        density = np.sum(edges_normalized > threshold) / edges.size
        
        return float(density)
    
    @register_function("spatial_autocorrelation")
    def compute_spatial_autocorrelation(
        self,
        state: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Autocorrélation spatiale (Moran's I simplifié).
        
        Args:
            state: Matrice 2D
            lag: Distance pixels voisins
        
        Returns:
            float: Corrélation spatiale
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure similarité pixels voisins
            - > 0 : Agrégation (pixels similaires groupés)
            - ≈ 0 : Distribution aléatoire
            - < 0 : Dispersion (alternance valeurs)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Décaler selon lag
        shifted_h = np.roll(state, lag, axis=0)
        shifted_v = np.roll(state, lag, axis=1)
        
        # Corrélations
        corr_h = np.corrcoef(state.flatten(), shifted_h.flatten())[0, 1]
        corr_v = np.corrcoef(state.flatten(), shifted_v.flatten())[0, 1]
        
        # Moyenne
        if np.isnan(corr_h):
            corr_h = 0.0
        if np.isnan(corr_v):
            corr_v = 0.0
        
        autocorr = (corr_h + corr_v) / 2.0
        
        return float(autocorr)
    
    @register_function("smoothness")
    def compute_smoothness(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Mesure de lissage (inverse rugosité).
        
        Args:
            state: Matrice 2D
            epsilon: Protection division
        
        Returns:
            float: 1 / (1 + variance_gradient)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Valeurs dans [0, 1]
            - 1.0 → parfaitement lisse
            - 0.0 → très rugueux
            - Basé sur variance du gradient
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Gradient
        gx, gy = np.gradient(state)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Variance gradient
        var_gradient = np.var(gradient_magnitude)
        
        # Smoothness
        smoothness = 1.0 / (1.0 + var_gradient + epsilon)
        
        return float(smoothness)