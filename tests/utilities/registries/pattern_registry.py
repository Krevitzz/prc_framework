# tests/utilities/registries/pattern_registry.py
"""
Registre de détection de patterns.

Fonctions disponibles :
- periodicity : Détection périodicité
- symmetry_score : Score symétrie radiale
- clustering_coefficient : Coefficient clustering
- diversity : Diversité valeurs (indice Simpson)
- uniformity : Uniformité distribution
- concentration_ratio : Ratio concentration (Gini)
"""

import numpy as np
from scipy import signal
from .base_registry import BaseRegistry, register_function


class PatternRegistry(BaseRegistry):
    """Registre de détection de patterns."""
    
    REGISTRY_KEY = "pattern"
    
    @register_function("periodicity")
    def compute_periodicity(
        self,
        state: np.ndarray,
        axis: int = 0,
        method: str = 'autocorr'
    ) -> float:
        """
        Détection de périodicité.
        
        Args:
            state: Tenseur rang N
            axis: Axe analyse (si multi-dim)
            method: 'autocorr' | 'fft'
        
        Returns:
            float: Score périodicité dans [0, 1]
        
        Notes:
            - 1.0 → signal parfaitement périodique
            - 0.0 → signal aléatoire
            - autocorr : Basé sur pics autocorrélation
            - fft : Basé sur pics spectre fréquence
        """
        # Prendre slice si multi-dim
        if state.ndim > 1:
            signal_1d = np.mean(state, axis=tuple(i for i in range(state.ndim) if i != axis))
        else:
            signal_1d = state
        
        if method == 'autocorr':
            # Autocorrélation
            autocorr = np.correlate(signal_1d, signal_1d, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normaliser
            
            # Chercher pics
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            
            if len(peaks) > 0:
                # Périodicité = hauteur pic max
                periodicity = float(np.max(autocorr[peaks + 1]))
            else:
                periodicity = 0.0
        
        elif method == 'fft':
            # FFT
            fft = np.fft.fft(signal_1d)
            power = np.abs(fft[:len(fft)//2])**2
            power = power / np.sum(power)
            
            # Périodicité = concentration énergie
            periodicity = float(np.max(power))
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        return np.clip(periodicity, 0.0, 1.0)
    
    @register_function("symmetry_score")
    def compute_symmetry_score(
        self,
        state: np.ndarray,
        symmetry_type: str = 'reflection',
        axis: int = 0
    ) -> float:
        """
        Score symétrie.
        
        Args:
            state: Matrice 2D
            symmetry_type: 'reflection' | 'rotation'
            axis: Axe réflexion (0 ou 1) si reflection
        
        Returns:
            float: Score symétrie dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → parfaitement symétrique
            - 0.0 → totalement asymétrique
            - reflection : Symétrie miroir
            - rotation : Symétrie rotation 180°
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if symmetry_type == 'reflection':
            # Réflexion selon axe
            if axis == 0:
                reflected = np.flipud(state)
            else:
                reflected = np.fliplr(state)
            
            # Corrélation avec réfléchi
            corr = np.corrcoef(state.flatten(), reflected.flatten())[0, 1]
        
        elif symmetry_type == 'rotation':
            # Rotation 180°
            rotated = np.rot90(state, k=2)
            
            # Corrélation avec tourné
            corr = np.corrcoef(state.flatten(), rotated.flatten())[0, 1]
        
        else:
            raise ValueError(f"Type symétrie inconnu: {symmetry_type}")
        
        # Convertir corrélation [-1, 1] en score [0, 1]
        score = (corr + 1.0) / 2.0
        
        if np.isnan(score):
            score = 0.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    @register_function("clustering_coefficient")
    def compute_clustering(
        self,
        state: np.ndarray,
        threshold: float = 0.5,
        normalize: bool = True
    ) -> float:
        """
        Coefficient clustering (valeurs similaires groupées).
        
        Args:
            state: Matrice 2D
            threshold: Seuil similarité (percentile si normalize)
            normalize: Interpréter threshold comme percentile ?
        
        Returns:
            float: Score clustering dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → valeurs similaires fortement groupées
            - 0.0 → distribution homogène
            - Basé sur corrélation pixels voisins
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if normalize:
            threshold = np.percentile(np.abs(state), threshold * 100)
        
        # Masque valeurs élevées
        mask = np.abs(state) > threshold
        
        if not np.any(mask):
            return 0.0
        
        # Compter voisins similaires
        neighbors = 0
        similar_neighbors = 0
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if mask[i, j]:
                    # 4-voisinage
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < state.shape[0] and 0 <= nj < state.shape[1]:
                            neighbors += 1
                            if mask[ni, nj]:
                                similar_neighbors += 1
        
        if neighbors == 0:
            return 0.0
        
        clustering = similar_neighbors / neighbors
        
        return float(clustering)
    
    @register_function("diversity")
    def compute_diversity(
        self,
        state: np.ndarray,
        bins: int = 50,
        epsilon: float = 1e-10
    ) -> float:
        """
        Diversité (indice Simpson).
        
        Args:
            state: Tenseur rang N
            bins: Nombre bins histogramme
            epsilon: Protection division
        
        Returns:
            float: Indice Simpson dans [0, 1]
        
        Notes:
            - 1.0 → distribution uniforme (haute diversité)
            - 0.0 → concentration (basse diversité)
            - D = 1 - Σ p_i²
        """
        # Histogramme
        values = state.flatten()
        hist, _ = np.histogram(values, bins=bins)
        
        # Normaliser
        probs = hist / (np.sum(hist) + epsilon)
        
        # Indice Simpson
        simpson = 1.0 - np.sum(probs ** 2)
        
        return float(simpson)
    
    @register_function("uniformity")
    def compute_uniformity(
        self,
        state: np.ndarray,
        bins: int = 50,
        epsilon: float = 1e-10
    ) -> float:
        """
        Uniformité distribution.
        
        Args:
            state: Tenseur rang N
            bins: Nombre bins histogramme
            epsilon: Protection division
        
        Returns:
            float: Score uniformité dans [0, 1]
        
        Notes:
            - 1.0 → distribution parfaitement uniforme
            - 0.0 → distribution très concentrée
            - Basé sur distance à distribution uniforme
        """
        # Histogramme
        values = state.flatten()
        hist, _ = np.histogram(values, bins=bins)
        
        # Normaliser
        probs = hist / (np.sum(hist) + epsilon)
        
        # Distribution uniforme cible
        uniform = np.ones(bins) / bins
        
        # Distance chi-carré
        chi_sq = np.sum((probs - uniform) ** 2 / (uniform + epsilon))
        
        # Convertir en score [0, 1]
        # chi_sq max ≈ bins (tous points dans 1 bin)
        uniformity = 1.0 - np.clip(chi_sq / bins, 0.0, 1.0)
        
        return float(uniformity)
    
    @register_function("concentration_ratio")
    def compute_concentration(
        self,
        state: np.ndarray,
        top_percent: float = 0.1,
        epsilon: float = 1e-10
    ) -> float:
        """
        Ratio concentration (Gini-like).
        
        Args:
            state: Tenseur rang N
            top_percent: Fraction valeurs top (ex: 0.1 = top 10%)
            epsilon: Protection division
        
        Returns:
            float: Fraction énergie dans top_percent valeurs
        
        Notes:
            - Mesure concentration distribution
            - 1.0 → toute énergie dans quelques valeurs
            - ~top_percent → distribution uniforme
            - Utile détecter activations parcimonieuses
        """
        values = np.abs(state.flatten())
        
        # Trier décroissant
        sorted_values = np.sort(values)[::-1]
        
        # Nombre top valeurs
        n_top = int(len(sorted_values) * top_percent)
        if n_top == 0:
            n_top = 1
        
        # Énergie top vs total
        energy_top = np.sum(sorted_values[:n_top] ** 2)
        energy_total = np.sum(sorted_values ** 2) + epsilon
        
        concentration = energy_top / energy_total
        
        return float(concentration)