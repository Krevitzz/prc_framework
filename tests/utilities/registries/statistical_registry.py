# tests/utilities/registries/statistical_registry.py
"""
Registre des analyses statistiques.

Fonctions disponibles :
- entropy : Entropie de Shannon
- kurtosis : Aplatissement distribution
- skewness : Asymétrie distribution
- variance : Variance
- std_normalized : Écart-type normalisé (coefficient variation)
- correlation_mean : Corrélation moyenne
- sparsity : Mesure de parcimonie
"""

import numpy as np
from scipy import stats
from .base_registry import BaseRegistry, register_function


class StatisticalRegistry(BaseRegistry):
    """Registre des analyses statistiques."""
    
    REGISTRY_KEY = "statistical"
    
    @register_function("entropy")
    def compute_entropy(
        self,
        state: np.ndarray,
        bins: int = 50,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Entropie de Shannon de la distribution.
        
        Args:
            state: Tenseur rang N
            bins: Nombre de bins histogramme
            normalize: Normaliser par log(bins) ?
            epsilon: Protection log(0)
        
        Returns:
            float: H = -Σ p_i log(p_i)
        
        Notes:
            - Mesure dispersion distribution
            - normalize=True → valeurs dans [0, 1]
            - Entropie max → distribution uniforme
            - Entropie min → distribution concentrée
        """
        # Aplatir tenseur
        values = state.flatten()
        
        # Histogramme
        hist, _ = np.histogram(values, bins=bins, density=True)
        
        # Normaliser en probabilités
        hist = hist / (np.sum(hist) + epsilon)
        
        # Entropie
        entropy = -np.sum(hist * np.log(hist + epsilon))
        
        if normalize:
            max_entropy = np.log(bins)
            entropy = entropy / (max_entropy + epsilon)
        
        return float(entropy)
    
    @register_function("kurtosis")
    def compute_kurtosis(
        self,
        state: np.ndarray,
        fisher: bool = True
    ) -> float:
        """
        Aplatissement (kurtosis) de la distribution.
        
        Args:
            state: Tenseur rang N
            fisher: Utiliser définition Fisher (excess kurtosis) ?
        
        Returns:
            float: Kurtosis
        
        Notes:
            - fisher=True → kurtosis - 3 (normale → 0)
            - fisher=False → kurtosis brute (normale → 3)
            - > 0 : Distribution pointue (leptokurtique)
            - < 0 : Distribution aplatie (platykurtique)
        """
        values = state.flatten()
        kurt = float(stats.kurtosis(values, fisher=fisher))
        return kurt
    
    @register_function("skewness")
    def compute_skewness(
        self,
        state: np.ndarray
    ) -> float:
        """
        Asymétrie (skewness) de la distribution.
        
        Args:
            state: Tenseur rang N
        
        Returns:
            float: Coefficient d'asymétrie
        
        Notes:
            - = 0 : Distribution symétrique
            - > 0 : Queue droite (valeurs extrêmes positives)
            - < 0 : Queue gauche (valeurs extrêmes négatives)
        """
        values = state.flatten()
        skew = float(stats.skew(values))
        return skew
    
    @register_function("variance")
    def compute_variance(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance de la distribution.
        
        Args:
            state: Tenseur rang N
            normalize: Diviser par carré de la moyenne ?
            epsilon: Protection division
        
        Returns:
            float: Var(X) ou Var(X) / E[X]²
        
        Notes:
            - Variance = E[(X - E[X])²]
            - normalize=True → coefficient de variation²
            - Mesure dispersion relative
        """
        values = state.flatten()
        var = float(np.var(values))
        
        if normalize:
            mean = np.mean(values)
            var = var / ((mean ** 2) + epsilon)
        
        return var
    
    @register_function("std_normalized")
    def compute_std_normalized(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient de variation (CV).
        
        Args:
            state: Tenseur rang N
            epsilon: Protection division
        
        Returns:
            float: CV = σ / |μ|
        
        Notes:
            - Mesure dispersion relative
            - Sans unité (comparable entre échelles)
            - CV faible → valeurs homogènes
            - CV élevé → valeurs dispersées
        """
        values = state.flatten()
        mean = np.mean(values)
        std = np.std(values)
        
        cv = std / (np.abs(mean) + epsilon)
        
        return float(cv)
    
    @register_function("correlation_mean")
    def compute_correlation_mean(
        self,
        state: np.ndarray,
        axis: int = 0,
        method: str = 'pearson'
    ) -> float:
        """
        Corrélation moyenne entre lignes ou colonnes.
        
        Args:
            state: Matrice 2D
            axis: 0 (lignes) ou 1 (colonnes)
            method: 'pearson' | 'spearman'
        
        Returns:
            float: Moyenne corrélations par paires
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure similarité patterns
            - Valeurs dans [-1, 1]
            - Proche de 1 → patterns similaires
            - Proche de 0 → patterns indépendants
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if axis == 0:
            vectors = state
        else:
            vectors = state.T
        
        n = vectors.shape[0]
        if n < 2:
            return 0.0
        
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                if method == 'pearson':
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                elif method == 'spearman':
                    corr, _ = stats.spearmanr(vectors[i], vectors[j])
                else:
                    raise ValueError(f"Méthode inconnue: {method}")
                
                if np.isfinite(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        return float(np.mean(correlations))
    
    @register_function("sparsity")
    def compute_sparsity(
        self,
        state: np.ndarray,
        threshold: float = 1e-6
    ) -> float:
        """
        Mesure de parcimonie (fraction valeurs proches de 0).
        
        Args:
            state: Tenseur rang N
            threshold: Seuil |x| < threshold → 0
        
        Returns:
            float: Fraction d'éléments "zéro" dans [0, 1]
        
        Notes:
            - 1.0 → tous éléments nuls (très parcimonieux)
            - 0.0 → aucun élément nul (dense)
            - Utile pour détecter structures creuses
        """
        values = state.flatten()
        num_zeros = np.sum(np.abs(values) < threshold)
        sparsity = num_zeros / len(values)
        
        return float(sparsity)