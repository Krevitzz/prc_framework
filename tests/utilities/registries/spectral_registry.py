# tests/utilities/registries/spectral_registry.py
"""
Registre des analyses spectrales.

Fonctions disponibles :
- eigenvalue_max : Plus grande valeur propre
- eigenvalue_distribution : Statistiques spectre
- spectral_gap : Écart spectral
- fft_power : Puissance fréquentielle
- fft_entropy : Entropie spectrale
- spectral_radius : Rayon spectral
"""

import numpy as np
from .base_registry import BaseRegistry, register_function


class SpectralRegistry(BaseRegistry):
    """Registre des analyses spectrales."""
    
    REGISTRY_KEY = "spectral"
    
    @register_function("eigenvalue_max")
    def compute_eigenvalue_max(
        self,
        state: np.ndarray,
        absolute: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Plus grande valeur propre.
        
        Args:
            state: Matrice carrée 2D
            absolute: Considérer |λ| ou λ ?
            epsilon: Protection si matrice quasi-singulière
        
        Returns:
            float: λ_max ou |λ|_max
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - absolute=True → plus grande magnitude
            - absolute=False → plus grande valeur algébrique
            - Peut être complexe → prend module si absolute=True
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.linalg.eigvals(state)
            
            if absolute:
                return float(np.max(np.abs(eigenvalues)))
            else:
                # Prendre partie réelle si complexe
                if np.iscomplexobj(eigenvalues):
                    eigenvalues = eigenvalues.real
                return float(np.max(eigenvalues))
        
        except np.linalg.LinAlgError:
            return float(epsilon)
    
    @register_function("eigenvalue_distribution")
    def compute_eigenvalue_stats(
        self,
        state: np.ndarray,
        stat: str = 'mean',
        epsilon: float = 1e-10
    ) -> float:
        """
        Statistiques distribution valeurs propres.
        
        Args:
            state: Matrice carrée 2D
            stat: 'mean' | 'std' | 'min' | 'max' | 'range'
            epsilon: Protection calculs
        
        Returns:
            float: Statistique choisie sur |λ_i|
        
        Raises:
            ValueError: Si non 2D, non carrée, ou stat inconnue
        
        Notes:
            - Calcul sur magnitudes (|λ|)
            - 'range' = max - min
            - Robuste aux valeurs propres complexes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.abs(np.linalg.eigvals(state))
            
            if stat == 'mean':
                return float(np.mean(eigenvalues))
            elif stat == 'std':
                return float(np.std(eigenvalues))
            elif stat == 'min':
                return float(np.min(eigenvalues))
            elif stat == 'max':
                return float(np.max(eigenvalues))
            elif stat == 'range':
                return float(np.max(eigenvalues) - np.min(eigenvalues))
            else:
                raise ValueError(f"Stat inconnue: {stat}")
        
        except np.linalg.LinAlgError:
            return float(epsilon)
    
    @register_function("spectral_gap")
    def compute_spectral_gap(
        self,
        state: np.ndarray,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Écart spectral (λ_max - λ_2).
        
        Args:
            state: Matrice carrée 2D
            normalize: Diviser par λ_max ?
            epsilon: Protection division
        
        Returns:
            float: λ_1 - λ_2 ou (λ_1 - λ_2) / λ_1
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure séparation valeur propre dominante
            - Gap grand → convergence rapide
            - normalize=True → valeur relative dans [0, 1]
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.abs(np.linalg.eigvals(state))
            eigenvalues = np.sort(eigenvalues)[::-1]  # Décroissant
            
            if len(eigenvalues) < 2:
                return float(0.0)
            
            gap = eigenvalues[0] - eigenvalues[1]
            
            if normalize and eigenvalues[0] > epsilon:
                gap = gap / eigenvalues[0]
            
            return float(gap)
        
        except np.linalg.LinAlgError:
            return float(0.0)
    
    @register_function("fft_power")
    def compute_fft_power(
        self,
        state: np.ndarray,
        normalize: bool = True,
        epsilon: float = 1e-10
    ) -> float:
        """
        Puissance totale spectre FFT.
        
        Args:
            state: Tenseur rang N
            normalize: Normaliser par nombre éléments ?
            epsilon: Protection calculs
        
        Returns:
            float: Somme |FFT(state)|²
        
        Notes:
            - FFT multi-dimensionnelle si rang > 1
            - Mesure énergie fréquentielle
            - normalize=True → comparable entre tailles
        """
        fft_result = np.fft.fftn(state)
        power = np.sum(np.abs(fft_result) ** 2)
        
        if normalize:
            power = power / (state.size + epsilon)
        
        return float(power)
    
    @register_function("fft_entropy")
    def compute_fft_entropy(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Entropie spectrale du spectre FFT.
        
        Args:
            state: Tenseur rang N
            epsilon: Protection log(0)
        
        Returns:
            float: -Σ p_i log(p_i) où p_i = |FFT_i|² / Σ|FFT|²
        
        Notes:
            - Entropie élevée → spectre dispersé
            - Entropie faible → spectre concentré
            - Valeurs dans [0, log(N)] où N = state.size
        """
        fft_result = np.fft.fftn(state)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Normaliser en distribution
        total_power = np.sum(power_spectrum) + epsilon
        probabilities = power_spectrum / total_power
        
        # Entropie de Shannon
        entropy = -np.sum(
            probabilities * np.log(probabilities + epsilon)
        )
        
        return float(entropy)
    
    @register_function("spectral_radius")
    def compute_spectral_radius(
        self,
        state: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Rayon spectral (max |λ_i|).
        
        Args:
            state: Matrice carrée 2D
            epsilon: Protection si singulière
        
        Returns:
            float: ρ(A) = max_i |λ_i|
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - ρ(A) ≤ ||A|| pour toute norme
            - ρ(A) < 1 → itérations A^n convergent
            - ρ(A) > 1 → potentiellement divergent
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        try:
            eigenvalues = np.linalg.eigvals(state)
            return float(np.max(np.abs(eigenvalues)))
        
        except np.linalg.LinAlgError:
            return float(epsilon)