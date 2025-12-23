

"""
operators/diffusion.py

Opérateurs de diffusion pour le framework PRC.

PRINCIPE MATHÉMATIQUE:
- Diffusion = propagation de corrélations via moyennage
- AUCUNE structure de graphe présupposée
- Action directe sur la matrice C

TRANSFORMATION GÉNÉRIQUE:
    C'[i,j] = C[i,j] + α * Σ_k (influence de chemins via k)

AUCUNE PHYSIQUE:
- Pas de "particules qui diffusent"
- Pas de "chaleur qui se propage"
- Juste: moyennage pondéré de corrélations
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

class PureDiffusionOperator(ConstrainedOperator):
    """
    Diffusion pure sur la matrice de corrélation.
    
    TRANSFORMATION MATHÉMATIQUE:
        C' = C + α * (C @ C - n * C)
    
    où:
    - C @ C représente l'influence via chemins de longueur 2
    - n = taille de la matrice (normalisation)
    - α = taux de diffusion (petit → lent, grand → rapide)
    
    INTUITION:
    Si I_i corrèle avec I_k, et I_k corrèle avec I_j,
    alors la corrélation I_i-I_j est renforcée.
    
    AUCUNE PRÉSUPPOSITION:
    - Pas de graphe
    - Pas de géométrie spatiale
    - Pas de notion de "voisins"
    """
    
    def __init__(self, alpha: float = 0.01, normalize: bool = True):
        """
        Args:
            alpha: Taux de diffusion (typiquement 0.001 - 0.1)
            normalize: Si True, normalise par la taille pour stabilité
        """
        assert alpha > 0, "alpha doit être > 0"
        self.alpha = alpha
        self.normalize = normalize
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la diffusion directement sur C.
        
        Principe: corrélations se propagent via chemins transitifs.
        """
        n = C.shape[0]
        
        # Influence via chemins de longueur 2
        indirect_influence = C @ C
        
        # Terme de diffusion (spread - dissipation)
        if self.normalize:
            diffusion_term = self.alpha * (indirect_influence - n * C)
        else:
            diffusion_term = self.alpha * indirect_influence
        
        # Nouvelle matrice
        C_next = C + diffusion_term
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "PureDiffusion",
            "alpha": self.alpha,
            "normalize": self.normalize
        }


class AdaptiveDiffusionOperator(ConstrainedOperator):
    """
    Diffusion avec taux adaptatif basé sur la structure locale.
    
    ADAPTATION:
        α_local = α_base * f(corrélations locales)
    
    où f peut être:
    - Fonction de la variance locale
    - Fonction de la densité de corrélation
    - Fonction du rang local
    
    PRINCIPE:
    Diffusion plus rapide dans régions homogènes,
    plus lente dans régions hétérogènes.
    """
    
    def __init__(self, 
                 alpha_base: float = 0.01,
                 adaptation_mode: str = "variance",
                 adaptation_strength: float = 0.5):
        """
        Args:
            alpha_base: Taux de base
            adaptation_mode: "variance", "density", "rank"
            adaptation_strength: Force de l'adaptation (0 = pas d'adaptation)
        """
        assert alpha_base > 0, "alpha_base doit être > 0"
        assert 0 <= adaptation_strength <= 1, "adaptation_strength dans [0,1]"
        assert adaptation_mode in ["variance", "density", "rank"], \
            f"Mode inconnu: {adaptation_mode}"
        
        self.alpha_base = alpha_base
        self.adaptation_mode = adaptation_mode
        self.adaptation_strength = adaptation_strength
    
    def _compute_local_rates(self, C: np.ndarray) -> np.ndarray:
        """
        Calcule les taux de diffusion locaux pour chaque élément.
        
        Returns:
            Matrice NxN de taux locaux
        """
        n = C.shape[0]
        
        if self.adaptation_mode == "variance":
            # Variance locale (ligne + colonne)
            row_var = np.var(C, axis=1, keepdims=True)
            col_var = np.var(C, axis=0, keepdims=True)
            local_metric = (row_var + col_var.T) / 2
            
        elif self.adaptation_mode == "density":
            # Densité de corrélation (somme des valeurs absolues)
            row_density = np.sum(np.abs(C), axis=1, keepdims=True)
            col_density = np.sum(np.abs(C), axis=0, keepdims=True)
            local_metric = (row_density + col_density.T) / (2 * n)
            
        elif self.adaptation_mode == "rank":
            # Approximation locale du rang via SVD partielle
            # (simplifié: utilise norme Frobenius locale)
            row_norm = np.linalg.norm(C, axis=1, keepdims=True)
            col_norm = np.linalg.norm(C, axis=0, keepdims=True)
            local_metric = (row_norm + col_norm.T) / 2
        
        # Normalise dans [0, 1]
        local_metric = local_metric / (np.max(local_metric) + 1e-10)
        
        # Adaptation: alpha_local = alpha_base * (1 + strength * metric)
        alpha_matrix = self.alpha_base * (
            1 + self.adaptation_strength * local_metric
        )
        
        return alpha_matrix
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique diffusion avec taux adaptatifs."""
        n = C.shape[0]
        
        # Calcule taux locaux
        alpha_matrix = self._compute_local_rates(C)
        
        # Influence indirecte
        indirect = C @ C
        
        # Diffusion avec taux adaptatifs (élément par élément)
        diffusion_term = alpha_matrix * (indirect - n * C)
        
        C_next = C + diffusion_term
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "AdaptiveDiffusion",
            "alpha_base": self.alpha_base,
            "mode": self.adaptation_mode,
            "strength": self.adaptation_strength
        }


class MultiScaleDiffusionOperator(ConstrainedOperator):
    """
    Diffusion multi-échelle: combine influences à différentes distances.
    
    TRANSFORMATION:
        C' = C + Σ_k α_k * (C^k - normalisation)
    
    où C^k = C @ C @ ... @ C (k fois) représente chemins de longueur k.
    
    PRINCIPE:
    - k=1: influence directe
    - k=2: chemins de longueur 2
    - k=3: chemins de longueur 3
    - etc.
    
    Combine plusieurs échelles de propagation.
    """
    
    def __init__(self, 
                 scales: list[int] = [2, 3],
                 alphas: Optional[list[float]] = None):
        """
        Args:
            scales: Longueurs de chemins à considérer [2, 3, ...]
            alphas: Taux pour chaque échelle (si None, décroît géométriquement)
        """
        assert all(k >= 1 for k in scales), "Scales doivent être ≥ 1"
        self.scales = sorted(scales)
        
        if alphas is None:
            # Décroissance géométrique: α_k = 0.01 / k
            self.alphas = [0.01 / k for k in self.scales]
        else:
            assert len(alphas) == len(scales), "alphas et scales incompatibles"
            self.alphas = alphas
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique diffusion multi-échelle."""
        n = C.shape[0]
        C_next = C.copy()
        
        for k, alpha in zip(self.scales, self.alphas):
            # Calcule C^k (chemins de longueur k)
            C_power_k = C.copy()
            for _ in range(k - 1):
                C_power_k = C_power_k @ C
            
            # Ajoute contribution de cette échelle
            diffusion_k = alpha * (C_power_k - (n ** (k-1)) * C)
            C_next += diffusion_k
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "MultiScaleDiffusion",
            "scales": self.scales,
            "alphas": self.alphas
        }


class AnisotropicDiffusionOperator(ConstrainedOperator):
    """
    Diffusion anisotrope: taux différent selon la direction dans l'espace des corrélations.
    
    PRINCIPE:
    La diffusion ne se fait pas uniformément, mais préférentiellement
    le long de certaines directions (modes principaux de C).
    
    MÉTHODE:
    1. Décompose C = U Λ U^T (SVD)
    2. Applique diffusion différente sur chaque mode
    3. Reconstruit
    
    Permet de préserver certaines structures tout en lissant d'autres.
    """
    
    def __init__(self,
                 alpha_principal: float = 0.01,
                 alpha_secondary: float = 0.005,
                 n_principal_modes: int = 5):
        """
        Args:
            alpha_principal: Taux pour modes principaux
            alpha_secondary: Taux pour modes secondaires
            n_principal_modes: Nombre de modes considérés principaux
        """
        self.alpha_principal = alpha_principal
        self.alpha_secondary = alpha_secondary
        self.n_principal = n_principal_modes
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique diffusion anisotrope via décomposition spectrale."""
        n = C.shape[0]
        
        # Décomposition en valeurs propres (C symétrique)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Ordonne par magnitude décroissante
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Diffusion anisotrope sur les valeurs propres
        diffused_eigenvalues = eigenvalues.copy()
        
        for i in range(n):
            if i < self.n_principal:
                alpha = self.alpha_principal
            else:
                alpha = self.alpha_secondary
            
            # Diffusion vers la moyenne
            mean_val = np.mean(eigenvalues)
            diffused_eigenvalues[i] += alpha * (mean_val - eigenvalues[i])
        
        # Reconstruction
        C_next = eigenvectors @ np.diag(diffused_eigenvalues) @ eigenvectors.T
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "AnisotropicDiffusion",
            "alpha_principal": self.alpha_principal,
            "alpha_secondary": self.alpha_secondary,
            "n_principal_modes": self.n_principal
        }