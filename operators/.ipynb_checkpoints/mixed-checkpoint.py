"""
operators/mixed.py

Opérateurs mixtes combinant plusieurs mécanismes pour le framework PRC.

PRINCIPE:
- Composition sophistiquée de transformations élémentaires
- Équilibre entre exploration (bruit) et exploitation (renforcement)
- Paramètres auto-adaptatifs

CES OPÉRATEURS SONT DES CANDIDATES Γ TESTABLES.
Ils combinent les transformations de base de manière cohérente.

AUCUNE PHYSIQUE:
- Pas de "lois naturelles"
- Pas de "principes thermodynamiques"
- Juste: compositions mathématiques de règles simples
"""

import numpy as np
from typing import Optional, List, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EvolutionOperator, ConstrainedOperator, CompositeOperator
from .diffusion import PureDiffusionOperator
from .hebbian import HebbianOperator, CompetitiveHebbianOperator
from .stochastic import GaussianNoiseOperator, LangevinOperator
from .nonlinear import ThresholdOperator, SaturationOperator, NormalizationOperator


# ============================================================================
# OPÉRATEURS MIXTES PRÉDÉFINIS
# ============================================================================

class BalancedMixedOperator(ConstrainedOperator):
    """
    Opérateur mixte équilibré: diffusion + Hebbian + saturation.
    
    COMPOSITION:
        1. Diffusion: propage les corrélations
        2. Hebbian: renforce la transitivité
        3. Saturation: limite la croissance
    
    EFFET:
    - Patterns émergent via diffusion et renforcement
    - Saturation prévient l'explosion
    - Équilibre stable entre expansion et contraction
    """
    
    def __init__(self,
                 alpha_diffusion: float = 0.01,
                 beta_hebbian: float = 0.01,
                 beta_saturation: float = 2.0,
                 normalize_each_step: bool = True):
        """
        Args:
            alpha_diffusion: Taux de diffusion
            beta_hebbian: Taux de renforcement Hebbian
            beta_saturation: Facteur de saturation (tanh)
            normalize_each_step: Normaliser après chaque sous-étape
        """
        self.alpha_diffusion = alpha_diffusion
        self.beta_hebbian = beta_hebbian
        self.beta_saturation = beta_saturation
        self.normalize_each_step = normalize_each_step
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique séquentiellement: diffusion → Hebbian → saturation."""
        n = C.shape[0]
        
        # 1. DIFFUSION
        indirect = C @ C
        C_temp = C + self.alpha_diffusion * (indirect - n * C)
        
        if self.normalize_each_step:
            from core import enforce_correlation_invariants
            C_temp = enforce_correlation_invariants(C_temp)
        
        # 2. HEBBIAN
        transitive = C_temp @ C_temp
        C_temp = C_temp + self.beta_hebbian * transitive
        
        if self.normalize_each_step:
            C_temp = enforce_correlation_invariants(C_temp)
        
        # 3. SATURATION
        C_next = np.tanh(self.beta_saturation * C_temp)
        np.fill_diagonal(C_next, 1.0)
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "BalancedMixed",
            "alpha_diffusion": self.alpha_diffusion,
            "beta_hebbian": self.beta_hebbian,
            "beta_saturation": self.beta_saturation,
            "normalize_each_step": self.normalize_each_step
        }


class ExplorativeOperator(ConstrainedOperator):
    """
    Opérateur exploratoire: diffusion + bruit + anti-concentration.
    
    OBJECTIF:
    Favorise l'exploration de l'espace des configurations.
    Évite la convergence prématurée vers un seul pattern.
    
    COMPOSITION:
        1. Diffusion: homogénéise
        2. Bruit: perturbe
        3. Anti-concentration: pénalise les structures trop stables
    """
    
    def __init__(self,
                 alpha_diffusion: float = 0.02,
                 sigma_noise: float = 0.01,
                 lambda_anticonc: float = 0.005,
                 seed: Optional[int] = None):
        """
        Args:
            alpha_diffusion: Taux de diffusion
            sigma_noise: Amplitude du bruit
            lambda_anticonc: Force de l'anti-concentration
            seed: Graine aléatoire
        """
        self.alpha_diffusion = alpha_diffusion
        self.sigma_noise = sigma_noise
        self.lambda_anticonc = lambda_anticonc
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique exploration."""
        n = C.shape[0]
        
        # 1. DIFFUSION
        indirect = C @ C
        C_temp = C + self.alpha_diffusion * (indirect - n * C)
        
        # 2. BRUIT
        noise = self.rng.randn(n, n)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        C_temp = C_temp + self.sigma_noise * noise
        
        # 3. ANTI-CONCENTRATION
        # Pénalise variance trop faible (structure trop homogène)
        row_var = np.var(C_temp, axis=1, keepdims=True)
        target_var = 0.3  # Variance cible
        anticonc_term = self.lambda_anticonc * (target_var - row_var)
        
        # Applique perturbation proportionnelle à l'écart de variance
        perturbation = anticonc_term * self.rng.randn(n, n)
        perturbation = (perturbation + perturbation.T) / 2
        np.fill_diagonal(perturbation, 0)
        
        C_next = C_temp + perturbation
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Explorative",
            "alpha_diffusion": self.alpha_diffusion,
            "sigma_noise": self.sigma_noise,
            "lambda_anticonc": self.lambda_anticonc,
            "seed": self.seed
        }


class CompetitiveOperator(ConstrainedOperator):
    """
    Opérateur compétitif: Hebbian compétitif + seuillage + normalisation.
    
    OBJECTIF:
    Favorise l'émergence de structures distinctes en compétition.
    Winner-take-all dynamics.
    
    COMPOSITION:
        1. Hebbian compétitif: gagnants renforcés, perdants affaiblis
        2. Seuillage: supprime corrélations faibles
        3. Normalisation: maintient l'échelle
    """
    
    def __init__(self,
                 beta_win: float = 0.02,
                 beta_lose: float = 0.01,
                 k_winners: int = 5,
                 threshold: float = 0.1):
        """
        Args:
            beta_win: Renforcement des gagnants
            beta_lose: Affaiblissement des perdants
            k_winners: Nombre de gagnants par DOF
            threshold: Seuil de coupure
        """
        self.beta_win = beta_win
        self.beta_lose = beta_lose
        self.k_winners = k_winners
        self.threshold = threshold
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique dynamique compétitive."""
        n = C.shape[0]
        C_temp = C.copy()
        
        # 1. HEBBIAN COMPÉTITIF
        for i in range(n):
            row = np.abs(C[i, :])
            row[i] = -np.inf
            
            k = min(self.k_winners, n - 1)
            winner_indices = np.argsort(row)[-k:]
            
            winner_mask = np.zeros(n, dtype=bool)
            winner_mask[winner_indices] = True
            winner_mask[i] = False
            
            transitive_i = C[i, :] @ C
            
            # Renforce gagnants
            C_temp[i, winner_mask] += self.beta_win * transitive_i[winner_mask]
            
            # Affaiblit perdants
            loser_mask = ~winner_mask
            loser_mask[i] = False
            C_temp[i, loser_mask] -= self.beta_lose * np.abs(transitive_i[loser_mask])
        
        # Symétrise
        C_temp = (C_temp + C_temp.T) / 2
        
        # 2. SEUILLAGE
        mask = np.abs(C_temp) > self.threshold
        np.fill_diagonal(mask, True)
        C_temp = np.where(mask, C_temp, 0)
        
        # 3. NORMALISATION
        norm = np.linalg.norm(C_temp, 'fro')
        if norm > 1e-10:
            C_next = C_temp / norm * np.sqrt(n)
        else:
            C_next = C_temp
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Competitive",
            "beta_win": self.beta_win,
            "beta_lose": self.beta_lose,
            "k_winners": self.k_winners,
            "threshold": self.threshold
        }


class AdaptiveOperator(ConstrainedOperator):
    """
    Opérateur adaptatif: paramètres qui s'ajustent selon l'état de C.
    
    PRINCIPE:
    - Diffusion forte si variance élevée
    - Hebbian fort si corrélations moyennes faibles
    - Bruit inversement proportionnel à la structure
    
    ADAPTATION AUTOMATIQUE sans objectif prédéfini.
    """
    
    def __init__(self,
                 alpha_base: float = 0.01,
                 beta_base: float = 0.01,
                 sigma_base: float = 0.01,
                 adaptation_rate: float = 0.1,
                 seed: Optional[int] = None):
        """
        Args:
            alpha_base: Diffusion de base
            beta_base: Hebbian de base
            sigma_base: Bruit de base
            adaptation_rate: Vitesse d'adaptation
            seed: Graine aléatoire
        """
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.sigma_base = sigma_base
        self.adaptation_rate = adaptation_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _compute_adaptive_params(self, C: np.ndarray) -> tuple:
        """
        Calcule les paramètres adaptatifs selon l'état actuel.
        
        Returns:
            (alpha_adapted, beta_adapted, sigma_adapted)
        """
        n = C.shape[0]
        
        # Métriques de l'état actuel
        mean_corr = np.mean(np.abs(C[~np.eye(n, dtype=bool)]))
        var_corr = np.var(C[~np.eye(n, dtype=bool)])
        sparsity = np.mean(np.abs(C[~np.eye(n, dtype=bool)]) < 0.1)
        
        # ADAPTATION:
        # 1. Diffusion ∝ variance (homogénéise si hétérogène)
        alpha = self.alpha_base * (1 + self.adaptation_rate * var_corr)
        
        # 2. Hebbian ∝ (1 - mean_corr) (renforce si corrélations faibles)
        beta = self.beta_base * (1 + self.adaptation_rate * (1 - mean_corr))
        
        # 3. Bruit ∝ (1 - sparsity) (perturbe si dense)
        sigma = self.sigma_base * (1 + self.adaptation_rate * (1 - sparsity))
        
        return alpha, beta, sigma
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique transformation avec paramètres adaptatifs."""
        n = C.shape[0]
        
        # Calcule paramètres adaptatifs
        alpha, beta, sigma = self._compute_adaptive_params(C)
        
        # 1. DIFFUSION adaptative
        indirect = C @ C
        C_temp = C + alpha * (indirect - n * C)
        
        # 2. HEBBIAN adaptatif
        transitive = C_temp @ C_temp
        C_temp = C_temp + beta * transitive
        
        # 3. BRUIT adaptatif
        noise = self.rng.randn(n, n)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        C_next = C_temp + sigma * noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Adaptive",
            "alpha_base": self.alpha_base,
            "beta_base": self.beta_base,
            "sigma_base": self.sigma_base,
            "adaptation_rate": self.adaptation_rate,
            "seed": self.seed
        }


# ============================================================================
# CONSTRUCTEUR PARAMÉTRIQUE
# ============================================================================

class ParametricMixedOperator(ConstrainedOperator):
    """
    Opérateur mixte entièrement paramétrable.
    
    Permet de construire n'importe quelle combinaison:
        C' = C + Σ_i weight_i * transformation_i(C)
    
    USAGE:
        >>> op = ParametricMixedOperator(
        ...     components=[
        ...         ("diffusion", 0.01),
        ...         ("hebbian", 0.01),
        ...         ("saturation", 2.0)
        ...     ]
        ... )
    """
    
    def __init__(self,
                 components: List[tuple],
                 normalize_weights: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            components: Liste de (nom, paramètre) pour chaque composante
            normalize_weights: Normaliser les poids pour sommer à 1
            seed: Graine aléatoire (pour composantes stochastiques)
        """
        self.components = components
        self.normalize_weights = normalize_weights
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None
        
        # Normalise les poids si demandé
        if normalize_weights and len(components) > 1:
            weights = [param for _, param in components]
            total = sum(abs(w) for w in weights)
            if total > 0:
                self.components = [(name, param / total) 
                                  for (name, param) in components]
    
    def _apply_component(self, C: np.ndarray, name: str, param: float) -> np.ndarray:
        """
        Applique une transformation nommée avec son paramètre.
        
        Args:
            C: Matrice actuelle
            name: Nom de la transformation
            param: Paramètre (alpha, beta, sigma, etc.)
        
        Returns:
            Terme à ajouter à C (pas C_next directement)
        """
        n = C.shape[0]
        
        if name == "diffusion":
            indirect = C @ C
            return param * (indirect - n * C)
        
        elif name == "hebbian":
            transitive = C @ C
            return param * transitive
        
        elif name == "noise":
            if self.rng is not None:
                noise = self.rng.randn(n, n)
                noise = (noise + noise.T) / 2
                np.fill_diagonal(noise, 0)
                return param * noise
            else:
                return np.zeros_like(C)
        
        elif name == "saturation":
            # Saturation: renvoie C transformé, pas un terme additif
            return np.tanh(param * C) - C
        
        elif name == "threshold":
            # Seuillage: supprime valeurs < param
            mask = np.abs(C) > param
            np.fill_diagonal(mask, True)
            return np.where(mask, C, 0) - C
        
        elif name == "decay":
            # Décroissance vers identité
            return param * (np.eye(n) - C)
        
        else:
            raise ValueError(f"Composante inconnue: {name}")
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique toutes les composantes."""
        C_next = C.copy()
        
        for name, param in self.components:
            term = self._apply_component(C_next, name, param)
            C_next = C_next + term
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "ParametricMixed",
            "components": self.components,
            "normalize_weights": self.normalize_weights,
            "seed": self.seed
        }