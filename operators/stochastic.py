

"""
operators/stochastic.py

Opérateurs stochastiques pour le framework PRC.

PRINCIPE MATHÉMATIQUE:
- Ajout de perturbations aléatoires contrôlées
- Exploration de l'espace des configurations
- Échappement de minima locaux

TRANSFORMATION GÉNÉRIQUE:
    C' = f(C) + σ * ε
    
où ε est un bruit aléatoire (Gaussien, uniforme, etc.)

AUCUNE PHYSIQUE:
- Pas de "température thermodynamique"
- Pas de "fluctuations quantiques"
- Juste: perturbations mathématiques stochastiques
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EvolutionOperator, ConstrainedOperator


class GaussianNoiseOperator(ConstrainedOperator):
    """
    Ajout de bruit Gaussien aux corrélations.
    
    TRANSFORMATION:
        C' = C + σ * N(0, 1)
    
    où N(0, 1) est une matrice de bruit Gaussien standard.
    
    Le bruit est symétrisé pour préserver la symétrie de C.
    """
    
    def __init__(self, sigma: float = 0.01, seed: Optional[int] = None):
        """
        Args:
            sigma: Amplitude du bruit
            seed: Graine aléatoire pour reproductibilité
        """
        assert sigma >= 0, "sigma doit être ≥ 0"
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Ajoute bruit Gaussien symétrique."""
        n = C.shape[0]
        
        # Génère bruit
        noise = self.rng.randn(n, n)
        
        # Symétrise
        noise = (noise + noise.T) / 2
        
        # Annule diagonale (on ne bruite pas l'auto-corrélation)
        np.fill_diagonal(noise, 0)
        
        C_next = C + self.sigma * noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "GaussianNoise",
            "sigma": self.sigma,
            "seed": self.seed
        }


class UniformNoiseOperator(ConstrainedOperator):
    """
    Ajout de bruit uniforme.
    
    TRANSFORMATION:
        C' = C + σ * U(-1, 1)
    
    où U(-1, 1) est une matrice de bruit uniforme dans [-1, 1].
    """
    
    def __init__(self, sigma: float = 0.01, seed: Optional[int] = None):
        """
        Args:
            sigma: Amplitude du bruit
            seed: Graine aléatoire
        """
        assert sigma >= 0, "sigma doit être ≥ 0"
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Ajoute bruit uniforme symétrique."""
        n = C.shape[0]
        
        # Génère bruit uniforme dans [-1, 1]
        noise = self.rng.uniform(-1, 1, size=(n, n))
        
        # Symétrise
        noise = (noise + noise.T) / 2
        
        # Annule diagonale
        np.fill_diagonal(noise, 0)
        
        C_next = C + self.sigma * noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "UniformNoise",
            "sigma": self.sigma,
            "seed": self.seed
        }


class LangevinOperator(ConstrainedOperator):
    """
    Dynamique de type Langevin: gradient + bruit.
    
    TRANSFORMATION:
        C' = C - α * ∇Φ(C) + √(2αT) * ε
    
    où:
    - Φ(C) est une fonction scalaire de C (ex: -Tr(C²))
    - ∇Φ est le gradient
    - T est un paramètre contrôlant l'amplitude du bruit
    - ε est du bruit Gaussien
    
    NOTE: Φ est une construction mathématique, PAS une "énergie libre"
    ou un "potentiel thermodynamique". C'est juste une fonction.
    """
    
    def __init__(self,
                 alpha: float = 0.01,
                 temperature: float = 0.1,
                 potential_type: str = "quadratic",
                 seed: Optional[int] = None):
        """
        Args:
            alpha: Taux de descente
            temperature: Paramètre de bruit (nom technique, pas physique)
            potential_type: "quadratic", "quartic", "trace"
            seed: Graine aléatoire
        """
        assert alpha > 0, "alpha doit être > 0"
        assert temperature >= 0, "temperature doit être ≥ 0"
        assert potential_type in ["quadratic", "quartic", "trace"], \
            f"Type inconnu: {potential_type}"
        
        self.alpha = alpha
        self.temperature = temperature
        self.potential_type = potential_type
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _compute_gradient(self, C: np.ndarray) -> np.ndarray:
        """
        Calcule ∇Φ(C) selon le type de fonction Φ.
        
        Choix de Φ:
        - "quadratic": Φ = -Tr(C²) → ∇Φ = -2C
        - "quartic": Φ = -Tr(C²) + Tr(C⁴) → ∇Φ = -2C + 4C³
        - "trace": Φ = -Tr(C) → ∇Φ = -I
        """
        if self.potential_type == "quadratic":
            return -2 * C
        
        elif self.potential_type == "quartic":
            C_squared = C @ C
            return -2 * C + 4 * (C_squared @ C)
        
        elif self.potential_type == "trace":
            n = C.shape[0]
            grad = -np.eye(n)
            return grad
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique dynamique de Langevin."""
        n = C.shape[0]
        
        # Terme de gradient
        gradient = self._compute_gradient(C)
        
        # Terme de bruit
        noise = self.rng.randn(n, n)
        noise = (noise + noise.T) / 2  # Symétrise
        np.fill_diagonal(noise, 0)  # Pas de bruit sur diagonale
        
        # Mise à jour Langevin
        C_next = C - self.alpha * gradient + np.sqrt(2 * self.alpha * self.temperature) * noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Langevin",
            "alpha": self.alpha,
            "temperature": self.temperature,
            "potential_type": self.potential_type,
            "seed": self.seed
        }


class CorrelatedNoiseOperator(ConstrainedOperator):
    """
    Bruit avec structure de corrélation.
    
    PRINCIPE:
    Au lieu d'ajouter du bruit indépendant, ajoute du bruit qui respecte
    la structure de corrélation existante.
    
    TRANSFORMATION:
        ε_corr = C @ ε @ C
        C' = C + σ * ε_corr
    
    EFFET:
    Les perturbations sont plus fortes là où les corrélations sont déjà fortes.
    """
    
    def __init__(self, sigma: float = 0.01, seed: Optional[int] = None):
        """
        Args:
            sigma: Amplitude du bruit
            seed: Graine aléatoire
        """
        assert sigma >= 0, "sigma doit être ≥ 0"
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Ajoute bruit corrélé."""
        n = C.shape[0]
        
        # Bruit de base
        noise = self.rng.randn(n, n)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        
        # Corrèle le bruit via la structure existante
        correlated_noise = C @ noise @ C
        
        # Renormalise
        noise_norm = np.linalg.norm(correlated_noise)
        if noise_norm > 1e-10:
            correlated_noise = correlated_noise / noise_norm
        
        C_next = C + self.sigma * correlated_noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "CorrelatedNoise",
            "sigma": self.sigma,
            "seed": self.seed
        }


class AnisotropicNoiseOperator(ConstrainedOperator):
    """
    Bruit anisotrope: amplitude variable selon la direction.
    
    PRINCIPE:
    Décompose C en modes propres, ajoute du bruit avec amplitude
    différente sur chaque mode.
    
    EFFET:
    Peut perturber principalement les modes faibles tout en
    préservant les modes principaux, ou vice-versa.
    """
    
    def __init__(self,
                 sigma_principal: float = 0.001,
                 sigma_secondary: float = 0.01,
                 n_principal: int = 5,
                 seed: Optional[int] = None):
        """
        Args:
            sigma_principal: Bruit sur modes principaux
            sigma_secondary: Bruit sur modes secondaires
            n_principal: Nombre de modes principaux
            seed: Graine aléatoire
        """
        assert sigma_principal >= 0 and sigma_secondary >= 0, "sigmas ≥ 0"
        self.sigma_principal = sigma_principal
        self.sigma_secondary = sigma_secondary
        self.n_principal = n_principal
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Ajoute bruit anisotrope dans la base propre."""
        n = C.shape[0]
        
        # Décomposition spectrale
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Ordonne par magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Génère bruit dans la base propre
        noise_eigenspace = self.rng.randn(n)
        
        # Amplitude différente selon mode
        for i in range(n):
            if i < self.n_principal:
                noise_eigenspace[i] *= self.sigma_principal
            else:
                noise_eigenspace[i] *= self.sigma_secondary
        
        # Transforme en bruit dans l'espace original
        noise = eigenvectors @ np.diag(noise_eigenspace) @ eigenvectors.T
        
        C_next = C + noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "AnisotropicNoise",
            "sigma_principal": self.sigma_principal,
            "sigma_secondary": self.sigma_secondary,
            "n_principal": self.n_principal,
            "seed": self.seed
        }


class OrnsteinUhlenbeckOperator(ConstrainedOperator):
    """
    Processus d'Ornstein-Uhlenbeck: bruit avec retour vers une cible.
    
    TRANSFORMATION:
        C' = C + α * (C_target - C) + σ * ε
    
    où:
    - C_target est une configuration cible
    - α contrôle la vitesse de retour
    - σ contrôle l'amplitude du bruit
    
    EFFET:
    Fluctuations autour d'une configuration d'équilibre.
    """
    
    def __init__(self,
                 alpha: float = 0.01,
                 sigma: float = 0.01,
                 target: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Args:
            alpha: Taux de retour vers la cible
            sigma: Amplitude du bruit
            target: Matrice cible (si None, utilise identité)
            seed: Graine aléatoire
        """
        assert alpha > 0, "alpha doit être > 0"
        assert sigma >= 0, "sigma doit être ≥ 0"
        
        self.alpha = alpha
        self.sigma = sigma
        self.target = target  # Sera défini au premier appel si None
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique processus O-U."""
        n = C.shape[0]
        
        # Définit cible si pas encore fait
        if self.target is None:
            self.target = np.eye(n)
        
        # Terme de retour
        return_term = self.alpha * (self.target - C)
        
        # Terme de bruit
        noise = self.rng.randn(n, n)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        
        C_next = C + return_term + self.sigma * noise
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "OrnsteinUhlenbeck",
            "alpha": self.alpha,
            "sigma": self.sigma,
            "has_target": self.target is not None,
            "seed": self.seed
        }