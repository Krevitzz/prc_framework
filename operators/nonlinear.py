"""
operators/nonlinear.py

Opérateurs non-linéaires pour le framework PRC.

PRINCIPE MATHÉMATIQUE:
- Transformations non-linéaires des corrélations
- Seuillage, saturation, activation
- Dynamiques riches (bifurcations, attracteurs)

TRANSFORMATION GÉNÉRIQUE:
    C' = f(C)
    
où f est une fonction non-linéaire élément par élément ou matricielle.

AUCUNE PHYSIQUE:
- Pas de "potentiels d'interaction"
- Pas de "transitions de phase" (terme descriptif OK, pas explicatif)
- Juste: fonctions mathématiques non-linéaires
"""

import numpy as np
from typing import Callable, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EvolutionOperator, ConstrainedOperator


class ThresholdOperator(ConstrainedOperator):
    """
    Seuillage des corrélations.
    
    TRANSFORMATION:
        C'[i,j] = C[i,j]  si |C[i,j]| > θ
        C'[i,j] = 0       sinon
    
    EFFET:
    - Supprime les corrélations faibles
    - Sparsifie la matrice
    - Crée des groupes distincts
    """
    
    def __init__(self, threshold: float = 0.1, soft: bool = False):
        """
        Args:
            threshold: Seuil de coupure
            soft: Si True, utilise seuillage doux (tanh)
        """
        assert 0 <= threshold < 1, "threshold dans [0, 1)"
        self.threshold = threshold
        self.soft = soft
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique seuillage."""
        if self.soft:
            # Seuillage doux: transition graduelle
            scale = 10 / (self.threshold + 1e-10)
            C_next = C * np.tanh(scale * (np.abs(C) - self.threshold))
        else:
            # Seuillage dur
            mask = np.abs(C) > self.threshold
            C_next = np.where(mask, C, 0)
        
        # Préserve diagonale
        np.fill_diagonal(C_next, 1.0)
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Threshold",
            "threshold": self.threshold,
            "soft": self.soft
        }


class SaturationOperator(ConstrainedOperator):
    """
    Saturation des corrélations.
    
    TRANSFORMATION:
        C'[i,j] = tanh(β * C[i,j])
    
    EFFET:
    - Limite la croissance des corrélations
    - Converge vers ±1
    - Préserve le signe
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: Facteur de saturation (grand → saturation rapide)
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique saturation."""
        C_next = np.tanh(self.beta * C)
        
        # Préserve diagonale = 1
        np.fill_diagonal(C_next, 1.0)
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Saturation",
            "beta": self.beta
        }


class PolynomialOperator(ConstrainedOperator):
    """
    Transformation polynomiale.
    
    TRANSFORMATION:
        C' = Σ_k a_k * C^k
    
    où C^k = C élément par élément à la puissance k.
    
    EFFET:
    Selon les coefficients:
    - Renforce corrélations fortes
    - Affaiblit corrélations faibles
    - Peut créer bifurcations
    """
    
    def __init__(self, coefficients: list[float] = [1.0, 0.0, -0.1]):
        """
        Args:
            coefficients: [a_0, a_1, a_2, ...] pour Σ a_k * C^k
        """
        assert len(coefficients) > 0, "Au moins un coefficient requis"
        self.coefficients = coefficients
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique transformation polynomiale."""
        C_next = np.zeros_like(C)
        C_power = np.ones_like(C)  # C^0 = 1
        
        for k, coeff in enumerate(self.coefficients):
            if k > 0:
                C_power = C_power * C  # C^k
            
            C_next += coeff * C_power
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Polynomial",
            "coefficients": self.coefficients,
            "degree": len(self.coefficients) - 1
        }


class LogisticOperator(ConstrainedOperator):
    """
    Carte logistique appliquée aux corrélations.
    
    TRANSFORMATION:
        C'[i,j] = r * C[i,j] * (1 - C[i,j])
    
    où les corrélations sont d'abord mappées dans [0, 1].
    
    EFFET:
    - Peut produire dynamiques chaotiques pour certains r
    - Bifurcations et attracteurs complexes
    - Sensibilité aux conditions initiales
    """
    
    def __init__(self, r: float = 3.0):
        """
        Args:
            r: Paramètre logistique (1 < r < 4 pour stabilité)
        """
        assert 1 < r < 4, "r doit être dans (1, 4) pour stabilité"
        self.r = r
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique carte logistique."""
        # Mappe [-1, 1] → [0, 1]
        C_normalized = (C + 1) / 2
        
        # Applique logistique
        C_transformed = self.r * C_normalized * (1 - C_normalized)
        
        # Remappe [0, 1] → [-1, 1]
        C_next = 2 * C_transformed - 1
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Logistic",
            "r": self.r
        }


class PowerLawOperator(ConstrainedOperator):
    """
    Transformation en loi de puissance.
    
    TRANSFORMATION:
        C'[i,j] = sign(C[i,j]) * |C[i,j]|^p
    
    EFFET:
    - p > 1: renforce corrélations fortes, affaiblit faibles
    - p < 1: égalise les corrélations
    - p = 1: identité
    """
    
    def __init__(self, power: float = 2.0):
        """
        Args:
            power: Exposant de la loi de puissance
        """
        assert power > 0, "power doit être > 0"
        self.power = power
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique loi de puissance."""
        C_next = np.sign(C) * np.abs(C) ** self.power
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "PowerLaw",
            "power": self.power
        }


class RectifiedOperator(ConstrainedOperator):
    """
    Rectification (type ReLU).
    
    TRANSFORMATION:
        C'[i,j] = max(0, C[i,j])  ou
        C'[i,j] = C[i,j] * (C[i,j] > 0)
    
    EFFET:
    - Supprime corrélations négatives
    - Crée asymétrie
    - Sparsifie si beaucoup de valeurs négatives
    """
    
    def __init__(self, negative_slope: float = 0.0):
        """
        Args:
            negative_slope: Pente pour valeurs négatives (0 = ReLU pur)
        """
        assert 0 <= negative_slope <= 1, "negative_slope dans [0, 1]"
        self.negative_slope = negative_slope
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique rectification."""
        C_next = np.where(
            C > 0,
            C,
            self.negative_slope * C
        )
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Rectified",
            "negative_slope": self.negative_slope
        }


class ContrastEnhancementOperator(ConstrainedOperator):
    """
    Augmentation du contraste.
    
    TRANSFORMATION:
        C' = sign(C) * (|C| - μ) * γ
    
    où μ est la moyenne et γ est le gain.
    
    EFFET:
    - Amplifie écarts par rapport à la moyenne
    - Rend structure plus visible
    - Peut créer bistabilité
    """
    
    def __init__(self, gain: float = 2.0, center: Optional[float] = None):
        """
        Args:
            gain: Facteur d'amplification
            center: Centre du contraste (si None, utilise moyenne)
        """
        assert gain > 0, "gain doit être > 0"
        self.gain = gain
        self.center = center
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique augmentation de contraste."""
        # Détermine centre
        if self.center is None:
            center = np.mean(C)
        else:
            center = self.center
        
        # Augmente contraste
        C_next = center + self.gain * (C - center)
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "ContrastEnhancement",
            "gain": self.gain,
            "center": self.center
        }


class NormalizationOperator(ConstrainedOperator):
    """
    Normalisation de la matrice.
    
    MÉTHODES:
    - "frobenius": Normalise par norme de Frobenius
    - "max": Normalise par valeur max
    - "rows": Normalise chaque ligne
    - "spectral": Normalise par norme spectrale
    
    EFFET:
    - Maintient échelle constante
    - Prévient divergence
    - Préserve structure relative
    """
    
    def __init__(self, method: str = "frobenius", target_norm: float = 1.0):
        """
        Args:
            method: Type de normalisation
            target_norm: Norme cible
        """
        assert method in ["frobenius", "max", "rows", "spectral"], \
            f"Méthode inconnue: {method}"
        assert target_norm > 0, "target_norm doit être > 0"
        
        self.method = method
        self.target_norm = target_norm
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique normalisation."""
        if self.method == "frobenius":
            current_norm = np.linalg.norm(C, 'fro')
            C_next = C * (self.target_norm / (current_norm + 1e-10))
        
        elif self.method == "max":
            current_max = np.max(np.abs(C))
            C_next = C * (self.target_norm / (current_max + 1e-10))
        
        elif self.method == "rows":
            row_norms = np.linalg.norm(C, axis=1, keepdims=True)
            C_next = C * (self.target_norm / (row_norms + 1e-10))
        
        elif self.method == "spectral":
            eigenvalues = np.linalg.eigvalsh(C)
            spectral_norm = np.max(np.abs(eigenvalues))
            C_next = C * (self.target_norm / (spectral_norm + 1e-10))
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Normalization",
            "method": self.method,
            "target_norm": self.target_norm
        }


class WinnerTakeAllOperator(ConstrainedOperator):
    """
    Winner-take-all: garde seulement les k corrélations les plus fortes.
    
    TRANSFORMATION:
    Pour chaque ligne, met à zéro tout sauf les k plus grandes valeurs.
    
    EFFET:
    - Sparsification extrême
    - Crée compétition
    - Structures très distinctes
    """
    
    def __init__(self, k: int = 3):
        """
        Args:
            k: Nombre de "gagnants" par ligne
        """
        assert k > 0, "k doit être > 0"
        self.k = k
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique winner-take-all."""
        n = C.shape[0]
        C_next = np.zeros_like(C)
        
        for i in range(n):
            # Trouve les k plus grandes valeurs (en valeur absolue)
            row = np.abs(C[i, :])
            row[i] = -np.inf  # Exclut diagonale
            
            k_actual = min(self.k, n - 1)
            threshold_idx = np.argsort(row)[-k_actual]
            threshold = row[threshold_idx]
            
            # Garde seulement valeurs > seuil
            mask = np.abs(C[i, :]) >= threshold
            mask[i] = True  # Garde diagonale
            
            C_next[i, :] = np.where(mask, C[i, :], 0)
        
        # Symétrise
        C_next = (C_next + C_next.T) / 2
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "WinnerTakeAll",
            "k": self.k
        }