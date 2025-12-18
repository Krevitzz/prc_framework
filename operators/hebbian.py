"""
operators/hebbian.py

Opérateurs de renforcement transitif pour le framework PRC.

PRINCIPE MATHÉMATIQUE:
- Si C[i,k] et C[k,j] sont forts, renforcer C[i,j]
- Transitivité des corrélations
- Terme emprunté à Hebb mais AUCUNE référence neuroscientifique ici

TRANSFORMATION GÉNÉRIQUE:
    C'[i,j] = C[i,j] + β * f(C[i,k] * C[k,j])

AUCUNE PHYSIQUE:
- Pas de "neurones qui apprennent"
- Pas de "mémoire qui se forme"
- Juste: amplification de transitivité
"""

import numpy as np
from typing import Optional, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EvolutionOperator, ConstrainedOperator


class HebbianOperator(ConstrainedOperator):
    """
    Renforcement transitif des corrélations.
    
    RÈGLE:
        C'[i,j] = C[i,j] + β * Σ_k C[i,k] * C[k,j]
        
    Équivalent à:
        C' = C + β * (C @ C)
    
    EFFET:
    - Corrélations fortes deviennent plus fortes
    - Corrélations faibles restent faibles
    - Création de clusters (groupes fortement corrélés)
    """
    
    def __init__(self, beta: float = 0.01, threshold: float = 0.0):
        """
        Args:
            beta: Taux de renforcement (petit → lent, grand → rapide)
            threshold: Seuil en dessous duquel pas de renforcement
        """
        assert beta > 0, "beta doit être > 0"
        assert 0 <= threshold < 1, "threshold dans [0, 1)"
        
        self.beta = beta
        self.threshold = threshold
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique renforcement transitif."""
        # Calcule le produit C @ C (transitivité)
        transitive = C @ C
        
        # Applique seuil si spécifié
        if self.threshold > 0:
            # Renforce seulement où corrélation > seuil
            mask = np.abs(C) > self.threshold
            reinforcement = np.where(mask, self.beta * transitive, 0)
        else:
            reinforcement = self.beta * transitive
        
        C_next = C + reinforcement
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "Hebbian",
            "beta": self.beta,
            "threshold": self.threshold
        }


class AntiHebbianOperator(ConstrainedOperator):
    """
    Anti-renforcement: affaiblit les corrélations transitives.
    
    RÈGLE:
        C'[i,j] = C[i,j] - β * Σ_k C[i,k] * C[k,j]
    
    EFFET:
    - Corrélations fortes s'affaiblissent
    - Favorise la décorrélation
    - Peut créer des patterns oscillatoires
    """
    
    def __init__(self, beta: float = 0.01):
        """
        Args:
            beta: Taux d'affaiblissement
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique anti-renforcement."""
        transitive = C @ C
        C_next = C - self.beta * transitive
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "AntiHebbian",
            "beta": self.beta
        }


class NonlinearHebbianOperator(ConstrainedOperator):
    """
    Renforcement avec non-linéarité.
    
    RÈGLE:
        C'[i,j] = C[i,j] + β * f(C[i,k] * C[k,j])
    
    où f est une fonction non-linéaire:
    - sigmoid: limite la croissance
    - tanh: symétrique avec saturation
    - power: f(x) = sign(x) * |x|^p
    
    EFFET:
    Évite la croissance explosive tout en renforçant la transitivité.
    """
    
    def __init__(self,
                 beta: float = 0.01,
                 nonlinearity: str = "tanh",
                 scale: float = 1.0):
        """
        Args:
            beta: Taux de renforcement
            nonlinearity: "tanh", "sigmoid", "power", "cubic"
            scale: Facteur d'échelle pour la non-linéarité
        """
        assert beta > 0, "beta doit être > 0"
        assert nonlinearity in ["tanh", "sigmoid", "power", "cubic"], \
            f"Nonlinearity inconnue: {nonlinearity}"
        
        self.beta = beta
        self.nonlinearity = nonlinearity
        self.scale = scale
    
    def _apply_nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """Applique la fonction non-linéaire."""
        if self.nonlinearity == "tanh":
            return np.tanh(self.scale * x)
        
        elif self.nonlinearity == "sigmoid":
            return 2 / (1 + np.exp(-self.scale * x)) - 1  # Centré en 0
        
        elif self.nonlinearity == "power":
            # f(x) = sign(x) * |x|^(1 + scale)
            power = 1.0 + self.scale
            return np.sign(x) * np.abs(x) ** power
        
        elif self.nonlinearity == "cubic":
            return x - self.scale * x**3
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique renforcement non-linéaire."""
        transitive = C @ C
        
        # Applique non-linéarité
        nonlinear_term = self._apply_nonlinearity(transitive)
        
        C_next = C + self.beta * nonlinear_term
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "NonlinearHebbian",
            "beta": self.beta,
            "nonlinearity": self.nonlinearity,
            "scale": self.scale
        }


class CompetitiveHebbianOperator(ConstrainedOperator):
    """
    Renforcement compétitif: gagnants renforcés, perdants affaiblis.
    
    PRINCIPE:
    Pour chaque DOF, renforce les k corrélations les plus fortes,
    affaiblit les autres.
    
    EFFET:
    - Crée des structures winner-take-all
    - Sparsifie la matrice de corrélation
    - Favorise l'émergence de groupes distincts
    """
    
    def __init__(self,
                 beta_win: float = 0.02,
                 beta_lose: float = 0.01,
                 k_winners: int = 5):
        """
        Args:
            beta_win: Taux de renforcement pour gagnants
            beta_lose: Taux d'affaiblissement pour perdants
            k_winners: Nombre de corrélations à renforcer par DOF
        """
        assert beta_win > 0 and beta_lose > 0, "betas doivent être > 0"
        assert k_winners > 0, "k_winners doit être > 0"
        
        self.beta_win = beta_win
        self.beta_lose = beta_lose
        self.k_winners = k_winners
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique renforcement compétitif."""
        n = C.shape[0]
        C_next = C.copy()
        
        # Pour chaque ligne (DOF)
        for i in range(n):
            # Trouve les k corrélations les plus fortes (hors diagonale)
            row = np.abs(C[i, :])
            row[i] = -np.inf  # Exclut diagonale
            
            # Indices des k winners
            k = min(self.k_winners, n - 1)
            winner_indices = np.argsort(row)[-k:]
            
            # Masque winners vs losers
            winner_mask = np.zeros(n, dtype=bool)
            winner_mask[winner_indices] = True
            winner_mask[i] = False  # Exclut diagonale
            
            # Calcule transitivité
            transitive_i = C[i, :] @ C
            
            # Renforce winners
            C_next[i, winner_mask] += self.beta_win * transitive_i[winner_mask]
            
            # Affaiblit losers
            loser_mask = ~winner_mask
            loser_mask[i] = False  # Exclut diagonale
            C_next[i, loser_mask] -= self.beta_lose * np.abs(transitive_i[loser_mask])
        
        # Symétrise
        C_next = (C_next + C_next.T) / 2
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "CompetitiveHebbian",
            "beta_win": self.beta_win,
            "beta_lose": self.beta_lose,
            "k_winners": self.k_winners
        }


class OjaRuleOperator(ConstrainedOperator):
    """
    Règle d'Oja: renforcement avec normalisation.
    
    TRANSFORMATION:
        C'[i,j] = C[i,j] + β * (transitive[i,j] - C[i,j] * ||C[i]||²)
    
    PRINCIPE:
    Le terme de normalisation -C[i,j] * ||C[i]||² empêche la croissance illimitée
    tout en préservant les directions principales.
    
    EFFET:
    - Converge vers les modes principaux de corrélation
    - Équivalent à une forme d'analyse en composantes principales
    """
    
    def __init__(self, beta: float = 0.01):
        """
        Args:
            beta: Taux d'apprentissage
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique règle d'Oja."""
        n = C.shape[0]
        
        # Transitivité
        transitive = C @ C
        
        # Normes carrées des lignes
        row_norms_sq = np.sum(C ** 2, axis=1, keepdims=True)
        
        # Terme de normalisation
        normalization = C * row_norms_sq
        
        # Mise à jour
        C_next = C + self.beta * (transitive - normalization)
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "OjaRule",
            "beta": self.beta
        }


class BCMRuleOperator(ConstrainedOperator):
    """
    Règle BCM (Bienenstock-Cooper-Munro): seuil adaptatif.
    
    PRINCIPE:
    - Si corrélation > seuil_local: renforcement
    - Si corrélation < seuil_local: affaiblissement
    - Seuil s'adapte dynamiquement
    
    TRANSFORMATION:
        seuil[i] = moyenne(C[i,:]²)
        C'[i,j] = C[i,j] + β * transitive[i,j] * (C[i,j] - seuil[i])
    
    EFFET:
    - Auto-stabilisation
    - Préserve diversité des patterns
    - Évite domination d'un seul mode
    """
    
    def __init__(self, beta: float = 0.01, threshold_adaptation: float = 1.0):
        """
        Args:
            beta: Taux de renforcement
            threshold_adaptation: Vitesse d'adaptation du seuil
        """
        assert beta > 0, "beta doit être > 0"
        self.beta = beta
        self.threshold_adaptation = threshold_adaptation
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique règle BCM."""
        n = C.shape[0]
        
        # Calcule seuils adaptatifs (par ligne)
        thresholds = self.threshold_adaptation * np.mean(C ** 2, axis=1, keepdims=True)
        
        # Transitivité
        transitive = C @ C
        
        # Terme modulé par distance au seuil
        modulation = C - thresholds
        
        # Mise à jour
        C_next = C + self.beta * transitive * modulation
        
        return C_next
    
    def get_parameters(self) -> dict:
        return {
            "type": "BCMRule",
            "beta": self.beta,
            "threshold_adaptation": self.threshold_adaptation
        }