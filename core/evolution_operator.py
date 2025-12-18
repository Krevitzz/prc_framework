"""
core/evolution_operator.py

Interface abstraite pour les opérateurs d'évolution Γ dans le framework PRC.

PRINCIPES FONDAMENTAUX (Thèse Ch. 2.3, 3.3):
- Γ transforme C en C' de manière aveugle
- Γ n'optimise rien, ne sélectionne rien, n'a aucun but
- Γ applique une règle fixe: C_{n+1} = Γ(C_n)
- Implémentation = responsabilité de l'utilisateur
- Le core fournit uniquement l'interface abstraite

ANTI-PATTERNS À ÉVITER:
- "Γ minimise l'énergie" ✗ → Γ transforme C selon sa règle
- "Γ sélectionne les patterns stables" ✗ → Patterns persistent ou se dissolvent
- "Γ optimise..." ✗ → Γ applique une transformation fixe
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class EvolutionOperator(ABC):
    """
    Interface abstraite pour les opérateurs d'évolution Γ.
    
    Un opérateur d'évolution est une transformation:
        Γ: C_n → C_{n+1}
    
    où C est une matrice de corrélation.
    
    PRINCIPES D'IMPLÉMENTATION:
    
    1. AVEUGLEMENT (Thèse 2.3.2)
       - Γ ne "sait" rien des patterns qu'il pourrait créer
       - Γ applique une règle fixe, point
       - Les patterns émergent statistiquement, pas par design
    
    2. UNIVERSALITÉ (Thèse 2.3.3)
       - Le même Γ doit fonctionner sur tout encodage D^(X)
       - Pas de paramètres domaine-spécifiques dans Γ
       - Seule exception: paramètres de la règle elle-même (α, β, etc.)
    
    3. LOCALITÉ (Thèse 3.3.2)
       - Γ devrait agir principalement sur corrélations proches
       - "Proche" = corrélation forte, pas distance géométrique
       - Localité stricte pas obligatoire, mais recommandée pour efficacité
    
    4. PRÉSERVATION DES INVARIANTS
       - Γ(C) doit rester une matrice de corrélation valide
       - Symétrie préservée
       - Diagonale = 1 préservée
       - Éléments dans [-1, 1] préservés
    """
    
    @abstractmethod
    def apply(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la transformation: C_{n+1} = Γ(C_n)
        
        RESPONSABILITÉS DE L'IMPLÉMENTATION:
        - Préserver la symétrie de C
        - Préserver diagonale = 1
        - Maintenir éléments dans [-1, 1]
        - Documentation de la règle appliquée
        
        Args:
            C: Matrice de corrélation à l'instant n
               Shape (N, N), symétrique, diag = 1, éléments dans [-1, 1]
        
        Returns:
            C': Matrice de corrélation à l'instant n+1
                Doit satisfaire les mêmes invariants que C
        
        Raises:
            AssertionError: Si les invariants sont violés dans C'
        
        Note:
            Cette méthode ne doit PAS modifier C en place.
            Elle doit retourner une nouvelle matrice.
        """
        pass
    
    def validate_output(self, C_out: np.ndarray) -> None:
        """
        Valide que la sortie de Γ est une matrice de corrélation valide.
        
        Cette méthode peut être appelée en mode debug pour vérifier
        que l'implémentation de apply() respecte les invariants.
        
        Args:
            C_out: Matrice sortie de apply()
        
        Raises:
            AssertionError: Si un invariant est violé
        """
        n = C_out.shape[0]
        
        # Symétrie
        assert np.allclose(C_out, C_out.T, rtol=1e-5, atol=1e-8), \
            "Γ(C) doit être symétrique"
        
        # Diagonale
        diag = np.diag(C_out)
        assert np.allclose(diag, 1.0, rtol=1e-5, atol=1e-8), \
            f"Diagonale de Γ(C) doit être 1, reçu range [{diag.min():.3f}, {diag.max():.3f}]"
        
        # Bornes
        assert np.all(C_out >= -1.0 - 1e-8) and np.all(C_out <= 1.0 + 1e-8), \
            f"Éléments de Γ(C) hors de [-1, 1], range [{C_out.min():.3f}, {C_out.max():.3f}]"
    
    def __call__(self, C: np.ndarray) -> np.ndarray:
        """
        Syntaxe alternative: gamma(C) au lieu de gamma.apply(C)
        
        Args:
            C: Matrice de corrélation
        
        Returns:
            Γ(C)
        """
        return self.apply(C)
    
    def iterate(self, C: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Applique Γ itérativement n fois: Γ^n(C)
        
        Utilitaire pour tests rapides sans créer de PRCKernel.
        
        Args:
            C: Matrice de corrélation initiale
            n_steps: Nombre d'itérations
        
        Returns:
            Γ^n(C) = Γ(Γ(...Γ(C)...))
        """
        C_current = C.copy()
        for _ in range(n_steps):
            C_current = self.apply(C_current)
        return C_current
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de l'opérateur (pour sérialisation).
        
        Par défaut, retourne juste le type. Les sous-classes peuvent
        override pour inclure leurs paramètres spécifiques.
        
        Returns:
            Dictionnaire de paramètres
            
        Example:
            >>> class MyGamma(EvolutionOperator):
            ...     def __init__(self, alpha=0.1):
            ...         self.alpha = alpha
            ...     def get_parameters(self):
            ...         return {"type": "MyGamma", "alpha": self.alpha}
        """
        return {
            "type": self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        """Représentation textuelle pour debug."""
        params = self.get_parameters()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "type")
        return f"{params['type']}({param_str})"


class IdentityOperator(EvolutionOperator):
    """
    Opérateur identité: Γ(C) = C
    
    Utile pour:
    - Tests de référence
    - Vérification que le kernel fonctionne sans évolution
    - Placeholder dans les compositions
    """
    
    def apply(self, C: np.ndarray) -> np.ndarray:
        """Retourne C inchangé."""
        return C.copy()


class ScalingOperator(EvolutionOperator):
    """
    Opérateur de mise à l'échelle: Γ(C) = α·C + (1-α)·I
    
    Contracte progressivement C vers l'identité.
    Utile pour tests de convergence.
    
    Args:
        alpha: Facteur de scaling (0 ≤ α ≤ 1)
               α=1 → identité, α=0 → tous vers I
    """
    
    def __init__(self, alpha: float = 0.9):
        assert 0.0 <= alpha <= 1.0, "alpha doit être dans [0, 1]"
        self.alpha = alpha
    
    def apply(self, C: np.ndarray) -> np.ndarray:
        """
        Transforme: C' = α·C + (1-α)·I
        
        Contracte les corrélations vers zéro (sauf diagonale).
        """
        n = C.shape[0]
        I = np.eye(n)
        C_out = self.alpha * C + (1 - self.alpha) * I
        return C_out
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "ScalingOperator",
            "alpha": self.alpha
        }


class CompositeOperator(EvolutionOperator):
    """
    Composition de plusieurs opérateurs: Γ = Γ_n ∘ ... ∘ Γ_2 ∘ Γ_1
    
    Applique séquentiellement: C_{n+1} = Γ_n(...Γ_2(Γ_1(C_n))...)
    
    Args:
        operators: Liste d'opérateurs à appliquer dans l'ordre
    
    Example:
        >>> gamma = CompositeOperator([
        ...     DiffusionOperator(alpha=0.01),
        ...     HebbianOperator(beta=0.02)
        ... ])
        >>> # Applique d'abord diffusion, puis Hebbian
    """
    
    def __init__(self, operators: list[EvolutionOperator]):
        assert len(operators) > 0, "Au moins un opérateur requis"
        self.operators = operators
    
    def apply(self, C: np.ndarray) -> np.ndarray:
        """Applique séquentiellement tous les opérateurs."""
        C_current = C
        for op in self.operators:
            C_current = op.apply(C_current)
        return C_current
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "CompositeOperator",
            "operators": [op.get_parameters() for op in self.operators],
            "n_operators": len(self.operators)
        }


# ============================================================================
# UTILITAIRES POUR CONTRAINTES (helpers pour implémentations)
# ============================================================================

def enforce_symmetry(C: np.ndarray) -> np.ndarray:
    """
    Force la symétrie: C' = (C + C^T) / 2
    
    Utilitaire pour implémentations de Γ qui pourraient
    introduire de légères asymétries numériques.
    
    Args:
        C: Matrice possiblement asymétrique
    
    Returns:
        Matrice symétrisée
    """
    return (C + C.T) / 2


def enforce_diagonal(C: np.ndarray) -> np.ndarray:
    """
    Force la diagonale à 1.
    
    Args:
        C: Matrice avec diagonale possiblement ≠ 1
    
    Returns:
        Matrice avec diagonale = 1
    """
    C_out = C.copy()
    np.fill_diagonal(C_out, 1.0)
    return C_out


def enforce_bounds(C: np.ndarray, 
                  lower: float = -1.0, 
                  upper: float = 1.0) -> np.ndarray:
    """
    Clip les éléments dans [lower, upper].
    
    Args:
        C: Matrice avec éléments possiblement hors bornes
        lower: Borne inférieure
        upper: Borne supérieure
    
    Returns:
        Matrice clippée
    """
    return np.clip(C, lower, upper)


def enforce_correlation_invariants(C: np.ndarray) -> np.ndarray:
    """
    Applique tous les invariants de corrélation en une fois.
    
    Ordre d'application:
    1. Symétrie
    2. Bornes [-1, 1]
    3. Diagonale = 1
    
    Utilitaire pratique pour implémentations de Γ.
    
    Args:
        C: Matrice possiblement invalide
    
    Returns:
        Matrice valide (au mieux de l'approximation)
    """
    C_out = enforce_symmetry(C)
    C_out = enforce_bounds(C_out, -1.0, 1.0)
    C_out = enforce_diagonal(C_out)
    return C_out


# ============================================================================
# CLASSE DE BASE AVEC CONTRAINTES AUTOMATIQUES (optionnel)
# ============================================================================

class ConstrainedOperator(EvolutionOperator):
    """
    Classe de base qui applique automatiquement les contraintes.
    
    Les sous-classes implémentent _apply_unconstrained() qui peut
    produire des matrices invalides. Les contraintes sont appliquées
    automatiquement après.
    
    Utile pour simplifier l'implémentation de nouveaux Γ.
    
    Example:
        >>> class MyGamma(ConstrainedOperator):
        ...     def _apply_unconstrained(self, C):
        ...         # Peut retourner matrice asymétrique, hors bornes, etc.
        ...         return some_transformation(C)
        >>> # Les contraintes sont appliquées automatiquement
    """
    
    def apply(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la transformation puis force les contraintes.
        
        Override _apply_unconstrained() dans les sous-classes.
        """
        C_out = self._apply_unconstrained(C)
        return enforce_correlation_invariants(C_out)
    
    @abstractmethod
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """
        Implémente la transformation sans se soucier des contraintes.
        
        Les contraintes seront appliquées automatiquement après.
        
        Args:
            C: Matrice de corrélation valide
        
        Returns:
            Matrice transformée (peut être invalide)
        """
        pass