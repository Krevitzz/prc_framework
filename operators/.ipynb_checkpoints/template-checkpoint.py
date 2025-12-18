"""
operators/template.py

TEMPLATE pour créer de nouveaux opérateurs Γ dans le framework PRC.

CE FICHIER EST UN GUIDE, PAS DU CODE À EXÉCUTER.

INSTRUCTIONS:
1. Copier ce template
2. Renommer la classe
3. Implémenter _apply_unconstrained()
4. Documenter la transformation mathématique
5. Tester sur plusieurs encodages

PRINCIPES À RESPECTER (CRITIQUE):
- Γ transforme AVEUGLÉMENT (pas d'optimisation, pas de but)
- Aucune physique (pas de "particules", "énergie", etc.)
- Action directe sur C (pas de graphe)
- Préservation des invariants (via ConstrainedOperator)
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EvolutionOperator, ConstrainedOperator


# ============================================================================
# TEMPLATE: OPÉRATEUR SIMPLE
# ============================================================================

class MyNewOperator(ConstrainedOperator):
    """
    [NOM DESCRIPTIF]: [description courte]
    
    TRANSFORMATION MATHÉMATIQUE:
        [Formule mathématique claire]
        
        Par exemple:
        C'[i,j] = C[i,j] + α * f(C)
        
        où f(C) = ...
    
    PRINCIPE:
        [Explication intuitive de ce que fait la transformation]
        
        Par exemple:
        - Si corrélations fortes → renforce
        - Si corrélations faibles → affaiblit
        - Etc.
    
    EFFET ATTENDU:
        [Quels patterns devraient émerger sous cette transformation]
        
        Par exemple:
        - Formation de clusters
        - Homogénéisation
        - Sparsification
        - Etc.
    
    AUCUNE PHYSIQUE:
        [Clarifier explicitement qu'aucune interprétation physique n'est faite]
        
        Par exemple:
        - Pas de "température"
        - Pas de "forces attractives"
        - Juste: transformation mathématique pure
    """
    
    def __init__(self, 
                 param1: float = 0.01,
                 param2: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialise l'opérateur avec ses paramètres.
        
        Args:
            param1: [Description du paramètre 1]
                    Typiquement: taux, amplitude, seuil
                    Contraintes: > 0, dans [0,1], etc.
            param2: [Description du paramètre 2]
                    Optionnel si valeur par défaut raisonnable
            seed: Graine aléatoire (si opérateur stochastique)
        
        Raises:
            AssertionError: Si contraintes sur paramètres violées
        """
        # VALIDATION DES PARAMÈTRES
        assert param1 > 0, "param1 doit être > 0"
        if param2 is not None:
            assert 0 <= param2 <= 1, "param2 doit être dans [0, 1]"
        
        # STOCKAGE
        self.param1 = param1
        self.param2 = param2 if param2 is not None else 0.5  # Défaut
        
        # GÉNÉRATEUR ALÉATOIRE (si nécessaire)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = None
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la transformation mathématique sur C.
        
        CETTE MÉTHODE EST LE CŒUR DE L'OPÉRATEUR.
        
        RESPONSABILITÉS:
        1. Implémenter la règle de transformation
        2. NE PAS modifier C en place (créer C_next)
        3. NE PAS s'inquiéter des contraintes (gérées automatiquement)
        
        ÉTAPES TYPIQUES:
        1. Extraire des métriques de C (variance, moyenne, etc.)
        2. Calculer termes intermédiaires (produits matriciels, etc.)
        3. Combiner pour former C_next
        4. Retourner C_next
        
        Args:
            C: Matrice de corrélation NxN à l'instant n
               Garanties: symétrique, diag=1, éléments dans [-1,1]
        
        Returns:
            C_next: Matrice transformée
                    Peut être asymétrique, hors bornes, etc.
                    Les contraintes seront appliquées automatiquement
        """
        n = C.shape[0]
        
        # ================================================================
        # VOTRE TRANSFORMATION ICI
        # ================================================================
        
        # Exemple 1: Transformation linéaire simple
        # C_next = self.param1 * C + (1 - self.param1) * np.eye(n)
        
        # Exemple 2: Transformation non-linéaire
        # C_next = np.tanh(self.param1 * C)
        
        # Exemple 3: Avec produit matriciel
        # C_squared = C @ C
        # C_next = C + self.param1 * (C_squared - n * C)
        
        # Exemple 4: Avec bruit (si stochastique)
        # if self.rng is not None:
        #     noise = self.rng.randn(n, n)
        #     noise = (noise + noise.T) / 2  # Symétrise
        #     np.fill_diagonal(noise, 0)
        #     C_next = C + self.param1 * noise
        
        # PLACEHOLDER (remplacer par vraie transformation)
        C_next = C.copy()
        
        # ================================================================
        
        return C_next
    
    def get_parameters(self) -> dict:
        """
        Retourne les paramètres de l'opérateur pour sérialisation.
        
        UTILISATION:
        - Sauvegarder l'opérateur en JSON
        - Logger les expériences
        - Comparer différentes configurations
        
        Returns:
            Dictionnaire avec tous les paramètres
        """
        return {
            "type": "MyNewOperator",  # Changer ce nom
            "param1": self.param1,
            "param2": self.param2,
            "seed": self.seed,
            # Ajouter tous les paramètres pertinents
        }


# ============================================================================
# TEMPLATE: OPÉRATEUR COMPLEXE (avec état interne)
# ============================================================================

class MyStatefulOperator(ConstrainedOperator):
    """
    Opérateur avec état interne qui évolue au fil des itérations.
    
    ATTENTION: Éviter les états internes si possible (rend Γ non-markovien).
    Utiliser seulement si absolument nécessaire.
    
    EXEMPLE D'USAGE:
    - Moyennes mobiles de métriques
    - Adaptation temporelle des paramètres
    - Mémoire limitée des configurations précédentes
    """
    
    def __init__(self, param: float = 0.01):
        self.param = param
        
        # ÉTAT INTERNE
        self._iteration_count = 0
        self._running_average = None  # Sera initialisé au premier appel
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """Applique transformation avec état interne."""
        n = C.shape[0]
        
        # Initialise état si première itération
        if self._running_average is None:
            self._running_average = np.mean(C)
        
        # Met à jour état interne
        current_mean = np.mean(C)
        alpha_ema = 0.1  # Taux de moyenne mobile
        self._running_average = (alpha_ema * current_mean + 
                                 (1 - alpha_ema) * self._running_average)
        
        # Utilise l'état dans la transformation
        C_centered = C - self._running_average
        C_next = C + self.param * C_centered
        
        # Incrémente compteur
        self._iteration_count += 1
        
        return C_next
    
    def reset_state(self):
        """Réinitialise l'état interne (utile pour tests)."""
        self._iteration_count = 0
        self._running_average = None
    
    def get_parameters(self) -> dict:
        return {
            "type": "MyStatefulOperator",
            "param": self.param,
            "iteration_count": self._iteration_count,
            "running_average": self._running_average
        }


# ============================================================================
# CHECKLIST POUR NOUVEL OPÉRATEUR
# ============================================================================

"""
AVANT DE SOUMETTRE UN NOUVEL OPÉRATEUR, VÉRIFIER:

□ La classe hérite de ConstrainedOperator
□ _apply_unconstrained() est implémentée
□ get_parameters() retourne tous les paramètres
□ Docstring explique la transformation mathématiquement
□ AUCUN concept physique dans le code
□ Validation des paramètres dans __init__()
□ Pas de modification de C en place
□ Graine aléatoire si stochastique
□ Testé sur au moins 3 encodages différents
□ Les invariants sont préservés (automatique si ConstrainedOperator)

TESTS MINIMAUX À FAIRE:

1. Test de symétrie:
   >>> C = np.random.randn(10, 10)
   >>> C = (C + C.T) / 2
   >>> np.fill_diagonal(C, 1.0)
   >>> op = MyNewOperator()
   >>> C_next = op.apply(C)
   >>> assert np.allclose(C_next, C_next.T)

2. Test de diagonale:
   >>> assert np.allclose(np.diag(C_next), 1.0)

3. Test de bornes:
   >>> assert np.all(C_next >= -1.01) and np.all(C_next <= 1.01)

4. Test d'universalité (même Γ sur différents encodages):
   >>> from core import InformationSpace, PRCKernel
   >>> from core import create_block_diagonal, create_exponential_decay
   >>> 
   >>> encodings = [
   ...     InformationSpace.random(20, seed=42),
   ...     create_block_diagonal([5, 5, 10], 0.7, 0.1),
   ...     create_exponential_decay(20, 2.0)
   ... ]
   >>> 
   >>> for enc in encodings:
   ...     kernel = PRCKernel(enc, MyNewOperator())
   ...     C_final = kernel.step(50)
   ...     # Vérifier que ça marche sans erreur

5. Test de reproductibilité (si stochastique):
   >>> op1 = MyNewOperator(seed=42)
   >>> op2 = MyNewOperator(seed=42)
   >>> C_test = InformationSpace.random(15, seed=123).C
   >>> C1 = op1.apply(C_test)
   >>> C2 = op2.apply(C_test)
   >>> assert np.allclose(C1, C2)
"""


# ============================================================================
# EXEMPLES D'IMPLÉMENTATIONS COMPLÈTES
# ============================================================================

class ExampleSimpleOperator(ConstrainedOperator):
    """
    Exemple simple: décroissance vers identité.
    
    TRANSFORMATION:
        C' = (1-α) * C + α * I
    
    EFFET: Tous les DOF deviennent progressivement indépendants.
    """
    
    def __init__(self, alpha: float = 0.01):
        assert 0 < alpha < 1, "alpha dans (0, 1)"
        self.alpha = alpha
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        return (1 - self.alpha) * C + self.alpha * np.eye(n)
    
    def get_parameters(self) -> dict:
        return {"type": "ExampleSimple", "alpha": self.alpha}


class ExampleComplexOperator(ConstrainedOperator):
    """
    Exemple complexe: diffusion + Hebbian + bruit.
    
    TRANSFORMATION:
        C' = C + α*(C²-nC) + β*C² + σ*ε
    """
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, 
                 sigma: float = 0.005, seed: Optional[int] = None):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed else None
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        C_squared = C @ C
        
        # Diffusion
        diff_term = self.alpha * (C_squared - n * C)
        
        # Hebbian
        hebb_term = self.beta * C_squared
        
        # Bruit
        if self.rng is not None:
            noise = self.rng.randn(n, n)
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0)
            noise_term = self.sigma * noise
        else:
            noise_term = 0
        
        return C + diff_term + hebb_term + noise_term
    
    def get_parameters(self) -> dict:
        return {
            "type": "ExampleComplex",
            "alpha": self.alpha,
            "beta": self.beta,
            "sigma": self.sigma,
            "seed": self.seed
        }