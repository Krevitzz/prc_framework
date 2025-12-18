"""
core/information_space.py

Représentation pure de l'espace d'information D dans le framework PRC.

PRINCIPES FONDAMENTAUX (Thèse Ch. 2.2):
- UNE SEULE primitive: la matrice de corrélation C
- Aucune structure topologique présupposée (pas de graphe dans l'état)
- Aucune présupposition physique (pas de "particules", "forces", etc.)
- Validation mathématique pure uniquement

INVARIANTS:
- C est une matrice carrée NxN
- C est symétrique: C[i,j] = C[j,i]
- Diagonale = 1 (auto-corrélation normalisée)
- Éléments dans [-1, 1] (corrélations normalisées)
"""

import numpy as np
from typing import Dict, Any, Optional


class InformationSpace:
    """
    Représentation fondamentale de l'espace d'information D.
    
    État = uniquement la matrice de corrélation C.
    Pas de vue dérivée (graphe, etc.) stockée dans l'objet.
    """
    
    def __init__(self, correlation_matrix: np.ndarray, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialise un espace d'information avec sa matrice de corrélation.
        
        Args:
            correlation_matrix: Matrice NxN de corrélations entre DOF
            metadata: Métadonnées optionnelles (domaine, labels, etc.)
        
        Raises:
            AssertionError: Si les invariants mathématiques sont violés
        """
        # VALIDATION MATHÉMATIQUE STRICTE
        self._validate_correlation_matrix(correlation_matrix)
        
        # ÉTAT FONDAMENTAL: uniquement C
        self.C = correlation_matrix.copy()  # Copy pour isolation
        
        # PROPRIÉTÉS DÉRIVÉES IMMÉDIATES
        self.n_dof = self.C.shape[0]
        
        # MÉTADONNÉES (pour traçabilité, pas pour logique)
        self.metadata = metadata if metadata is not None else {}
    
    @staticmethod
    def _validate_correlation_matrix(C: np.ndarray) -> None:
        """
        Valide les invariants mathématiques de la matrice de corrélation.
        
        Vérifie:
        1. C est une matrice 2D
        2. C est carrée (NxN)
        3. C est symétrique
        4. Diagonale = 1 (auto-corrélation)
        5. Éléments dans [-1, 1]
        
        Args:
            C: Matrice à valider
            
        Raises:
            AssertionError: Si un invariant est violé (avec message explicite)
        """
        # 1. Dimensionnalité
        assert C.ndim == 2, \
            f"C doit être une matrice 2D, reçu shape {C.shape}"
        
        # 2. Carré
        n, m = C.shape
        assert n == m, \
            f"C doit être carrée (NxN), reçu shape ({n}, {m})"
        
        # 3. Symétrie (avec tolérance numérique)
        assert np.allclose(C, C.T, rtol=1e-5, atol=1e-8), \
            f"C doit être symétrique, max diff = {np.max(np.abs(C - C.T))}"
        
        # 4. Diagonale = 1
        diag = np.diag(C)
        assert np.allclose(diag, 1.0, rtol=1e-5, atol=1e-8), \
            f"Diagonale de C doit être 1, reçu range [{diag.min():.3f}, {diag.max():.3f}]"
        
        # 5. Éléments dans [-1, 1]
        assert np.all(C >= -1.0 - 1e-8) and np.all(C <= 1.0 + 1e-8), \
            f"Éléments de C doivent être dans [-1, 1], reçu range [{C.min():.3f}, {C.max():.3f}]"
    
    def get_correlation(self, i: int, j: int) -> float:
        """
        Accès direct à un élément de corrélation.
        
        Args:
            i, j: Indices des DOF (0 ≤ i,j < n_dof)
            
        Returns:
            Corrélation C[i,j]
        """
        return float(self.C[i, j])
    
    def get_subspace(self, indices: np.ndarray) -> 'InformationSpace':
        """
        Extrait un sous-espace (restriction à certains DOF).
        
        Opération mathématique pure: extraction de sous-matrice.
        Aucune sémantique physique présupposée.
        
        Args:
            indices: Indices des DOF à garder
            
        Returns:
            Nouvel InformationSpace avec sous-matrice C[indices, indices]
        """
        indices = np.asarray(indices)
        assert np.all(indices >= 0) and np.all(indices < self.n_dof), \
            "Indices hors limites"
        
        # Extraction de sous-matrice
        C_sub = self.C[np.ix_(indices, indices)]
        
        # Métadonnées du sous-espace
        sub_metadata = {
            "parent_n_dof": self.n_dof,
            "subspace_indices": indices.tolist(),
            **self.metadata
        }
        
        return InformationSpace(C_sub, sub_metadata)
    
    def copy(self) -> 'InformationSpace':
        """
        Crée une copie indépendante de l'espace d'information.
        
        Returns:
            Nouvelle instance avec C copié et métadonnées copiées
        """
        return InformationSpace(
            self.C.copy(),
            self.metadata.copy() if self.metadata else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialise l'état en dictionnaire (pour JSON).
        
        Format minimal: uniquement ce qui est nécessaire pour reconstruction.
        
        Returns:
            {"C": list, "n_dof": int, "metadata": dict}
        """
        return {
            "C": self.C.tolist(),
            "n_dof": self.n_dof,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InformationSpace':
        """
        Désérialise depuis un dictionnaire.
        
        Args:
            data: Dictionnaire au format to_dict()
            
        Returns:
            Nouvelle instance InformationSpace
        """
        C = np.array(data["C"])
        metadata = data.get("metadata", {})
        return cls(C, metadata)
    
    @classmethod
    def identity(cls, n_dof: int, 
                 metadata: Optional[Dict[str, Any]] = None) -> 'InformationSpace':
        """
        Crée un espace avec matrice identité (corrélations nulles).
        
        État initial typique: chaque DOF est indépendant.
        
        Args:
            n_dof: Nombre de degrés de liberté
            metadata: Métadonnées optionnelles
            
        Returns:
            InformationSpace avec C = I_n
        """
        C = np.eye(n_dof)
        return cls(C, metadata)
    
    @classmethod
    def uniform(cls, n_dof: int, correlation: float = 0.5,
                metadata: Optional[Dict[str, Any]] = None) -> 'InformationSpace':
        """
        Crée un espace avec corrélations uniformes.
        
        Tous les DOF également corrélés (hors diagonale).
        
        Args:
            n_dof: Nombre de degrés de liberté
            correlation: Valeur de corrélation uniforme (dans [-1, 1])
            metadata: Métadonnées optionnelles
            
        Returns:
            InformationSpace avec C[i,j] = correlation si i≠j, 1 si i=j
        """
        assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"
        
        C = np.full((n_dof, n_dof), correlation)
        np.fill_diagonal(C, 1.0)
        
        return cls(C, metadata)
    
    @classmethod
    def random(cls, n_dof: int, 
               mean: float = 0.0,
               std: float = 0.3,
               seed: Optional[int] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'InformationSpace':
        """
        Crée un espace avec corrélations aléatoires.
        
        Utile pour tests ou initialisation exploratoire.
        Garantit la symétrie et diagonale = 1.
        
        Args:
            n_dof: Nombre de degrés de liberté
            mean: Moyenne des corrélations
            std: Écart-type des corrélations
            seed: Graine aléatoire (pour reproductibilité)
            metadata: Métadonnées optionnelles
            
        Returns:
            InformationSpace avec corrélations aléatoires valides
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Génère matrice aléatoire
        C = np.random.normal(mean, std, (n_dof, n_dof))
        
        # Rend symétrique
        C = (C + C.T) / 2
        
        # Fixe diagonale
        np.fill_diagonal(C, 1.0)
        
        # Clip dans [-1, 1]
        C = np.clip(C, -1.0, 1.0)
        
        return cls(C, metadata)
    
    def __repr__(self) -> str:
        """Représentation textuelle pour debug."""
        domain = self.metadata.get("domain", "unknown")
        return f"InformationSpace(n_dof={self.n_dof}, domain='{domain}')"
    
    def __eq__(self, other: 'InformationSpace') -> bool:
        """
        Égalité structurelle (pour tests).
        
        Deux InformationSpace sont égaux si leurs matrices C sont égales
        (aux erreurs numériques près).
        """
        if not isinstance(other, InformationSpace):
            return False
        
        return (self.n_dof == other.n_dof and 
                np.allclose(self.C, other.C, rtol=1e-5, atol=1e-8))


# ============================================================================
# UTILITAIRES DE CONSTRUCTION (patterns mathématiques communs)
# ============================================================================

def create_block_diagonal(block_sizes: list[int],
                         intra_correlation: float = 0.7,
                         inter_correlation: float = 0.1,
                         metadata: Optional[Dict[str, Any]] = None) -> InformationSpace:
    """
    Crée une structure par blocs (corrélations fortes intra, faibles inter).
    
    Pattern mathématique utile pour représenter des groupes/clusters.
    Aucune sémantique physique présupposée (pas de "particul", juste des blocs).
    
    Args:
        block_sizes: Liste des tailles de chaque bloc [n1, n2, ...]
        intra_correlation: Corrélation à l'intérieur des blocs
        inter_correlation: Corrélation entre les blocs
        metadata: Métadonnées optionnelles
        
    Returns:
        InformationSpace avec structure par blocs
        
    Example:
        >>> space = create_block_diagonal([5, 5, 10], 0.8, 0.1)
        >>> # 3 blocs: 5 DOF fortement corrélés, 5 DOF fortement corrélés, 
        >>> #           10 DOF fortement corrélés, faiblement corrélés entre blocs
    """
    n_dof = sum(block_sizes)
    C = np.full((n_dof, n_dof), inter_correlation)
    
    # Remplit les blocs diagonaux
    idx = 0
    for size in block_sizes:
        block_slice = slice(idx, idx + size)
        C[block_slice, block_slice] = intra_correlation
        idx += size
    
    # Diagonale = 1
    np.fill_diagonal(C, 1.0)
    
    return InformationSpace(C, metadata)


def create_exponential_decay(n_dof: int,
                            decay_rate: float = 1.0,
                            metadata: Optional[Dict[str, Any]] = None) -> InformationSpace:
    """
    Crée des corrélations avec décroissance exponentielle.
    
    C[i,j] = exp(-|i-j| / decay_rate)
    
    Pattern mathématique représentant des corrélations décroissantes avec la "distance"
    (ici, distance = différence d'indice, aucune géométrie présupposée).
    
    Args:
        n_dof: Nombre de degrés de liberté
        decay_rate: Taux de décroissance (plus grand = décroissance plus lente)
        metadata: Métadonnées optionnelles
        
    Returns:
        InformationSpace avec corrélations exponentielles
    """
    indices = np.arange(n_dof)
    distance_matrix = np.abs(indices[:, None] - indices[None, :])
    C = np.exp(-distance_matrix / decay_rate)
    
    return InformationSpace(C, metadata)


def create_power_law(n_dof: int,
                    exponent: float = -2.0,
                    cutoff: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> InformationSpace:
    """
    Crée des corrélations en loi de puissance.
    
    C[i,j] = (|i-j| + cutoff)^exponent
    
    Pattern mathématique pour corrélations longue portée.
    
    Args:
        n_dof: Nombre de degrés de liberté
        exponent: Exposant de la loi de puissance (négatif pour décroissance)
        cutoff: Coupure pour éviter divergence en 0
        metadata: Métadonnées optionnelles
        
    Returns:
        InformationSpace avec corrélations en loi de puissance
    """
    indices = np.arange(n_dof)
    distance_matrix = np.abs(indices[:, None] - indices[None, :])
    C = (distance_matrix + cutoff) ** exponent
    
    # Normalise pour que max = 1 (hors diagonale)
    C = C / np.max(C)
    np.fill_diagonal(C, 1.0)
    
    # Clip dans [-1, 1]
    C = np.clip(C, -1.0, 1.0)
    
    return InformationSpace(C, metadata)