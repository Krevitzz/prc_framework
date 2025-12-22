"""
operators/gamma_hyp_002.py

Implémentation de HYP-GAM-002 : Diffusion pure comme mécanisme Γ.

HYPOTHÈSE TESTÉE :
Γ peut être défini comme un opérateur de diffusion pur agissant sur C,
propageant les corrélations via transitivité mathématique.

TRANSFORMATION :
    C_{t+1} = C_t + α·(C_t² - n·C_t)

PRINCIPE :
- Propage les corrélations via chemins de longueur 2
- Homogénéise localement (diffusion classique)
- Pas de renforcement directionnel
- Compatible A3 (ne cherche pas à annuler D)

AUCUNE PHYSIQUE :
- Pas de "chaleur qui se propage"
- Pas de "particules qui diffusent"
- Juste : moyennage pondéré de corrélations transitives
"""

import numpy as np
import sys
from pathlib import Path

# Import du core PRC
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import ConstrainedOperator


class PureDiffusionGamma(ConstrainedOperator):
    """
    Opérateur Γ de diffusion pure (HYP-GAM-002).
    
    TRANSFORMATION :
        C_{t+1} = C_t + α·(C_t @ C_t - n·C_t)
    
    où :
    - C_t @ C_t = influence via chemins transitifs (longueur 2)
    - n = normalisation par nombre de DOF
    - α = taux de diffusion
    
    PROPRIÉTÉS :
    - Propage les corrélations existantes
    - Homogénéise progressivement
    - Terme de dissipation : -n·C_t (empêche divergence naïve)
    - Préserve symétrie par construction
    
    COMPORTEMENT ATTENDU :
    - Si α grand : homogénéisation rapide (toutes corrélations → moyenne)
    - Si α petit : évolution lente, structure peut persister temporairement
    - Diagonale toujours préservée = 1
    
    TEST CRITIQUE :
    Cette hypothèse échoue si on observe :
    - Convergence triviale (toutes corrélations → valeur uniforme)
    - Perte totale de diversité structurelle
    - Annihilation informationnelle par moyennage
    """
    
    def __init__(self, alpha: float = 0.01, normalize: bool = True):
        """
        Initialise l'opérateur de diffusion.
        
        Args:
            alpha: Taux de diffusion
                   - alpha > 0.1 : diffusion très rapide (quelques itérations)
                   - alpha ∈ [0.01, 0.1] : diffusion modérée
                   - alpha < 0.01 : évolution très lente
            normalize: Si True, normalise par n (recommandé pour stabilité)
                   
        Contraintes :
            alpha > 0 (strictement positif)
        """
        assert alpha > 0, "alpha doit être strictement positif"
        self.alpha = alpha
        self.normalize = normalize
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la diffusion : C' = C + α·(C² - n·C)
        
        La diagonale est préservée = 1 après application des contraintes
        automatiques (via ConstrainedOperator).
        
        Args:
            C: Matrice de corrélation NxN
            
        Returns:
            C_next: Matrice après diffusion
        """
        n = C.shape[0]
        
        # Influence via chemins transitifs (longueur 2)
        indirect_influence = C @ C
        
        # Terme de diffusion
        if self.normalize:
            # Avec normalisation : (C² - n·C) → diffusion sans explosion
            diffusion_term = self.alpha * (indirect_influence - n * C)
        else:
            # Sans normalisation : juste C² (peut diverger)
            diffusion_term = self.alpha * indirect_influence
        
        # Mise à jour
        C_next = C + diffusion_term
        
        # Note : La diagonale sera automatiquement rétablie à 1
        # par enforce_correlation_invariants() de ConstrainedOperator
        
        return C_next
    
    def get_parameters(self) -> dict:
        """
        Retourne les paramètres pour sérialisation.
        
        Returns:
            Dictionnaire avec type et paramètres
        """
        return {
            "type": "PureDiffusionGamma",
            "hypothesis": "HYP-GAM-002",
            "alpha": self.alpha,
            "normalize": self.normalize,
            "description": "Diffusion pure via transitivité mathématique"
        }
    
    def __repr__(self) -> str:
        """Représentation textuelle."""
        norm_str = "normalized" if self.normalize else "unnormalized"
        return f"PureDiffusionGamma(α={self.alpha}, {norm_str})"


# ============================================================================
# MÉTADONNÉES POUR TRAÇABILITÉ
# ============================================================================

HYPOTHESIS_ID = "HYP-GAM-002"
HYPOTHESIS_LEVEL = "L2"
HYPOTHESIS_STATUS = "WIP[R0-open]"

OPERATOR_METADATA = {
    "hypothesis_id": HYPOTHESIS_ID,
    "level": HYPOTHESIS_LEVEL,
    "status": HYPOTHESIS_STATUS,
    "transformation": "C_{t+1} = C_t + α·(C_t² - n·C_t)",
    "axioms_dependencies": ["A2", "A3"],
    "definitions_dependencies": ["DEF-GAMMA"],
    "expected_failure_modes": [
        "trivial_convergence",
        "homogenization",
        "information_loss",
        "uniform_correlations"
    ],
    "notes": [
        "Hypothèse fondamentale : propagation pure",
        "Pas de renforcement directionnel",
        "Pas de régulation",
        "Teste si diffusion seule peut maintenir structure"
    ]
}


def get_operator_metadata() -> dict:
    """
    Retourne les métadonnées de traçabilité.
    
    Utilisé par les tests pour documenter automatiquement
    les résultats dans le format PRC.
    """
    return OPERATOR_METADATA.copy()


# ============================================================================
# POINT D'ENTRÉE POUR TESTS
# ============================================================================

if __name__ == "__main__":
    # Démonstration basique (pas un test complet)
    print(f"Opérateur : {HYPOTHESIS_ID}")
    print(f"Status : {HYPOTHESIS_STATUS}")
    print(f"Transformation : {OPERATOR_METADATA['transformation']}")
    
    # Instanciation exemple
    gamma = PureDiffusionGamma(alpha=0.02)
    print(f"\n{gamma}")
    print(f"Paramètres : {gamma.get_parameters()}")
    
    # Test trivial de forme
    C_test = np.array([[1.0, 0.5, 0.3], 
                       [0.5, 1.0, 0.4],
                       [0.3, 0.4, 1.0]])
    C_out = gamma.apply(C_test)
    print(f"\nTest forme :")
    print(f"C_in :\n{C_test}")
    print(f"C_out :\n{C_out}")
    print(f"Diagonale préservée : {np.allclose(np.diag(C_out), 1.0)}")
    print(f"Symétrie préservée : {np.allclose(C_out, C_out.T)}")
    
    # Vérifie effet de diffusion
    C_squared = C_test @ C_test
    print(f"\nEffet diffusion :")
    print(f"C² :\n{C_squared}")
    print(f"Terme diffusion (non normalisé) :\n{0.02 * C_squared}")