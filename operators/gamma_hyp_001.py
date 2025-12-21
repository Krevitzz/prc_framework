"""
operators/gamma_hyp_001.py

Implémentation de HYP-GAM-001 : Saturation pure comme mécanisme Γ.

HYPOTHÈSE TESTÉE :
Γ peut être défini comme une transformation non-linéaire de saturation
appliquée uniformément à C, sans mécanisme de renforcement additionnel.

TRANSFORMATION :
    C_{t+1} = tanh(β · C_t)

PRINCIPE :
- Préserve la structure (signe des corrélations)
- Limite la croissance (borne dans [-1, 1])
- Ne cherche pas à annuler D, juste à le réguler (compatible A3)
- Teste si la régulation seule suffit sans transitivité

AUCUNE PHYSIQUE :
- Pas de "température"
- Pas de "potentiel"
- Juste : fonction mathématique de saturation
"""

import numpy as np
import sys
from pathlib import Path

# Import du core PRC
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import ConstrainedOperator


class PureSaturationGamma(ConstrainedOperator):
    """
    Opérateur Γ de saturation pure (HYP-GAM-001).
    
    TRANSFORMATION :
        C'[i,j] = tanh(β * C[i,j])
    
    où β contrôle la vitesse de saturation.
    
    PROPRIÉTÉS :
    - Préserve le signe des corrélations
    - Converge asymptotiquement vers ±1
    - Aucun renforcement transitif
    - Aucune diffusion
    
    COMPORTEMENT ATTENDU :
    - Si β grand : saturation rapide vers {-1, 0, 1}
    - Si β petit : évolution lente, préserve nuances
    - Diagonale toujours préservée = 1
    
    TEST CRITIQUE :
    Cette hypothèse échoue si on observe :
    - Convergence triviale (toutes corrélations → 0 ou uniformes)
    - Perte de diversité structurelle
    - Annihilation informationnelle
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Initialise l'opérateur de saturation.
        
        Args:
            beta: Facteur de saturation
                  - beta > 5 : saturation très rapide (quelques itérations)
                  - beta ∈ [1, 5] : saturation modérée
                  - beta < 1 : évolution très lente
                  
        Contraintes :
            beta > 0 (strictement positif)
        """
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta
    
    def _apply_unconstrained(self, C: np.ndarray) -> np.ndarray:
        """
        Applique la saturation : C' = tanh(β·C)
        
        La diagonale est préservée = 1 après application des contraintes
        automatiques (via ConstrainedOperator).
        
        Args:
            C: Matrice de corrélation NxN
            
        Returns:
            C_next: Matrice saturée
        """
        # Saturation uniforme élément par élément
        C_next = np.tanh(self.beta * C)
        
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
            "type": "PureSaturationGamma",
            "hypothesis": "HYP-GAM-001",
            "beta": self.beta,
            "description": "Saturation pure sans renforcement transitif"
        }
    
    def __repr__(self) -> str:
        """Représentation textuelle."""
        return f"PureSaturationGamma(β={self.beta})"


# ============================================================================
# MÉTADONNÉES POUR TRAÇABILITÉ
# ============================================================================

HYPOTHESIS_ID = "HYP-GAM-001"
HYPOTHESIS_LEVEL = "L2"
HYPOTHESIS_STATUS = "WIP"

OPERATOR_METADATA = {
    "hypothesis_id": HYPOTHESIS_ID,
    "level": HYPOTHESIS_LEVEL,
    "status": HYPOTHESIS_STATUS,
    "transformation": "C_{t+1} = tanh(β · C_t)",
    "axioms_dependencies": ["A2", "A3"],
    "definitions_dependencies": ["DEF-GAMMA"],
    "expected_failure_modes": [
        "trivial_convergence",
        "structural_homogenization",
        "information_loss"
    ],
    "notes": [
        "Hypothèse minimale : régulation seule",
        "Pas de renforcement transitif",
        "Pas de diffusion",
        "Teste si saturation suffit à stabiliser D"
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
    gamma = PureSaturationGamma(beta=2.0)
    print(f"\n{gamma}")
    print(f"Paramètres : {gamma.get_parameters()}")
    
    # Test trivial de forme
    C_test = np.array([[1.0, 0.5], [0.5, 1.0]])
    C_out = gamma.apply(C_test)
    print(f"\nTest forme :")
    print(f"C_in :\n{C_test}")
    print(f"C_out :\n{C_out}")
    print(f"Diagonale préservée : {np.allclose(np.diag(C_out), 1.0)}")
    print(f"Symétrie préservée : {np.allclose(C_out, C_out.T)}")