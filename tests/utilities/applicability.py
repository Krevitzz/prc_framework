"""
tests/utilities/applicability.py

Matrice d'applicabilité technique des tests (Section 14.4).

Ce module détermine quels tests peuvent être appliqués selon :
  - Type de D (SYM, ASY, R3)
  - Forme du tenseur (rang, dimensions)
  - Type de Γ (pointwise, coupled, etc.)

RÈGLE CRITIQUE :
  Avant d'appliquer un test, TOUJOURS vérifier applicabilité.
  Un test non applicable stocke applicable=False dans TestObservations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# MATRICE D'APPLICABILITÉ TECHNIQUE
# =============================================================================

TEST_APPLICABILITY = {
    # =========================================================================
    # TESTS UNIVERSELS - Toujours applicables
    # =========================================================================
    
    "UNIV-001": {
        "name": "Évolution norme Frobenius",
        "requires_rank": None,           # Tous rangs
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure ||D||_F à chaque itération",
    },
    
    "UNIV-002": {
        "name": "Évolution diversité (std)",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure σ(D_flat) à chaque itération",
    },
    
    "UNIV-003": {
        "name": "Convergence vers point fixe",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure ||D_{t+1} - D_t||",
    },
    
    "UNIV-004": {
        "name": "Sensibilité conditions initiales",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Compare trajectoires (multi-seeds)",
        "note": "Nécessite plusieurs runs avec seeds différents",
    },
    
    # =========================================================================
    # TESTS SYMÉTRIE - Rang 2 carrées uniquement
    # =========================================================================
    
    "SYM-001": {
        "name": "Préservation symétrie",
        "requires_rank": 2,
        "requires_square": False,
        "d_types": ["SYM", "ASY"],      # Pas R3
        "gamma_types": ["any"],
        "description": "Teste si Γ préserve symétrie d'un D symétrique",
    },
    
    "SYM-002": {
        "name": "Création symétrie",
        "requires_rank": 2,
        "requires_square": False,
        "d_types": ["SYM", "ASY"],
        "gamma_types": ["any"],
        "description": "Teste si Γ crée symétrie sur D asymétrique",
    },
    
    "SYM-003": {
        "name": "Évolution asymétrie",
        "requires_rank": 2,
        "requires_square": False,
        "d_types": ["SYM", "ASY"],
        "gamma_types": ["any"],
        "description": "Observe ratio asymétrie/symétrie",
    },
    
    # =========================================================================
    # TESTS STRUCTURE - Spécifiques au rang
    # =========================================================================
    
    "STR-001": {
        "name": "Préservation rang",
        "requires_rank": 2,
        "requires_square": False,       # Applicable aux non-carrées aussi
        "d_types": ["SYM", "ASY"],
        "gamma_types": ["any"],
        "description": "Vérifie rank(D_t) via SVD",
    },
    
    "STR-002": {
        "name": "Évolution spectre",
        "requires_rank": 2,
        "requires_square": False,
        "d_types": ["SYM", "ASY"],             # Symétriques uniquement (eigvalsh)
        "gamma_types": ["any"],
        "description": "Mesure valeurs propres λ_i",
    },
    
    "STR-003": {
        "name": "Corrélations temporelles",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure corr(D_flat(t), D_flat(t+k))",
    },
    
    # =========================================================================
    # TESTS BORNES/CONTRAINTES
    # =========================================================================
    
    "BND-001": {
        "name": "Respect bornes initiales",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Vérifie min/max(D_t) vs bornes initiales",
    },
    
    "BND-002": {
        "name": "Préservation positivité",
        "requires_rank": 2,
        "requires_square": False,
        "d_types": ["SYM", "ASY"],             
        "gamma_types": ["any"],
        "description": "Vérifie min(eig(D_t)) > 0",
        "note": "Applicable si D initial défini positif (SYM-004)",
    },
    
    # =========================================================================
    # TESTS LOCALITÉ
    # =========================================================================
    
    "LOC-001": {
        "name": "Propagation information",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Perturbe D[i0,j0], mesure rayon propagation",
        "note": "Test coûteux - exécuter sur sous-échantillon",
    },
    
    "LOC-002": {
        "name": "Préservation sparsité",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure % éléments |D_ij| < ε",
        "note": "Applicable si D initial sparse (M3, SYM-005)",
    },
    
    # =========================================================================
    # TESTS DIVERSITÉ COMPLÉMENTAIRES
    # =========================================================================
    
    "DIV-ENTROPY": {
        "name": "Évolution entropie",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Entropie Shannon histogramme valeurs",
    },
    
    "DIV-UNIFORM": {
        "name": "Test uniformité",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Coefficient variation CV = std / |mean|",
    },
    
    "DIV-RANGE": {
        "name": "Évolution plage",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure max(D) - min(D)",
    },
    
    "DIV-DISTINCT": {
        "name": "Valeurs distinctes",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Compte valeurs distinctes avec tolérance",
    },
	# Nouveaux tests diversité
    "UNIV-002b": {
        "requires_rank": 2,
        "requires_square": False,  # Peut être rectangulaire
        "requires_min_size": 5,    # Pour patches 5×5
        "d_types": ["SYM", "ASY"],
        "description": "Diversité locale via patches",
    },
    
    "DIV-HETERO": {
        "requires_rank": 2,
        "requires_square": False,
        "requires_min_size": 10,   # Pour grille 10×10
        "d_types": ["SYM", "ASY"],
        "description": "Hétérogénéité spatiale",
    },
    
    # =========================================================================
    # TESTS CONVERGENCE COMPLÉMENTAIRES
    # =========================================================================
    
    "CONV-LYAPUNOV": {
        "name": "Exposant Lyapunov",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Estime λ = lim log(||δD_t||/||δD_0||) / t",
    },
    
    "CONV-SPEED": {
        "name": "Vitesse convergence",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Mesure nombre iterations avant ||D_t - D_{t-1}|| < ε",
    },
    
    "CONV-OSCILLATION": {
        "name": "Détection oscillations",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["any"],
        "description": "Détecte cycles périodiques",
    },
    
    # =========================================================================
    # TESTS POINTWISE (Γ spécifique)
    # =========================================================================
    
    "PW-001": {
        "name": "Indépendance pointwise",
        "requires_rank": None,
        "requires_square": False,
        "d_types": ["SYM", "ASY", "R3"],
        "gamma_types": ["pointwise"],    # Seulement Γ pointwise
        "description": "Vérifie absence couplage inter-éléments",
    },
}


# =============================================================================
# FONCTIONS DE VÉRIFICATION
# =============================================================================

def is_test_applicable(test_name: str, d_base_id: str, 
                       state_shape: tuple, gamma_id: str = None) -> bool:
    """
    Vérifie si un test est applicable au contexte donné.
    
    Args:
        test_name: Nom du test (ex: "UNIV-002b")
        d_base_id: ID de la base D (ex: "SYM-001")
        state_shape: Shape de l'état (ex: (50, 50))
        gamma_id: ID du Γ (optionnel, pour tests spécifiques Γ)
    
    Returns:
        True si test applicable, False sinon
    """
    if test_name not in TEST_APPLICABILITY:
        return False
    
    spec = TEST_APPLICABILITY[test_name]
    
    # Vérifier rang
    if spec.get("requires_rank") is not None:
        if len(state_shape) != spec["requires_rank"]:
            return False
    
    # Vérifier carré
    if spec.get("requires_square", False):
        if len(state_shape) == 2 and state_shape[0] != state_shape[1]:
            return False
    
    # Vérifier taille minimale (NOUVEAU)
    if spec.get("requires_min_size") is not None:
        min_size = spec["requires_min_size"]
        if any(dim < min_size for dim in state_shape):
            return False
    
    # Vérifier type D
    d_types = spec.get("d_types", [])
    if d_types:
        if not any(d_base_id.startswith(dt) for dt in d_types):
            return False
    
    # Vérifier type Γ (si spécifié)
    gamma_types = spec.get("gamma_types")
    if gamma_types and gamma_id:
        if gamma_id not in gamma_types:
            return False
    
    return True


def get_applicable_tests(d_base_id: str, 
                        state_shape: Tuple[int, ...],
                        gamma_id: str = None) -> List[str]:
    """
    Retourne la liste de tous les tests applicables.
    
    Args:
        d_base_id: ID de la base D
        state_shape: Shape du tenseur
        gamma_id: ID du Γ (optionnel)
    
    Returns:
        Liste des test_name applicables
    
    Exemple:
        >>> get_applicable_tests("SYM-001", (50, 50))
        ['UNIV-001', 'UNIV-002', 'UNIV-003', 'SYM-001', 'SYM-002', ...]
    """
    applicable = []
    
    for test_name in TEST_APPLICABILITY.keys():
        if is_test_applicable(test_name, d_base_id, state_shape, gamma_id):
            applicable.append(test_name)
    
    return sorted(applicable)


def print_applicability_matrix():
    """Affiche la matrice d'applicabilité complète."""
    print("\n" + "="*80)
    print("MATRICE D'APPLICABILITÉ DES TESTS")
    print("="*80)
    
    # Grouper par catégorie
    categories = {}
    for test_name, spec in TEST_APPLICABILITY.items():
        category = test_name.split('-')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append((test_name, spec))
    
    # Afficher par catégorie
    for category in sorted(categories.keys()):
        print(f"\n{category}")
        print("─" * 80)
        
        for test_name, spec in categories[category]:
            d_types = ", ".join(spec["d_types"])
            rank = str(spec["requires_rank"]) if spec["requires_rank"] else "any"
            square = "yes" if spec["requires_square"] else "no"
            
            print(f"  {test_name:<15} {spec['name']:<35} "
                  f"rank={rank:<3} square={square:<3} D=[{d_types}]")
    
    print("\n" + "="*80 + "\n")


def get_test_description(test_name: str) -> Optional[str]:
    """Retourne la description d'un test."""
    if test_name in TEST_APPLICABILITY:
        return TEST_APPLICABILITY[test_name]["description"]
    return None


def validate_applicability_matrix():
    """
    Valide la cohérence de la matrice d'applicabilité.
    
    Vérifie :
      - Tous les tests ont les champs requis
      - Les types D sont valides
      - Les rangs sont cohérents
    """
    required_fields = ["name", "requires_rank", "requires_square", "d_types", "gamma_types", "description"]
    valid_d_types = ["SYM", "ASY", "R3"]
    
    errors = []
    
    for test_name, spec in TEST_APPLICABILITY.items():
        # Vérifier champs requis
        for field in required_fields:
            if field not in spec:
                errors.append(f"{test_name}: missing field '{field}'")
        
        # Vérifier types D
        for d_type in spec.get("d_types", []):
            if d_type not in valid_d_types:
                errors.append(f"{test_name}: invalid d_type '{d_type}'")
        
        # Vérifier cohérence rang/square
        if spec.get("requires_square") and spec.get("requires_rank") != 2:
            if spec.get("requires_rank") is not None:
                errors.append(f"{test_name}: requires_square=True but rank != 2")
    
    if errors:
        print("❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Applicability matrix valid")
        return True


# =============================================================================
# AUTO-VALIDATION AU CHARGEMENT
# =============================================================================

if __name__ == "__main__":
    validate_applicability_matrix()
    print_applicability_matrix()