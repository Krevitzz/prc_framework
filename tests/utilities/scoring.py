"""
tests/utilities/scoring.py

Scoring contextuel des observations (Section 14.4).

PRINCIPE FONDAMENTAL (Section 14.4) :
  Tests observent → Scoring interprète

Les tests retournent des observations brutes.
Le scoring applique un CONTEXTE et des PONDÉRATIONS pour calculer des scores 0-1.

ÉCHELLE UNIVERSELLE :
  0.0 = Destruction/échec complet
  0.5 = Neutre/conservation
  1.0 = Création/amélioration

Tous les scores sont multipliés par weights depuis config YAML.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Imports des résultats de tests
from .test_norm import NormResult, BoundsResult
from .test_symmetry import SymmetryResult
from .test_diversity import DiversityResult
from .test_convergence import ConvergenceResult


# =============================================================================
# CHARGEMENT CONFIGURATION
# =============================================================================

def load_weights_config(config_id: str = "weights_default") -> Dict[str, float]:
    """
    Charge les pondérations depuis un fichier YAML.
    
    Args:
        config_id: ID de la config (ex: "weights_default")
    
    Returns:
        dict {test_name: weight}
    
    Raises:
        FileNotFoundError: Si config non trouvée
    """
    config_path = Path("config") / f"{config_id}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('weights', {})


# =============================================================================
# SCORING PAR TYPE DE TEST
# =============================================================================

def score_norm_evolution(obs: NormResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-UNIV-001 (évolution norme).
    
    Échelle:
      0.0 = Explosion (max_norm > 1000)
      0.5 = Stable/oscillant
      1.0 = Convergence contrôlée
    
    Args:
        obs: NormResult du test
        context: Contexte (d_base_id, gamma_id, ...)
    
    Returns:
        Score 0-1
    """
    if obs.evolution == "explosive":
        return 0.0  # Échec total
    
    elif obs.evolution in ["stable", "oscillating"]:
        # Stable/oscillant = neutre
        return 0.5
    
    elif obs.evolution == "increasing":
        # Croissance - score selon max_norm
        if obs.max_norm > 1000:
            return 0.0  # Explosion
        elif obs.max_norm > 100:
            return 0.2  # Croissance problématique
        else:
            return 0.4  # Croissance modérée
    
    elif obs.evolution == "decreasing":
        # Décroissance - peut être OK ou effondrement
        if obs.final_norm < 1e-6:
            return 0.1  # Effondrement
        else:
            return 0.6  # Décroissance contrôlée
    
    return 0.5  # Par défaut neutre


def score_diversity_preservation(obs: DiversityResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-UNIV-002 (évolution diversité).
    
    Échelle:
      0.0 = Effondrement (collapsed)
      0.5 = Préservée
      1.0 = Augmentée (si pertinent)
    
    Args:
        obs: DiversityResult du test
        context: Contexte
    
    Returns:
        Score 0-1
    """
    ratio = obs.ratio_final_initial
    
    if obs.evolution == "collapsed":
        return 0.0  # Échec total
    
    # Ratio acceptable : [0.3, 3.0] selon Feuille de Route
    if 0.3 <= ratio <= 3.0:
        # Dans la zone acceptable
        if 0.8 <= ratio <= 1.2:
            return 1.0  # Excellente préservation
        else:
            return 0.7  # Bonne préservation
    
    # Hors zone acceptable
    if ratio < 0.3:
        return 0.2  # Perte importante mais pas effondrement
    elif ratio > 3.0:
        return 0.3  # Augmentation excessive
    
    return 0.5  # Par défaut neutre
	
def score_diversity_revised(obs: DiversityResult, context: dict) -> float:
    """
    Scoring UNIV-002 révisé (post-audit 2025-12-28).
    
    CHANGEMENTS vs version originale :
    - Bornes adaptées : [0.1, 10.0] au lieu de [0.3, 3.0]
    - Mapping linéaire par morceaux (moins compressif)
    - Saturation plancher réduite
    
    Justification (Audit UNIV-002 Section 3) :
    - 57% des ratios < 0.3 dans version originale (saturation)
    - Corrélation linéaire r=0.775 meilleure que sigmoïde
    - Zone informative originale trop étroite (21%)
    
    Mapping :
    - ratio < 0.1  : score = 0.0 (collapse total)
    - [0.1, 1.0]   : score = ratio (décroissance linéaire)
    - [1.0, 10.0]  : score = 1.0 (maintien/croissance)
    - ratio > 10.0 : score = 0.5 (explosion, pénaliser légèrement)
    
    Args:
        obs: DiversityResult avec ratio diversity_final/initial
        context: Contexte d'exécution
    
    Returns:
        Score entre 0.0 et 1.0
    """
    ratio = obs.ratio
    
    # Gérer cas pathologiques
    if np.isnan(ratio) or np.isinf(ratio):
        return 0.0
    
    # Mapping révisé
    if ratio < 0.1:
        # Collapse très sévère
        score = 0.0
    
    elif ratio < 1.0:
        # Décroissance : mapping linéaire
        # ratio=0.1 → 0.0, ratio=1.0 → 1.0
        score = (ratio - 0.1) / 0.9
    
    elif ratio <= 10.0:
        # Maintien ou croissance contrôlée : score maximal
        score = 1.0
    
    else:
        # Explosion (ratio > 10) : pénaliser légèrement
        # Peut indiquer instabilité, pas nécessairement bon
        score = 0.5
    
    return np.clip(score, 0.0, 1.0)



def score_convergence(obs: ConvergenceResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-UNIV-003 (convergence point fixe).
    
    Échelle:
      0.0 = Trivial ou divergent
      0.5 = Pas de convergence (neutre)
      1.0 = Convergence non-triviale
    
    Args:
        obs: ConvergenceResult du test
        context: Contexte
    
    Returns:
        Score 0-1
    """
    if obs.convergence_type == "non_trivial_fixed_point":
        return 1.0  # Excellent
    
    elif obs.convergence_type == "trivial_fixed_point":
        return 0.0  # Échec (trivialité bloquante)
    
    elif obs.convergence_type == "divergent":
        return 0.0  # Échec
    
    elif obs.convergence_type == "limit_cycle":
        return 0.7  # Cycle périodique = intéressant
    
    else:  # "none"
        return 0.5  # Neutre


def score_symmetry_preservation(obs: SymmetryResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-SYM-001 (préservation symétrie).
    
    Échelle dépend du contexte D :
      - Si D initial symétrique (SYM-*) :
        0.0 = Destroyed, 0.5 = Preserved, 1.0 = N/A
      - Si D initial asymétrique (ASY-*) :
        0.0 = Remains asymmetric, 0.5 = Neutral, 1.0 = Created
    
    Args:
        obs: SymmetryResult du test
        context: Contexte (DOIT contenir d_base_id)
    
    Returns:
        Score 0-1
    """
    d_base_id = context.get('d_base_id', '')
    d_type = d_base_id.split('-')[0]  # "SYM-001" → "SYM"
    
    if d_type == "SYM":
        # Base symétrique - on attend préservation
        if obs.transition == "preserved":
            return 0.5  # Neutre (comportement attendu)
        elif obs.transition == "destroyed":
            return 0.0  # Échec
        elif obs.transition == "created":
            return 0.5  # Neutre (déjà symétrique)
        else:  # "absent"
            return 0.5
    
    elif d_type == "ASY":
        # Base asymétrique - création symétrie = positif
        if obs.transition == "created":
            return 1.0  # Excellent (symétrisation)
        elif obs.transition == "destroyed":
            return 0.5  # Neutre (déjà asymétrique)
        elif obs.transition == "preserved":
            return 0.5  # Neutre
        else:  # "absent"
            return 0.0  # Reste asymétrique
    
    return 0.5  # Par défaut neutre


def score_bounds_preservation(obs: BoundsResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-BND-001 (respect bornes).
    
    Échelle:
      0.0 = Violation sévère
      0.5 = Violation mineure
      1.0 = Respectées
    
    Args:
        obs: BoundsResult du test
        context: Contexte
    
    Returns:
        Score 0-1
    """
    if not obs.violated:
        return 1.0  # Bornes respectées
    
    # Bornes violées - score selon sévérité
    if obs.status == "FAIL":
        return 0.0  # Violation sévère
    else:
        return 0.5  # Violation mineure


def score_lyapunov(obs: ConvergenceResult, context: Dict[str, Any]) -> float:
    """
    Score pour TEST-CONV-LYAPUNOV (exposant Lyapunov).
    
    Échelle:
      0.0 = Chaotique (λ > 0.1)
      0.5 = Neutre (λ ≈ 0)
      1.0 = Stable (λ < -0.01)
    
    Args:
        obs: ConvergenceResult du test
        context: Contexte
    
    Returns:
        Score 0-1
    """
    lambda_estimate = obs.final_distance  # λ stocké dans final_distance
    
    if lambda_estimate < -0.01:
        return 1.0  # Stable
    elif lambda_estimate > 0.1:
        return 0.0  # Chaotique
    else:
        # Neutre - interpolation linéaire
        # λ ∈ [-0.01, 0.1] → score ∈ [1.0, 0.0]
        return 1.0 - (lambda_estimate + 0.01) / 0.11

def score_local_diversity(obs: DiversityResult, context: dict) -> float:
    """
    Scoring UNIV-002b : Diversité locale.
    
    Mapping similaire à UNIV-002 révisé, mais interprétation différente :
    - Score élevé : Structure locale maintenue
    - Score faible : Lissage local
    
    Comparaison UNIV-002 vs UNIV-002b :
    - Si UNIV-002 bas, UNIV-002b haut : Collapse global, structure locale OK
    - Si UNIV-002 haut, UNIV-002b bas : Variance globale, mais lisse localement
    - Si les deux bas : Homogénéisation complète
    - Si les deux hauts : Maintien diversité à toutes échelles
    
    Args:
        obs: DiversityResult avec ratio local
        context: Contexte
    
    Returns:
        Score 0-1
    """
    ratio = obs.ratio_final_initial
    
    if np.isnan(ratio) or np.isinf(ratio):
        return 0.0
    
    # Même mapping que UNIV-002 révisé
    if ratio < 0.1:
        score = 0.0
    elif ratio < 1.0:
        score = (ratio - 0.1) / 0.9
    elif ratio <= 10.0:
        score = 1.0
    else:
        score = 0.5
    
    return np.clip(score, 0.0, 1.0)


def score_heterogeneity(obs: DiversityResult, context: dict) -> float:
    """
    Scoring DIV-HETERO : Hétérogénéité spatiale.
    
    Mapping :
    - ratio < 0.5  : score = 0.0 (homogénéisation)
    - [0.5, 1.5]   : score = linéaire
    - ratio > 1.5  : score = 1.0 (maintien/augmentation)
    
    Args:
        obs: DiversityResult avec ratio hétérogénéité
        context: Contexte
    
    Returns:
        Score 0-1
    """
    ratio = obs.ratio_final_initial
    
    if np.isnan(ratio) or np.isinf(ratio):
        return 0.0
    
    if ratio < 0.5:
        score = 0.0
    elif ratio < 1.5:
        score = (ratio - 0.5) / 1.0
    else:
        score = 1.0
    
    return np.clip(score, 0.0, 1.0)
# =============================================================================
# FONCTION PRINCIPALE DE SCORING
# =============================================================================

def score_observation(test_name: str, 
                     observation: Any,
                     context: Dict[str, Any],
                     weights: Dict[str, float]) -> Dict[str, float]:
    """
    Calcule score contextuel 0-1 pour une observation.
    
    Args:
        test_name: Nom du test (ex: "UNIV-001")
        observation: Résultat du test (NormResult, SymmetryResult, etc.)
        context: Contexte (d_base_id, gamma_id, modifier_id, seed)
        weights: Pondérations depuis config YAML
    
    Returns:
        dict {
            'score': float (0-1),
            'weight': float,
            'weighted_score': float
        }
    
    Exemple:
        >>> obs = test_norm_evolution(history)
        >>> context = {'d_base_id': 'SYM-001', 'gamma_id': 'GAM-001'}
        >>> weights = {'UNIV-001': 2.0}
        >>> score_observation('UNIV-001', obs, context, weights)
        {'score': 0.5, 'weight': 2.0, 'weighted_score': 1.0}
    """
    # Dispatcher vers fonction de scoring appropriée
    if test_name == "UNIV-001":
        score = score_norm_evolution(observation, context)
    
    elif test_name == "UNIV-002":
        score = score_diversity_revised(observation, context)
		
		    
    elif test_name == "UNIV-002b":
        # Nouveau test diversité locale
        score = score_local_diversity(observation, context)
    
    elif test_name == "DIV-HETERO":
        # Nouveau test hétérogénéité
        score = score_heterogeneity(observation, context)
    
    elif test_name == "UNIV-003":
        score = score_convergence(observation, context)
    
    elif test_name == "SYM-001":
        score = score_symmetry_preservation(observation, context)
    
    elif test_name == "SYM-002":
        score = score_symmetry_preservation(observation, context)  # Même logique
    
    elif test_name == "BND-001":
        score = score_bounds_preservation(observation, context)
    
    elif test_name == "CONV-LYAPUNOV":
        score = score_lyapunov(observation, context)
    
    else:
        # Test non implémenté - score neutre par défaut
        score = 0.5
    
    # Récupérer pondération
    weight = weights.get(test_name, 1.0)
    
    # Calculer score pondéré
    weighted_score = score * weight
    
    return {
        'score': score,
        'weight': weight,
        'weighted_score': weighted_score
    }


# =============================================================================
# CALCUL SCORE GLOBAL
# =============================================================================

def compute_global_score(test_scores: Dict[str, Dict[str, float]]) -> float:
    """
    Calcule score global pondéré sur échelle /20.
    
    Args:
        test_scores: dict {test_name: {'score': ..., 'weight': ..., 'weighted_score': ...}}
    
    Returns:
        Score global sur échelle /20
    
    Formule:
        score_global = (Σ weighted_scores) / (Σ weights) × 20
    
    Exemple:
        >>> scores = {
        ...     'UNIV-001': {'score': 0.8, 'weight': 2.0, 'weighted_score': 1.6},
        ...     'SYM-001': {'score': 0.5, 'weight': 1.5, 'weighted_score': 0.75},
        ...     'BND-001': {'score': 1.0, 'weight': 1.0, 'weighted_score': 1.0}
        ... }
        >>> compute_global_score(scores)
        14.88  # (1.6 + 0.75 + 1.0) / (2.0 + 1.5 + 1.0) × 20
    """
    if not test_scores:
        return 0.0
    
    total_weighted = sum(s['weighted_score'] for s in test_scores.values())
    total_weights = sum(s['weight'] for s in test_scores.values())
    
    if total_weights == 0:
        return 0.0
    
    # Score normalisé 0-1
    score_normalized = total_weighted / total_weights
    
    # Ramener sur échelle /20
    score_20 = score_normalized * 20
    
    return score_20


# =============================================================================
# HELPERS
# =============================================================================

def score_all_observations(observations: Dict[str, Any],
                          context: Dict[str, Any],
                          config_id: str = "weights_default") -> Dict[str, Any]:
    """
    Score toutes les observations d'un run.
    
    Args:
        observations: dict {test_name: observation_result}
        context: Contexte (d_base_id, gamma_id, ...)
        config_id: ID config pondérations
    
    Returns:
        dict {
            'test_scores': {test_name: {'score': ..., 'weight': ..., 'weighted_score': ...}},
            'global_score': float (0-20)
        }
    """
    # Charger pondérations
    weights = load_weights_config(config_id)
    
    # Scorer chaque observation
    test_scores = {}
    for test_name, observation in observations.items():
        test_scores[test_name] = score_observation(
            test_name, observation, context, weights
        )
    
    # Calculer score global
    global_score = compute_global_score(test_scores)
    
    return {
        'test_scores': test_scores,
        'global_score': global_score,
        'config_id': config_id
    }


def print_scores_summary(results: Dict[str, Any]):
    """Affiche résumé des scores."""
    print("\n" + "="*70)
    print("SCORES")
    print("="*70)
    
    print(f"\nConfig: {results['config_id']}")
    print(f"Score global: {results['global_score']:.2f}/20")
    
    print(f"\nDétail par test:")
    for test_name, scores in sorted(results['test_scores'].items()):
        print(f"  {test_name:<15} score={scores['score']:.3f}  "
              f"weight={scores['weight']:.1f}  "
              f"weighted={scores['weighted_score']:.3f}")
    
    print("="*70 + "\n")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_scoring_functions():
    """Valide que toutes les fonctions de scoring sont bien définies."""
    required_functions = [
        'score_norm_evolution',
        'score_diversity_preservation',
        'score_convergence',
        'score_symmetry_preservation',
        'score_bounds_preservation',
        'score_lyapunov',
    ]
    
    print("Validation fonctions de scoring:")
    for func_name in required_functions:
        if func_name in globals():
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} MANQUANT")
    
    # Vérifier cohérence échelle 0-1
    print("\nVérification échelle 0-1...")
    # TODO: Implémenter tests unitaires


if __name__ == "__main__":
    validate_scoring_functions()