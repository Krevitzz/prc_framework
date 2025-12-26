"""
tests/utilities/test_convergence.py

Tests modulaires observant la convergence (Section 14.4).

PRINCIPE: OBSERVATION PURE.

Ces tests détectent:
- TEST-UNIV-003: Convergence vers point fixe
- TEST-CONV-LYAPUNOV: Stabilité via exposant Lyapunov
- TEST-CONV-SPEED: Vitesse de convergence
- TEST-CONV-OSCILLATION: Détection cycles périodiques
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Résultat d'un test de convergence."""
    test_name: str
    converged: bool
    convergence_iteration: Optional[int]
    final_distance: float
    convergence_type: str  # "fixed_point" | "limit_cycle" | "none" | "divergent"
    status: str  # "PASS" | "FAIL" | "NEUTRAL"
    message: str
    blocking: bool = False
    
    # Métriques additionnelles
    mean_distance: Optional[float] = None
    std_distance: Optional[float] = None
    trend: Optional[str] = None
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# TEST-UNIV-003: Convergence vers point fixe
# ============================================================================

def test_convergence_to_fixed_point(
    history: List[np.ndarray],
    threshold: float = 1e-6,
    window: int = 10,
    trivial_threshold: float = 1e-3,
    name: str = "TEST-UNIV-003"
) -> ConvergenceResult:
    """
    TEST-UNIV-003: Observe convergence vers point fixe.
    
    MESURE: ||D_{t+1} - D_t|| à chaque itération
    
    VERDICT:
    - PASS si convergé ET distance_finale > trivial_threshold (non-trivial)
    - FAIL si convergé ET distance_finale < trivial_threshold (trivial)
    - NEUTRAL si non convergé
    
    Args:
        history: Liste d'états
        threshold: Seuil ||D_{t+1} - D_t|| pour convergence
        window: Nombre iterations consécutives < threshold
        trivial_threshold: Seuil distinction trivial/non-trivial
        name: Nom du test
    
    Returns:
        ConvergenceResult avec status et détails
    """
    if len(history) < 2:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="none",
            status="NEUTRAL",
            message="Historique insuffisant",
            blocking=False
        )
    
    # Calculer distances successives
    distances = []
    for i in range(len(history) - 1):
        diff = history[i+1] - history[i]
        if diff.ndim == 2:
            dist = np.linalg.norm(diff, 'fro')
        else:
            dist = np.linalg.norm(diff.flatten())
        distances.append(dist)
    
    if not distances:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="none",
            status="NEUTRAL",
            message="Pas de distances calculables",
            blocking=False
        )
    
    # Détecter convergence
    converged = False
    convergence_iter = None
    
    for i in range(len(distances) - window + 1):
        if all(d < threshold for d in distances[i:i+window]):
            converged = True
            convergence_iter = i
            break
    
    final_distance = distances[-1]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Détecter tendance
    if len(distances) > 1:
        trend_coef = np.polyfit(range(len(distances)), distances, 1)[0]
        if trend_coef < -1e-6:
            trend = "decreasing"
        elif trend_coef > 1e-6:
            trend = "increasing"
        else:
            trend = "stable"
    else:
        trend = "unknown"
    
    # Classifier type de convergence
    if not converged:
        if trend == "increasing":
            convergence_type = "divergent"
        else:
            convergence_type = "none"
    else:
        # Convergé - vérifier si trivial
        # Mesure distance au point initial (proxy "trivialité")
        final_state = history[-1]
        initial_state = history[0]
        
        if final_state.ndim == 2:
            dist_to_initial = np.linalg.norm(final_state - initial_state, 'fro')
        else:
            dist_to_initial = np.linalg.norm((final_state - initial_state).flatten())
        
        if dist_to_initial < trivial_threshold:
            convergence_type = "trivial_fixed_point"
        else:
            convergence_type = "non_trivial_fixed_point"
    
    # Verdict
    if converged and convergence_type == "non_trivial_fixed_point":
        status = "PASS"
        message = f"Convergence non-triviale iter={convergence_iter}, dist={final_distance:.2e}"
        blocking = False
    elif converged and convergence_type == "trivial_fixed_point":
        status = "FAIL"
        message = f"Convergence triviale iter={convergence_iter}, dist_init={dist_to_initial:.2e}"
        blocking = True  # Trivialité est bloquante
    elif convergence_type == "divergent":
        status = "FAIL"
        message = f"Divergence détectée, trend={trend}"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Pas de convergence, final_dist={final_distance:.2e}, trend={trend}"
        blocking = False
    
    return ConvergenceResult(
        test_name=name,
        converged=converged,
        convergence_iteration=convergence_iter,
        final_distance=final_distance,
        convergence_type=convergence_type,
        status=status,
        message=message,
        blocking=blocking,
        mean_distance=mean_distance,
        std_distance=std_distance,
        trend=trend
    )


# ============================================================================
# TEST-CONV-LYAPUNOV: Exposant de Lyapunov
# ============================================================================

def test_lyapunov_exponent(
    history: List[np.ndarray],
    perturbation_size: float = 1e-8,
    name: str = "TEST-CONV-LYAPUNOV"
) -> ConvergenceResult:
    """
    TEST-CONV-LYAPUNOV: Estime l'exposant de Lyapunov.
    
    MESURE: λ = lim_{t→∞} (1/t) log(||δD_t|| / ||δD_0||)
    
    INTERPRÉTATION:
    - λ < 0 : Stabilité (perturbations décroissent)
    - λ = 0 : Neutre
    - λ > 0 : Instabilité/chaos
    
    VERDICT:
    - PASS si λ < 0 (stable)
    - FAIL si λ > 0.1 (chaotique)
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états
        perturbation_size: Taille perturbation initiale
        name: Nom du test
    
    Returns:
        ConvergenceResult avec estimation λ
    
    Note:
        Estimation simplifiée - pas de rerun avec perturbation.
        Utilise croissance distances successives comme proxy.
    """
    if len(history) < 10:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="insufficient_data",
            status="NEUTRAL",
            message="Historique trop court pour Lyapunov",
            blocking=False
        )
    
    # Calculer distances successives (proxy de ||δD_t||)
    distances = []
    for i in range(len(history) - 1):
        diff = history[i+1] - history[i]
        if diff.ndim == 2:
            dist = np.linalg.norm(diff, 'fro')
        else:
            dist = np.linalg.norm(diff.flatten())
        distances.append(dist)
    
    # Filtrer zéros (log impossible)
    distances_nonzero = [d for d in distances if d > 1e-12]
    
    if len(distances_nonzero) < 5:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="insufficient_data",
            status="NEUTRAL",
            message="Pas assez de distances non-nulles",
            blocking=False
        )
    
    # Estimer λ via régression log(dist) vs t
    log_distances = np.log(distances_nonzero)
    t = np.arange(len(log_distances))
    
    # Régression linéaire
    lambda_estimate = np.polyfit(t, log_distances, 1)[0]
    
    # Verdict
    if lambda_estimate < -0.01:
        status = "PASS"
        message = f"Stable: λ={lambda_estimate:.4f}"
        convergence_type = "stable"
        blocking = False
    elif lambda_estimate > 0.1:
        status = "FAIL"
        message = f"Instable/chaotique: λ={lambda_estimate:.4f}"
        convergence_type = "chaotic"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Neutre: λ={lambda_estimate:.4f}"
        convergence_type = "neutral"
        blocking = False
    
    return ConvergenceResult(
        test_name=name,
        converged=(lambda_estimate < 0),
        convergence_iteration=None,
        final_distance=lambda_estimate,  # Stocke λ dans final_distance
        convergence_type=convergence_type,
        status=status,
        message=message,
        blocking=blocking
    )


# ============================================================================
# TEST-CONV-SPEED: Vitesse de convergence
# ============================================================================

def test_convergence_speed(
    history: List[np.ndarray],
    threshold: float = 1e-6,
    name: str = "TEST-CONV-SPEED"
) -> ConvergenceResult:
    """
    TEST-CONV-SPEED: Mesure vitesse de convergence.
    
    MESURE: Nombre iterations avant ||D_{t+1} - D_t|| < threshold
    
    INTERPRÉTATION:
    - Rapide (<100 iter) : Convergence rapide
    - Normale (100-500 iter) : Convergence standard
    - Lente (>500 iter) : Convergence lente
    
    VERDICT: NEUTRAL (observationnel)
    
    Args:
        history: Liste d'états
        threshold: Seuil convergence
        name: Nom du test
    
    Returns:
        ConvergenceResult avec vitesse
    """
    if len(history) < 2:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="none",
            status="NEUTRAL",
            message="Historique insuffisant",
            blocking=False
        )
    
    # Calculer distances
    for i in range(len(history) - 1):
        diff = history[i+1] - history[i]
        if diff.ndim == 2:
            dist = np.linalg.norm(diff, 'fro')
        else:
            dist = np.linalg.norm(diff.flatten())
        
        if dist < threshold:
            # Convergence atteinte
            if i < 100:
                speed_type = "rapid"
            elif i < 500:
                speed_type = "normal"
            else:
                speed_type = "slow"
            
            return ConvergenceResult(
                test_name=name,
                converged=True,
                convergence_iteration=i,
                final_distance=dist,
                convergence_type=speed_type,
                status="NEUTRAL",
                message=f"Convergence {speed_type}: iter={i}",
                blocking=False
            )
    
    # Pas de convergence
    final_dist = np.linalg.norm((history[-1] - history[-2]).flatten())
    
    return ConvergenceResult(
        test_name=name,
        converged=False,
        convergence_iteration=None,
        final_distance=final_dist,
        convergence_type="none",
        status="NEUTRAL",
        message=f"Pas de convergence: final_dist={final_dist:.2e}",
        blocking=False
    )


# ============================================================================
# TEST-CONV-OSCILLATION: Détection oscillations
# ============================================================================

def test_oscillation_detection(
    history: List[np.ndarray],
    max_period: int = 50,
    tolerance: float = 1e-4,
    name: str = "TEST-CONV-OSCILLATION"
) -> ConvergenceResult:
    """
    TEST-CONV-OSCILLATION: Détecte cycles périodiques.
    
    MESURE: Cherche période T telle que ||D_{t+T} - D_t|| < tolerance
    
    INTERPRÉTATION:
    - Période détectée : Oscillation régulière
    - Pas de période : Soit convergence, soit chaos
    
    VERDICT: NEUTRAL (observationnel)
    
    Args:
        history: Liste d'états
        max_period: Période maximale à tester
        tolerance: Seuil similarité
        name: Nom du test
    
    Returns:
        ConvergenceResult avec période détectée (si existe)
    """
    if len(history) < 2 * max_period:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            convergence_iteration=None,
            final_distance=0.0,
            convergence_type="insufficient_data",
            status="NEUTRAL",
            message="Historique trop court pour détecter oscillations",
            blocking=False
        )
    
    # Tester chaque période possible
    for period in range(2, max_period + 1):
        # Vérifier si D[t] ≈ D[t+period] pour plusieurs t
        matches = 0
        total_checks = min(10, len(history) - period)
        
        for i in range(len(history) - period - total_checks, len(history) - period):
            diff = history[i + period] - history[i]
            if diff.ndim == 2:
                dist = np.linalg.norm(diff, 'fro')
            else:
                dist = np.linalg.norm(diff.flatten())
            
            if dist < tolerance:
                matches += 1
        
        # Si ≥80% des vérifications matchent
        if matches >= 0.8 * total_checks:
            return ConvergenceResult(
                test_name=name,
                converged=True,
                convergence_iteration=period,  # Stocke période dans convergence_iteration
                final_distance=tolerance,
                convergence_type="limit_cycle",
                status="NEUTRAL",
                message=f"Cycle périodique détecté: période={period}",
                blocking=False
            )
    
    # Pas de cycle détecté
    return ConvergenceResult(
        test_name=name,
        converged=False,
        convergence_iteration=None,
        final_distance=0.0,
        convergence_type="none",
        status="NEUTRAL",
        message="Pas de cycle périodique détecté",
        blocking=False
    )