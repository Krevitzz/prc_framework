"""
tests/utilities/test_convergence.py

Tests modulaires observant la convergence vers points fixes.

PRINCIPE: OBSERVATION PURE.

Ces tests détectent:
- TEST-UNIV-003: Convergence vers point fixe
- Trivialité (convergence vers identité, zéro, etc.)
- Oscillations périodiques
- Chaos
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Résultat d'un test de convergence."""
    test_name: str
    converged: bool
    iteration_converged: Optional[int]
    final_distance: float
    distance_to_identity: float
    distance_to_zero: float
    convergence_rate: float
    attractor_type: str  # "point_fixe", "oscillation", "chaos", "non_converged"
    status: str
    message: str
    blocking: bool = False
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# TEST-UNIV-003: Convergence vers point fixe
# ============================================================================

def test_convergence_to_fixed_point(history: List[np.ndarray],
                                   convergence_threshold: float = 1e-6,
                                   trivial_threshold: float = 0.01,
                                   window_size: int = 10,
                                   name: str = "TEST-UNIV-003") -> ConvergenceResult:
    """
    TEST-UNIV-003: Observe convergence vers point fixe.
    
    MESURE: ||D_{t+1} - D_t|| à chaque itération
    
    OUTPUT:
    - converged: bool (si ||D_{t+1} - D_t|| < threshold sur window_size itérations)
    - distance_final: distance au dernier point
    - iterations_to_convergence: nombre d'itérations
    
    VERDICT:
    - PASS si converged ET distance_final > trivial_threshold (non-trivial)
    - FAIL si converged ET distance_final < trivial_threshold (trivial)
    - NEUTRAL si non converged
    
    Args:
        history: Liste d'états
        convergence_threshold: Seuil pour détecter convergence
        trivial_threshold: Seuil pour détecter trivialité
        window_size: Nombre d'itérations stables requises
        name: Nom du test
    
    Returns:
        ConvergenceResult avec status
    """
    if not history or len(history) < 2:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            iteration_converged=None,
            final_distance=0.0,
            distance_to_identity=0.0,
            distance_to_zero=0.0,
            convergence_rate=0.0,
            attractor_type="unknown",
            status="FAIL",
            message="Historique insuffisant",
            blocking=True
        )
    
    # Calcule distances successives ||D_{t+1} - D_t||
    distances = []
    for i in range(len(history) - 1):
        dist = np.linalg.norm(history[i+1] - history[i], 'fro')
        distances.append(dist)
    
    # Détecte convergence (window_size valeurs consécutives < threshold)
    converged = False
    iteration_converged = None
    
    for i in range(len(distances) - window_size + 1):
        window = distances[i:i+window_size]
        if all(d < convergence_threshold for d in window):
            converged = True
            iteration_converged = i
            break
    
    # Distance finale
    if distances:
        final_distance = distances[-1]
    else:
        final_distance = 0.0
    
    # Mesure trivialité : distance à attracteurs triviaux
    final_state = history[-1]
    
    # Distance à identité (pour matrices carrées seulement)
    if final_state.ndim == 2 and final_state.shape[0] == final_state.shape[1]:
        identity = np.eye(final_state.shape[0])
        distance_to_identity = np.linalg.norm(final_state - identity, 'fro')
    else:
        distance_to_identity = np.inf
    
    # Distance à zéro
    distance_to_zero = np.linalg.norm(final_state, 'fro')
    
    # Taux de convergence (fit exponentiel)
    if len(distances) > 10:
        # Essayer fit exponentiel: distance ≈ a * exp(-rate * t)
        log_distances = np.log(np.array(distances) + 1e-10)
        try:
            rate = -np.polyfit(range(len(log_distances)), log_distances, 1)[0]
        except:
            rate = 0.0
    else:
        rate = 0.0
    
    # Détecte type d'attracteur
    if converged:
        if distance_to_zero < trivial_threshold:
            attractor_type = "zero"
        elif distance_to_identity < trivial_threshold:
            attractor_type = "identity"
        else:
            attractor_type = "point_fixe"
    else:
        # Tester oscillations
        if len(distances) > 20:
            # Autocorrélation pour détecter périodicité
            autocorr = np.correlate(distances, distances, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Chercher pics secondaires
            peaks = []
            for i in range(1, min(len(autocorr), 50)):
                if i > 0 and i < len(autocorr) - 1:
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        if autocorr[i] > 0.5:
                            peaks.append(i)
            
            if peaks:
                attractor_type = "oscillation"
            else:
                attractor_type = "chaos"
        else:
            attractor_type = "non_converged"
    
    # Verdict
    if converged:
        if attractor_type in ["zero", "identity"]:
            status = "FAIL"
            message = f"Convergence triviale vers {attractor_type} (dist={distance_to_zero:.2e})"
            blocking = True
        elif distance_to_zero < trivial_threshold or distance_to_identity < trivial_threshold:
            status = "FAIL"
            message = f"Convergence proche attracteur trivial (dist_zero={distance_to_zero:.2e}, dist_I={distance_to_identity:.2e})"
            blocking = True
        else:
            status = "PASS"
            message = f"Convergence non-triviale (iter={iteration_converged}, dist={final_distance:.2e})"
            blocking = False
    else:
        if attractor_type == "oscillation":
            status = "NEUTRAL"
            message = "Oscillations périodiques détectées"
            blocking = False
        elif attractor_type == "chaos":
            status = "NEUTRAL"
            message = "Comportement chaotique (non convergé)"
            blocking = False
        else:
            status = "NEUTRAL"
            message = "Non convergé dans le temps imparti"
            blocking = False
    
    return ConvergenceResult(
        test_name=name,
        converged=converged,
        iteration_converged=iteration_converged,
        final_distance=final_distance,
        distance_to_identity=distance_to_identity,
        distance_to_zero=distance_to_zero,
        convergence_rate=rate,
        attractor_type=attractor_type,
        status=status,
        message=message,
        blocking=blocking
    )


# ============================================================================
# TESTS COMPLÉMENTAIRES
# ============================================================================

def test_lyapunov_exponent(history: List[np.ndarray],
                          name: str = "TEST-CONV-LYAPUNOV") -> ConvergenceResult:
    """
    Estime l'exposant de Lyapunov (sensibilité conditions initiales).
    
    Exposant > 0 → chaos
    Exposant = 0 → quasi-périodique
    Exposant < 0 → convergence
    
    NOTE: Estimation grossière via distance entre états successifs.
    """
    if not history or len(history) < 10:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            iteration_converged=None,
            final_distance=0.0,
            distance_to_identity=0.0,
            distance_to_zero=0.0,
            convergence_rate=0.0,
            attractor_type="unknown",
            status="NEUTRAL",
            message="Historique insuffisant pour Lyapunov",
            blocking=False
        )
    
    # Calcule log des distances
    log_distances = []
    for i in range(len(history) - 1):
        dist = np.linalg.norm(history[i+1] - history[i], 'fro')
        if dist > 0:
            log_distances.append(np.log(dist))
    
    if not log_distances:
        lyapunov = 0.0
    else:
        # Exposant de Lyapunov ≈ pente de log(distance) vs temps
        lyapunov = np.polyfit(range(len(log_distances)), log_distances, 1)[0]
    
    # Classification
    if lyapunov > 0.01:
        attractor_type = "chaos"
        status = "NEUTRAL"
        message = f"Exposant Lyapunov > 0: chaos détecté (λ={lyapunov:.3f})"
    elif lyapunov < -0.01:
        attractor_type = "convergent"
        status = "PASS"
        message = f"Exposant Lyapunov < 0: convergent (λ={lyapunov:.3f})"
    else:
        attractor_type = "quasi_periodic"
        status = "NEUTRAL"
        message = f"Exposant Lyapunov ≈ 0: quasi-périodique (λ={lyapunov:.3f})"
    
    return ConvergenceResult(
        test_name=name,
        converged=attractor_type == "convergent",
        iteration_converged=None,
        final_distance=0.0,
        distance_to_identity=0.0,
        distance_to_zero=0.0,
        convergence_rate=-lyapunov,
        attractor_type=attractor_type,
        status=status,
        message=message,
        blocking=False
    )


def test_periodic_orbit(history: List[np.ndarray],
                       max_period: int = 50,
                       tolerance: float = 1e-4,
                       name: str = "TEST-CONV-PERIODIC") -> ConvergenceResult:
    """
    Détecte orbites périodiques (cycles).
    
    Recherche T tel que ||D_{t+T} - D_t|| < tolerance
    """
    if not history or len(history) < max_period + 1:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            iteration_converged=None,
            final_distance=0.0,
            distance_to_identity=0.0,
            distance_to_zero=0.0,
            convergence_rate=0.0,
            attractor_type="unknown",
            status="NEUTRAL",
            message="Historique insuffisant pour détecter périodicité",
            blocking=False
        )
    
    # Chercher période
    period_found = None
    for period in range(1, max_period + 1):
        # Vérifier si D_{t+period} ≈ D_t pour plusieurs t
        matches = 0
        for t in range(len(history) - period):
            dist = np.linalg.norm(history[t + period] - history[t], 'fro')
            if dist < tolerance:
                matches += 1
        
        # Si >80% des points respectent la période
        if matches > 0.8 * (len(history) - period):
            period_found = period
            break
    
    if period_found:
        status = "NEUTRAL"
        message = f"Orbite périodique détectée: période={period_found}"
        attractor_type = f"periodic_{period_found}"
        converged = True
    else:
        status = "NEUTRAL"
        message = "Pas d'orbite périodique détectée"
        attractor_type = "non_periodic"
        converged = False
    
    return ConvergenceResult(
        test_name=name,
        converged=converged,
        iteration_converged=period_found,
        final_distance=0.0,
        distance_to_identity=0.0,
        distance_to_zero=0.0,
        convergence_rate=0.0,
        attractor_type=attractor_type,
        status=status,
        message=message,
        blocking=False
    )


def test_distance_to_custom_target(history: List[np.ndarray],
                                  target: np.ndarray,
                                  threshold: float = 0.01,
                                  name: str = "TEST-CONV-TARGET") -> ConvergenceResult:
    """
    Mesure distance à une cible spécifique.
    
    Utile pour tester convergence vers pattern connu.
    """
    if not history:
        return ConvergenceResult(
            test_name=name,
            converged=False,
            iteration_converged=None,
            final_distance=0.0,
            distance_to_identity=0.0,
            distance_to_zero=0.0,
            convergence_rate=0.0,
            attractor_type="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Calcule distances à la cible
    distances = []
    for state in history:
        if state.shape != target.shape:
            return ConvergenceResult(
                test_name=name,
                converged=False,
                iteration_converged=None,
                final_distance=0.0,
                distance_to_identity=0.0,
                distance_to_zero=0.0,
                convergence_rate=0.0,
                attractor_type="error",
                status="FAIL",
                message="Forme incompatible avec cible",
                blocking=True
            )
        
        dist = np.linalg.norm(state - target, 'fro')
        distances.append(dist)
    
    final_distance = distances[-1]
    min_distance = min(distances)
    
    # Détecte convergence
    converged = final_distance < threshold
    
    if converged:
        # Trouve première fois sous threshold
        iteration_converged = next((i for i, d in enumerate(distances) if d < threshold), None)
        status = "PASS"
        message = f"Convergence vers cible (dist={final_distance:.2e})"
    else:
        iteration_converged = None
        if min_distance < threshold:
            status = "NEUTRAL"
            message = f"Approche cible (min_dist={min_distance:.2e}) mais diverge"
        else:
            status = "FAIL"
            message = f"Pas de convergence vers cible (dist={final_distance:.2e})"
    
    return ConvergenceResult(
        test_name=name,
        converged=converged,
        iteration_converged=iteration_converged,
        final_distance=final_distance,
        distance_to_identity=0.0,
        distance_to_zero=0.0,
        convergence_rate=0.0,
        attractor_type="target",
        status=status,
        message=message,
        blocking=False
    )