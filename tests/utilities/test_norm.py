"""
tests/utilities/test_norm.py

Tests modulaires observant l'évolution des normes et bornes.

PRINCIPE: OBSERVATION PURE.

Ces tests détectent:
- TEST-UNIV-001: Évolution norme Frobenius
- TEST-BND-001: Respect bornes initiales
- Explosions numériques (NaN, Inf)
- Croissance/décroissance incontrôlée
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class NormResult:
    """Résultat d'un test de norme."""
    test_name: str
    initial_norm: float
    final_norm: float
    max_norm: float
    min_norm: float
    evolution: str  # "increasing", "decreasing", "stable", "oscillating", "explosive"
    trend_coefficient: float
    status: str  # "PASS", "FAIL", "NEUTRAL"
    message: str
    blocking: bool = False
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


@dataclass
class BoundsResult:
    """Résultat d'un test de bornes."""
    test_name: str
    initial_bounds: tuple
    final_bounds: tuple
    max_violation: float
    violated: bool
    status: str
    message: str
    blocking: bool = False
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# TEST-UNIV-001: Évolution norme Frobenius
# ============================================================================

def test_norm_evolution(history: List[np.ndarray],
                       norm_type: str = "frobenius",
                       explosion_threshold: float = 1000.0,
                       name: str = "TEST-UNIV-001") -> NormResult:
    """
    TEST-UNIV-001: Observe l'évolution de la norme.
    
    MESURE: ||D||_F à chaque itération
    
    OUTPUT:
    - trend: "increasing" | "decreasing" | "stable" | "oscillating"
    - initial, final, max normes
    
    VERDICT:
    - PASS si trend ∈ {stable, oscillating} ET max < explosion_threshold
    - FAIL si trend = increasing ET max > explosion_threshold (explosion)
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états
        norm_type: "frobenius", "spectral", "max", "1", "2", "inf"
        explosion_threshold: Seuil d'explosion (1000 par défaut)
        name: Nom du test
    
    Returns:
        NormResult avec status et détails
    """
    if not history:
        return NormResult(
            test_name=name,
            initial_norm=0.0,
            final_norm=0.0,
            max_norm=0.0,
            min_norm=0.0,
            evolution="unknown",
            trend_coefficient=0.0,
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Calcule normes selon type
    norms = []
    for state in history:
        # Vérifier NaN/Inf
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return NormResult(
                test_name=name,
                initial_norm=0.0,
                final_norm=0.0,
                max_norm=np.inf,
                min_norm=0.0,
                evolution="explosive",
                trend_coefficient=np.inf,
                status="FAIL",
                message="NaN ou Inf détecté (explosion numérique)",
                blocking=True
            )
        
        if norm_type == "frobenius":
            norm = np.linalg.norm(state, 'fro')
        elif norm_type == "spectral":
            if state.ndim == 2:
                norm = np.linalg.norm(state, 2)
            else:
                norm = np.linalg.norm(state.flatten(), 2)
        elif norm_type == "max":
            norm = np.max(np.abs(state))
        elif norm_type in ["1", "2", "inf"]:
            norm = np.linalg.norm(state.flatten(), int(norm_type) if norm_type != "inf" else np.inf)
        else:
            raise ValueError(f"norm_type inconnu: {norm_type}")
        
        norms.append(norm)
    
    # Statistiques
    initial_norm = norms[0]
    final_norm = norms[-1]
    max_norm = max(norms)
    min_norm = min(norms)
    
    # Détecte tendance via régression linéaire
    if len(norms) > 1:
        trend_coef = np.polyfit(range(len(norms)), norms, 1)[0]
    else:
        trend_coef = 0.0
    
    # Classification tendance
    relative_change = abs(final_norm - initial_norm) / (initial_norm + 1e-10)
    
    if relative_change < 0.1:
        evolution = "stable"
    elif trend_coef > 0:
        evolution = "increasing"
    elif trend_coef < 0:
        evolution = "decreasing"
    
    # Détecte oscillations
    if len(norms) > 2:
        diffs = np.diff(norms)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        if sign_changes > len(norms) * 0.2:
            evolution = "oscillating"
    
    # Détecte explosion
    if max_norm > explosion_threshold:
        evolution = "explosive"
    
    # Verdict
    if evolution in ["stable", "oscillating"] and max_norm < explosion_threshold:
        status = "PASS"
        message = f"Évolution contrôlée: {evolution}, max={max_norm:.2f}"
    elif evolution == "explosive":
        status = "FAIL"
        message = f"Explosion détectée: max={max_norm:.2e} > {explosion_threshold}"
        blocking = True
    elif evolution == "increasing" and max_norm > explosion_threshold:
        status = "FAIL"
        message = f"Croissance explosive: {initial_norm:.2f} → {final_norm:.2e}"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Évolution: {evolution}, {initial_norm:.2f} → {final_norm:.2f}"
        blocking = False
    
    return NormResult(
        test_name=name,
        initial_norm=initial_norm,
        final_norm=final_norm,
        max_norm=max_norm,
        min_norm=min_norm,
        evolution=evolution,
        trend_coefficient=trend_coef,
        status=status,
        message=message,
        blocking=blocking if evolution == "explosive" else False
    )


# ============================================================================
# TEST-BND-001: Respect bornes initiales
# ============================================================================

def test_bounds_preservation(history: List[np.ndarray],
                            initial_bounds: tuple = None,
                            tolerance_factor: float = 1.1,
                            name: str = "TEST-BND-001") -> BoundsResult:
    """
    TEST-BND-001: Vérifie que Γ respecte les bornes initiales.
    
    MESURE: min(D_t), max(D_t) à chaque itération
    
    VERDICT:
    - PASS si final dans tolerance_factor × initial_bounds
    - FAIL si final hors 2× initial_bounds
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états
        initial_bounds: (min, max) attendus. Si None, déduit de D_0
        tolerance_factor: Facteur de tolérance (1.1 = 10% dépassement OK)
        name: Nom du test
    
    Returns:
        BoundsResult avec status
    """
    if not history:
        return BoundsResult(
            test_name=name,
            initial_bounds=(0, 0),
            final_bounds=(0, 0),
            max_violation=0.0,
            violated=True,
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Déterminer bornes initiales
    if initial_bounds is None:
        initial_min = np.min(history[0])
        initial_max = np.max(history[0])
        initial_bounds = (initial_min, initial_max)
    else:
        initial_min, initial_max = initial_bounds
    
    # Calculer bornes finales
    final_min = np.min(history[-1])
    final_max = np.max(history[-1])
    final_bounds = (final_min, final_max)
    
    # Calculer violation maximale
    max_violation_low = max(0, initial_min - final_min)
    max_violation_high = max(0, final_max - initial_max)
    max_violation = max(max_violation_low, max_violation_high)
    
    # Bornes tolérées
    range_width = initial_max - initial_min
    tolerance = range_width * (tolerance_factor - 1.0)
    
    tolerated_min = initial_min - tolerance
    tolerated_max = initial_max + tolerance
    
    strict_min = initial_min - 2 * tolerance
    strict_max = initial_max + 2 * tolerance
    
    # Verdict
    violated = (final_min < tolerated_min or final_max > tolerated_max)
    severe_violation = (final_min < strict_min or final_max > strict_max)
    
    if not violated:
        status = "PASS"
        message = f"Bornes respectées: [{final_min:.3f}, {final_max:.3f}] ⊆ [{initial_min:.3f}, {initial_max:.3f}]×{tolerance_factor}"
        blocking = False
    elif severe_violation:
        status = "FAIL"
        message = f"Violation sévère: [{final_min:.3f}, {final_max:.3f}] >> [{initial_min:.3f}, {initial_max:.3f}]"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Dépassement mineur: [{final_min:.3f}, {final_max:.3f}] vs [{initial_min:.3f}, {initial_max:.3f}]"
        blocking = False
    
    return BoundsResult(
        test_name=name,
        initial_bounds=initial_bounds,
        final_bounds=final_bounds,
        max_violation=max_violation,
        violated=violated,
        status=status,
        message=message,
        blocking=blocking
    )


# ============================================================================
# TESTS COMPLÉMENTAIRES
# ============================================================================

def test_spectral_evolution(history: List[np.ndarray],
                           name: str = "TEST-STR-002") -> NormResult:
    """
    TEST-STR-002: Évolution du rayon spectral (valeurs propres).
    
    APPLICABLE: Matrices symétriques uniquement.
    
    MESURE: max(|λ_i|) à chaque itération
    """
    if not history or any(state.ndim != 2 for state in history):
        return NormResult(
            test_name=name,
            initial_norm=0.0,
            final_norm=0.0,
            max_norm=0.0,
            min_norm=0.0,
            evolution="N/A",
            trend_coefficient=0.0,
            status="NEUTRAL",
            message="Test non applicable (tenseur rang ≠ 2)",
            blocking=False
        )
    
    spectral_radii = []
    for state in history:
        try:
            # Essayer décomposition symétrique (plus rapide)
            eigenvalues = np.linalg.eigvalsh(state)
        except np.linalg.LinAlgError:
            # Si non symétrique, utiliser eig général
            try:
                eigenvalues = np.linalg.eigvals(state)
            except np.linalg.LinAlgError:
                return NormResult(
                    test_name=name,
                    initial_norm=0.0,
                    final_norm=0.0,
                    max_norm=0.0,
                    min_norm=0.0,
                    evolution="error",
                    trend_coefficient=0.0,
                    status="FAIL",
                    message="Calcul valeurs propres échoué",
                    blocking=True
                )
        
        spectral_radius = np.max(np.abs(eigenvalues))
        spectral_radii.append(spectral_radius)
    
    initial_sr = spectral_radii[0]
    final_sr = spectral_radii[-1]
    max_sr = max(spectral_radii)
    min_sr = min(spectral_radii)
    
    # Tendance
    if len(spectral_radii) > 1:
        trend_coef = np.polyfit(range(len(spectral_radii)), spectral_radii, 1)[0]
    else:
        trend_coef = 0.0
    
    if abs(trend_coef) < 0.01:
        evolution = "stable"
    elif trend_coef > 0:
        evolution = "increasing"
    else:
        evolution = "decreasing"
    
    # Verdict
    if max_sr > 100:
        status = "FAIL"
        message = f"Explosion spectrale: max(|λ|) = {max_sr:.2e}"
        blocking = True
    elif evolution == "stable":
        status = "PASS"
        message = f"Rayon spectral stable: {final_sr:.3f}"
        blocking = False
    else:
        status = "NEUTRAL"
        message = f"Évolution spectrale: {evolution}, {initial_sr:.3f} → {final_sr:.3f}"
        blocking = False
    
    return NormResult(
        test_name=name,
        initial_norm=initial_sr,
        final_norm=final_sr,
        max_norm=max_sr,
        min_norm=min_sr,
        evolution=evolution,
        trend_coefficient=trend_coef,
        status=status,
        message=message,
        blocking=blocking
    )


def test_element_wise_bounds(history: List[np.ndarray],
                            lower_bound: float = -1.0,
                            upper_bound: float = 1.0,
                            name: str = "TEST-BND-ELEM") -> BoundsResult:
    """
    Vérifie que TOUS les éléments restent dans [lower_bound, upper_bound].
    
    Plus strict que TEST-BND-001 qui regarde min/max globaux.
    """
    if not history:
        return BoundsResult(
            test_name=name,
            initial_bounds=(lower_bound, upper_bound),
            final_bounds=(0, 0),
            max_violation=0.0,
            violated=True,
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    violations = []
    for i, state in enumerate(history):
        min_val = np.min(state)
        max_val = np.max(state)
        
        if min_val < lower_bound or max_val > upper_bound:
            violation = max(lower_bound - min_val, max_val - upper_bound, 0)
            violations.append((i, min_val, max_val, violation))
    
    if not violations:
        status = "PASS"
        message = f"Tous éléments dans [{lower_bound}, {upper_bound}]"
        max_violation = 0.0
        violated = False
        blocking = False
    else:
        max_violation = max(v[3] for v in violations)
        first_violation = violations[0]
        
        if max_violation > 10 * (upper_bound - lower_bound):
            status = "FAIL"
            message = f"Violation massive: {len(violations)} itérations, max={max_violation:.2e}"
            violated = True
            blocking = True
        else:
            status = "NEUTRAL"
            message = f"{len(violations)} violations mineures (max={max_violation:.3f})"
            violated = True
            blocking = False
    
    final_bounds = (np.min(history[-1]), np.max(history[-1]))
    
    return BoundsResult(
        test_name=name,
        initial_bounds=(lower_bound, upper_bound),
        final_bounds=final_bounds,
        max_violation=max_violation,
        violated=violated,
        status=status,
        message=message,
        blocking=blocking
    )