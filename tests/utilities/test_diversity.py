"""
tests/utilities/test_diversity.py

Tests modulaires observant l'évolution de la diversité structurale.

PRINCIPE: OBSERVATION PURE.

Ces tests détectent:
- TEST-UNIV-002: Évolution diversité (écart-type)
- Homogénéisation (perte de diversité)
- Différenciation (augmentation diversité)
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class DiversityResult:
    """Résultat d'un test de diversité."""
    test_name: str
    initial_diversity: float
    final_diversity: float
    max_diversity: float
    min_diversity: float
    ratio_final_initial: float
    evolution: str  # "increasing", "decreasing", "stable", "collapsed"
    status: str
    message: str
    blocking: bool = False
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# TEST-UNIV-002: Évolution diversité (std)
# ============================================================================

def test_diversity_preservation(history: List[np.ndarray],
                               collapse_threshold: float = 0.1,
                               explosion_threshold: float = 10.0,
                               name: str = "TEST-UNIV-002") -> DiversityResult:
    """
    TEST-UNIV-002: Observe l'évolution de la diversité.
    
    MESURE: σ(D_flat) à chaque itération (écart-type de tous les éléments)
    
    OUTPUT:
    - trend: évolution de la diversité
    - ratio_final/initial
    
    VERDICT:
    - PASS si ratio ∈ [0.3, 3.0] (diversité préservée)
    - FAIL si ratio < collapse_threshold (homogénéisation) OU > explosion_threshold (explosion)
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états
        collapse_threshold: Ratio minimal acceptable (0.1 = 90% perte)
        explosion_threshold: Ratio maximal acceptable (10.0 = 10× augmentation)
        name: Nom du test
    
    Returns:
        DiversityResult avec status
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Calcule diversité (écart-type) pour chaque état
    diversities = []
    for state in history:
        flat = state.flatten()
        diversity = np.std(flat)
        diversities.append(diversity)
    
    # Statistiques
    initial_div = diversities[0]
    final_div = diversities[-1]
    max_div = max(diversities)
    min_div = min(diversities)
    
    # Ratio (avec protection division par zéro)
    if initial_div < 1e-10:
        if final_div < 1e-10:
            ratio = 1.0
        else:
            ratio = np.inf
    else:
        ratio = final_div / initial_div
    
    # Détecte tendance
    if len(diversities) > 1:
        trend_coef = np.polyfit(range(len(diversities)), diversities, 1)[0]
        relative_change = abs(final_div - initial_div) / (initial_div + 1e-10)
        
        if relative_change < 0.1:
            evolution = "stable"
        elif trend_coef > 0:
            evolution = "increasing"
        elif trend_coef < 0:
            evolution = "decreasing"
        else:
            evolution = "stable"
    else:
        evolution = "stable"
        trend_coef = 0.0
    
    # Détecte effondrement
    if ratio < collapse_threshold:
        evolution = "collapsed"
    
    # Verdict
    if 0.3 <= ratio <= 3.0:
        status = "PASS"
        message = f"Diversité préservée: ratio={ratio:.2f}"
        blocking = False
    elif ratio < collapse_threshold:
        status = "FAIL"
        message = f"Homogénéisation: {initial_div:.3e} → {final_div:.3e} (ratio={ratio:.3f})"
        blocking = True
    elif ratio > explosion_threshold:
        status = "FAIL"
        message = f"Explosion diversité: {initial_div:.3e} → {final_div:.3e} (ratio={ratio:.1f})"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Évolution diversité: {evolution}, ratio={ratio:.2f}"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_div,
        final_diversity=final_div,
        max_diversity=max_div,
        min_diversity=min_div,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )


# ============================================================================
# TESTS COMPLÉMENTAIRES
# ============================================================================

def test_entropy_evolution(history: List[np.ndarray],
                          name: str = "TEST-DIV-ENTROPY") -> DiversityResult:
    """
    Mesure l'entropie de la distribution des valeurs.
    
    Complément à la diversité (std) : détecte structures émergentes.
    
    MESURE: Entropie de Shannon de l'histogramme des valeurs
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def compute_entropy(state, n_bins=50):
        """Calcule l'entropie de Shannon."""
        flat = state.flatten()
        # Histogramme normalisé
        hist, _ = np.histogram(flat, bins=n_bins, density=True)
        # Normaliser pour avoir une distribution de probabilité
        hist = hist / (np.sum(hist) + 1e-10)
        # Entropie (éviter log(0))
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log(hist_nonzero))
        return entropy
    
    entropies = [compute_entropy(state) for state in history]
    
    initial_ent = entropies[0]
    final_ent = entropies[-1]
    max_ent = max(entropies)
    min_ent = min(entropies)
    
    ratio = final_ent / (initial_ent + 1e-10)
    
    if ratio < 0.5:
        evolution = "decreasing"
        status = "NEUTRAL"
        message = f"Entropie réduite: {initial_ent:.2f} → {final_ent:.2f}"
    elif ratio > 2.0:
        evolution = "increasing"
        status = "NEUTRAL"
        message = f"Entropie augmentée: {initial_ent:.2f} → {final_ent:.2f}"
    else:
        evolution = "stable"
        status = "PASS"
        message = f"Entropie stable: {final_ent:.2f}"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_ent,
        final_diversity=final_ent,
        max_diversity=max_ent,
        min_diversity=min_ent,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )


def test_range_evolution(history: List[np.ndarray],
                        name: str = "TEST-DIV-RANGE") -> DiversityResult:
    """
    Mesure l'évolution de la plage dynamique (max - min).
    
    Complément à std : détecte compression/expansion de la distribution.
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    ranges = []
    for state in history:
        range_val = np.max(state) - np.min(state)
        ranges.append(range_val)
    
    initial_range = ranges[0]
    final_range = ranges[-1]
    max_range = max(ranges)
    min_range = min(ranges)
    
    ratio = final_range / (initial_range + 1e-10)
    
    if ratio < 0.3:
        evolution = "compressed"
        status = "NEUTRAL"
        message = f"Plage compressée: {initial_range:.3f} → {final_range:.3f}"
    elif ratio > 3.0:
        evolution = "expanded"
        status = "NEUTRAL"
        message = f"Plage élargie: {initial_range:.3f} → {final_range:.3f}"
    else:
        evolution = "stable"
        status = "PASS"
        message = f"Plage stable: {final_range:.3f}"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_range,
        final_diversity=final_range,
        max_diversity=max_range,
        min_diversity=min_range,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )


def test_uniformity(history: List[np.ndarray],
                   name: str = "TEST-DIV-UNIFORM") -> DiversityResult:
    """
    Détecte convergence vers valeur uniforme.
    
    MESURE: Coefficient de variation CV = std / |mean|
    
    CV → 0 indique uniformisation
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def compute_cv(state):
        """Coefficient de variation."""
        flat = state.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        return std / (abs(mean) + 1e-10)
    
    cvs = [compute_cv(state) for state in history]
    
    initial_cv = cvs[0]
    final_cv = cvs[-1]
    max_cv = max(cvs)
    min_cv = min(cvs)
    
    ratio = final_cv / (initial_cv + 1e-10)
    
    # Détecte uniformisation
    if final_cv < 0.01:
        evolution = "uniform"
        status = "FAIL"
        message = f"Uniformisation détectée: CV={final_cv:.3e}"
        blocking = True
    elif final_cv < 0.1:
        evolution = "nearly_uniform"
        status = "NEUTRAL"
        message = f"Quasi-uniforme: CV={final_cv:.3f}"
        blocking = False
    else:
        if ratio < 0.5:
            evolution = "decreasing"
        elif ratio > 2.0:
            evolution = "increasing"
        else:
            evolution = "stable"
        
        status = "PASS"
        message = f"Diversité maintenue: CV={final_cv:.3f}"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_cv,
        final_diversity=final_cv,
        max_diversity=max_cv,
        min_diversity=min_cv,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )


def test_distinct_values(history: List[np.ndarray],
                        tolerance: float = 1e-6,
                        name: str = "TEST-DIV-DISTINCT") -> DiversityResult:
    """
    Compte le nombre de valeurs distinctes (avec tolérance).
    
    Détecte collapsus vers ensemble fini de valeurs.
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def count_distinct(state, tol):
        """Compte valeurs distinctes avec tolérance."""
        flat = np.sort(state.flatten())
        diffs = np.diff(flat)
        n_distinct = 1 + np.sum(diffs > tol)
        return n_distinct
    
    distinct_counts = [count_distinct(state, tolerance) for state in history]
    
    initial_distinct = distinct_counts[0]
    final_distinct = distinct_counts[-1]
    max_distinct = max(distinct_counts)
    min_distinct = min(distinct_counts)
    
    ratio = final_distinct / (initial_distinct + 1e-10)
    
    # Verdict
    n_elements = history[-1].size
    final_ratio = final_distinct / n_elements
    
    if final_ratio < 0.01:
        evolution = "collapsed"
        status = "FAIL"
        message = f"Collapsus: {initial_distinct} → {final_distinct} valeurs distinctes"
        blocking = True
    elif ratio < 0.5:
        evolution = "decreasing"
        status = "NEUTRAL"
        message = f"Diversité réduite: {initial_distinct} → {final_distinct} valeurs"
        blocking = False
    else:
        evolution = "preserved"
        status = "PASS"
        message = f"Diversité préservée: {final_distinct} valeurs distinctes"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=float(initial_distinct),
        final_diversity=float(final_distinct),
        max_diversity=float(max_distinct),
        min_diversity=float(min_distinct),
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )