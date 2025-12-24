"""
tests/utilities/test_symmetry.py

Tests modulaires observant l'évolution de la symétrie.

PRINCIPE: OBSERVATION PURE, pas de prescription.

Ces tests détectent:
- Préservation de symétrie existante
- Création de symétrie
- Destruction de symétrie
- Antisymétrie

Le TM décide de l'interprétation (PASS/FAIL selon contexte).
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SymmetryResult:
    """Résultat d'un test de symétrie."""
    test_name: str
    initial_symmetric: bool
    final_symmetric: bool
    transition: str  # "preserved", "created", "destroyed", "absent"
    max_asymmetry: float
    mean_asymmetry: float
    asymmetry_trend: str  # "increasing", "decreasing", "stable"
    status: str  # "PASS", "FAIL", "NEUTRAL"
    message: str
    blocking: bool = False  # Ce test est-il bloquant ?
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def compute_asymmetry(matrix: np.ndarray, norm_type: str = 'fro') -> float:
    """
    Calcule l'asymétrie d'une matrice.
    
    Args:
        matrix: Matrice à tester (N×N)
        norm_type: Type de norme ('fro', 'max', 'spectral')
    
    Returns:
        Mesure d'asymétrie ||A - A^T|| selon norme choisie
    """
    if matrix.ndim != 2:
        raise ValueError("compute_asymmetry nécessite une matrice rang 2")
    
    diff = matrix - matrix.T
    
    if norm_type == 'fro':
        return np.linalg.norm(diff, 'fro')
    elif norm_type == 'max':
        return np.max(np.abs(diff))
    elif norm_type == 'spectral':
        return np.linalg.norm(diff, 2)
    else:
        raise ValueError(f"norm_type inconnu: {norm_type}")


def is_symmetric(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    """Teste si une matrice est symétrique."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return compute_asymmetry(matrix, 'fro') < tol


def compute_antisymmetry(matrix: np.ndarray) -> float:
    """Calcule la mesure d'antisymétrie ||A + A^T||."""
    if matrix.ndim != 2:
        raise ValueError("compute_antisymmetry nécessite une matrice rang 2")
    return np.linalg.norm(matrix + matrix.T, 'fro')


def is_antisymmetric(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    """Teste si une matrice est antisymétrique."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return compute_antisymmetry(matrix) < tol


# ============================================================================
# TEST-SYM-001: Préservation symétrie
# ============================================================================

def test_symmetry_preservation(history: List[np.ndarray],
                               tol: float = 1e-4,
                               name: str = "TEST-SYM-001") -> SymmetryResult:
    """
    TEST-SYM-001: Observe si Γ préserve la symétrie d'un D symétrique initial.
    
    APPLICABLE: Si D initial symétrique (D-SYM-*)
    
    MESURE: max_t ||D_t - D_t^T|| / ||D_t||
    
    VERDICT:
    - PASS si max_asymmetry < tol (symétrie préservée)
    - FAIL si max_asymmetry > 100*tol (symétrie détruite)
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états [D_0, D_1, ..., D_T]
        tol: Seuil de tolérance (1e-4 par défaut)
        name: Nom du test
    
    Returns:
        SymmetryResult avec status et détails
    """
    if not history:
        return SymmetryResult(
            test_name=name,
            initial_symmetric=False,
            final_symmetric=False,
            transition="unknown",
            max_asymmetry=0.0,
            mean_asymmetry=0.0,
            asymmetry_trend="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Vérifier que toutes les matrices sont rang 2
    if any(state.ndim != 2 for state in history):
        return SymmetryResult(
            test_name=name,
            initial_symmetric=False,
            final_symmetric=False,
            transition="N/A",
            max_asymmetry=0.0,
            mean_asymmetry=0.0,
            asymmetry_trend="N/A",
            status="NEUTRAL",
            message="Test non applicable (tenseur rang ≠ 2)",
            blocking=False
        )
    
    # Calcule asymétries pour tous les états
    asymmetries = [compute_asymmetry(state, 'fro') for state in history]
    
    # Normalise par norme de l'état (pour avoir mesure relative)
    norms = [np.linalg.norm(state, 'fro') for state in history]
    relative_asymmetries = [a / (n + 1e-10) for a, n in zip(asymmetries, norms)]
    
    # Détecte tendance
    if len(asymmetries) > 1:
        trend_coef = np.polyfit(range(len(asymmetries)), asymmetries, 1)[0]
        if abs(trend_coef) < 1e-8:
            trend = "stable"
        elif trend_coef > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"
    
    # État initial/final
    initial_sym = is_symmetric(history[0], tol)
    final_sym = is_symmetric(history[-1], tol)
    
    # Détermination transition
    if initial_sym and final_sym:
        transition = "preserved"
    elif not initial_sym and final_sym:
        transition = "created"
    elif initial_sym and not final_sym:
        transition = "destroyed"
    else:
        transition = "absent"
    
    # Verdict
    max_asym = max(relative_asymmetries)
    mean_asym = np.mean(relative_asymmetries)
    
    if transition == "preserved" and max_asym < tol:
        status = "PASS"
        message = f"Symétrie préservée (max asym: {max_asym:.2e})"
    elif transition == "destroyed":
        status = "FAIL"
        message = f"Symétrie détruite (max asym: {max_asym:.2e})"
    elif transition == "created":
        status = "NEUTRAL"
        message = f"Symétrie créée (observation intéressante)"
    else:
        status = "NEUTRAL"
        message = f"Symétrie absente (max asym: {max_asym:.2e})"
    
    return SymmetryResult(
        test_name=name,
        initial_symmetric=initial_sym,
        final_symmetric=final_sym,
        transition=transition,
        max_asymmetry=max_asym,
        mean_asymmetry=mean_asym,
        asymmetry_trend=trend,
        status=status,
        message=message,
        blocking=False  # Non bloquant par défaut
    )


# ============================================================================
# TEST-SYM-002: Création symétrie
# ============================================================================

def test_symmetry_creation(history: List[np.ndarray],
                          tol: float = 1e-4,
                          name: str = "TEST-SYM-002") -> SymmetryResult:
    """
    TEST-SYM-002: Observe si Γ crée de la symétrie à partir d'un D asymétrique.
    
    APPLICABLE: Si D initial asymétrique (D-ASY-*)
    
    MESURE: ||D_final - D_final^T|| / ||D_final||
    
    VERDICT:
    - PASS si symétrie créée (final symétrique, initial asymétrique)
    - NEUTRAL sinon (pas de création attendue)
    
    Args:
        history: Liste d'états
        tol: Seuil de tolérance
        name: Nom du test
    
    Returns:
        SymmetryResult avec status
    """
    if not history or any(state.ndim != 2 for state in history):
        return SymmetryResult(
            test_name=name,
            initial_symmetric=False,
            final_symmetric=False,
            transition="N/A",
            max_asymmetry=0.0,
            mean_asymmetry=0.0,
            asymmetry_trend="N/A",
            status="NEUTRAL",
            message="Test non applicable",
            blocking=False
        )
    
    initial_sym = is_symmetric(history[0], tol)
    final_sym = is_symmetric(history[-1], tol)
    
    initial_asym = compute_asymmetry(history[0], 'fro') / (np.linalg.norm(history[0], 'fro') + 1e-10)
    final_asym = compute_asymmetry(history[-1], 'fro') / (np.linalg.norm(history[-1], 'fro') + 1e-10)
    
    if not initial_sym and final_sym:
        status = "PASS"
        message = f"Symétrie créée: {initial_asym:.2e} → {final_asym:.2e}"
        transition = "created"
    elif initial_sym:
        status = "NEUTRAL"
        message = "Initial déjà symétrique (test non applicable)"
        transition = "preserved" if final_sym else "destroyed"
    else:
        status = "NEUTRAL"
        message = f"Pas de création (asym: {initial_asym:.2e} → {final_asym:.2e})"
        transition = "absent"
    
    return SymmetryResult(
        test_name=name,
        initial_symmetric=initial_sym,
        final_symmetric=final_sym,
        transition=transition,
        max_asymmetry=final_asym,
        mean_asymmetry=(initial_asym + final_asym) / 2,
        asymmetry_trend="decreasing" if final_asym < initial_asym else "increasing",
        status=status,
        message=message,
        blocking=False
    )


# ============================================================================
# TEST-SYM-003: Amplification/réduction asymétrie
# ============================================================================

def test_asymmetry_evolution(history: List[np.ndarray],
                            name: str = "TEST-SYM-003") -> SymmetryResult:
    """
    TEST-SYM-003: Observe l'évolution du ratio asymétrie/symétrie.
    
    APPLICABLE: Tous D rang 2
    
    MESURE: ratio ||D - D^T|| / ||D + D^T|| initial vs final
    
    VERDICT: NEUTRAL (observation pure, pas de verdict)
    
    Args:
        history: Liste d'états
        name: Nom du test
    
    Returns:
        SymmetryResult avec observation
    """
    if not history or any(state.ndim != 2 for state in history):
        return SymmetryResult(
            test_name=name,
            initial_symmetric=False,
            final_symmetric=False,
            transition="N/A",
            max_asymmetry=0.0,
            mean_asymmetry=0.0,
            asymmetry_trend="N/A",
            status="NEUTRAL",
            message="Test non applicable",
            blocking=False
        )
    
    def compute_ratio(state):
        asym = np.linalg.norm(state - state.T, 'fro')
        sym = np.linalg.norm(state + state.T, 'fro')
        return asym / (sym + 1e-10)
    
    ratios = [compute_ratio(state) for state in history]
    
    initial_ratio = ratios[0]
    final_ratio = ratios[-1]
    mean_ratio = np.mean(ratios)
    
    if final_ratio < initial_ratio:
        trend = "decreasing"
        message = f"Asymétrie réduite: ratio {initial_ratio:.3f} → {final_ratio:.3f}"
    elif final_ratio > initial_ratio:
        trend = "increasing"
        message = f"Asymétrie amplifiée: ratio {initial_ratio:.3f} → {final_ratio:.3f}"
    else:
        trend = "stable"
        message = f"Asymétrie stable: ratio ≈ {final_ratio:.3f}"
    
    return SymmetryResult(
        test_name=name,
        initial_symmetric=initial_ratio < 1e-6,
        final_symmetric=final_ratio < 1e-6,
        transition="observation",
        max_asymmetry=max(ratios),
        mean_asymmetry=mean_ratio,
        asymmetry_trend=trend,
        status="NEUTRAL",
        message=message,
        blocking=False
    )


# ============================================================================
# TEST ANTISYMÉTRIE (bonus)
# ============================================================================

def test_antisymmetry_preservation(history: List[np.ndarray],
                                  tol: float = 1e-4,
                                  name: str = "TEST-SYM-ANTI") -> SymmetryResult:
    """
    Teste si Γ préserve l'antisymétrie (A = -A^T).
    
    APPLICABLE: Si D initial antisymétrique (D-ASY-003)
    """
    if not history or any(state.ndim != 2 for state in history):
        return SymmetryResult(
            test_name=name,
            initial_symmetric=False,
            final_symmetric=False,
            transition="N/A",
            max_asymmetry=0.0,
            mean_asymmetry=0.0,
            asymmetry_trend="N/A",
            status="NEUTRAL",
            message="Test non applicable",
            blocking=False
        )
    
    initial_anti = is_antisymmetric(history[0], tol)
    final_anti = is_antisymmetric(history[-1], tol)
    
    antisym_measures = [compute_antisymmetry(state) for state in history]
    max_antisym = max(antisym_measures)
    
    if initial_anti and final_anti:
        status = "PASS"
        message = "Antisymétrie préservée"
        transition = "preserved"
    elif initial_anti and not final_anti:
        status = "FAIL"
        message = "Antisymétrie détruite"
        transition = "destroyed"
    else:
        status = "NEUTRAL"
        message = "Antisymétrie non applicable"
        transition = "absent"
    
    return SymmetryResult(
        test_name=name,
        initial_symmetric=initial_anti,
        final_symmetric=final_anti,
        transition=transition,
        max_asymmetry=max_antisym,
        mean_asymmetry=np.mean(antisym_measures),
        asymmetry_trend="observation",
        status=status,
        message=message,
        blocking=False
    )