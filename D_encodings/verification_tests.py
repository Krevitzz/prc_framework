"""
D_encodings/verification_tests.py

Tests de vérification pour tous les générateurs D^(base).

PRINCIPE: Vérifier que chaque générateur produit des tenseurs
          avec les propriétés annoncées dans leurs docstrings.

NOTE: Ces tests vérifient la CRÉATION, pas le comportement dynamique.
      Le comportement sous Γ est testé dans tests/utilities/.
"""

import numpy as np
from typing import Tuple, Dict, Any


class VerificationResult:
    """Résultat d'un test de vérification."""
    
    def __init__(self, test_name: str, passed: bool, 
                 message: str = "", details: Dict[str, Any] = None):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.test_name}: {self.message}"


# ============================================================================
# TESTS GÉNÉRIQUES (applicable à tout tenseur)
# ============================================================================

def verify_shape(tensor: np.ndarray, expected_shape: Tuple[int, ...],
                name: str = "") -> VerificationResult:
    """Vérifie la forme du tenseur."""
    if tensor.shape == expected_shape:
        return VerificationResult(
            f"shape_{name}",
            True,
            f"Shape correct: {expected_shape}"
        )
    else:
        return VerificationResult(
            f"shape_{name}",
            False,
            f"Expected {expected_shape}, got {tensor.shape}"
        )


def verify_bounds(tensor: np.ndarray, min_val: float, max_val: float,
                 name: str = "", tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie que le tenseur respecte les bornes."""
    actual_min = np.min(tensor)
    actual_max = np.max(tensor)
    
    if (actual_min >= min_val - tolerance and 
        actual_max <= max_val + tolerance):
        return VerificationResult(
            f"bounds_{name}",
            True,
            f"Bounds OK: [{actual_min:.3f}, {actual_max:.3f}] ⊆ [{min_val}, {max_val}]"
        )
    else:
        return VerificationResult(
            f"bounds_{name}",
            False,
            f"Out of bounds: [{actual_min:.3f}, {actual_max:.3f}] not in [{min_val}, {max_val}]"
        )


def verify_not_constant(tensor: np.ndarray, name: str = "",
                       tolerance: float = 1e-10) -> VerificationResult:
    """Vérifie que le tenseur n'est pas constant."""
    std = np.std(tensor)
    
    if std > tolerance:
        return VerificationResult(
            f"non_constant_{name}",
            True,
            f"Non-constant (std={std:.6f})"
        )
    else:
        return VerificationResult(
            f"non_constant_{name}",
            False,
            f"Nearly constant (std={std:.6e})"
        )


# ============================================================================
# TESTS SYMÉTRIE (rang 2)
# ============================================================================

def verify_symmetry(matrix: np.ndarray, name: str = "",
                   tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie la symétrie d'une matrice."""
    if matrix.ndim != 2:
        return VerificationResult(
            f"symmetry_{name}",
            False,
            "Not a matrix (rank ≠ 2)"
        )
    
    asymmetry = np.linalg.norm(matrix - matrix.T)
    
    if asymmetry < tolerance:
        return VerificationResult(
            f"symmetry_{name}",
            True,
            f"Symmetric (||A-A^T|| = {asymmetry:.2e})"
        )
    else:
        return VerificationResult(
            f"symmetry_{name}",
            False,
            f"Not symmetric (||A-A^T|| = {asymmetry:.2e})"
        )


def verify_antisymmetry(matrix: np.ndarray, name: str = "",
                       tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie l'antisymétrie d'une matrice."""
    if matrix.ndim != 2:
        return VerificationResult(
            f"antisymmetry_{name}",
            False,
            "Not a matrix"
        )
    
    deviation = np.linalg.norm(matrix + matrix.T)
    
    if deviation < tolerance:
        return VerificationResult(
            f"antisymmetry_{name}",
            True,
            f"Antisymmetric (||A+A^T|| = {deviation:.2e})"
        )
    else:
        return VerificationResult(
            f"antisymmetry_{name}",
            False,
            f"Not antisymmetric (||A+A^T|| = {deviation:.2e})"
        )


def verify_diagonal_ones(matrix: np.ndarray, name: str = "",
                        tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie que la diagonale vaut 1."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return VerificationResult(
            f"diagonal_{name}",
            False,
            "Not a square matrix"
        )
    
    diag = np.diag(matrix)
    deviation = np.linalg.norm(diag - 1.0)
    
    if deviation < tolerance:
        return VerificationResult(
            f"diagonal_{name}",
            True,
            f"Diagonal = 1 (deviation = {deviation:.2e})"
        )
    else:
        return VerificationResult(
            f"diagonal_{name}",
            False,
            f"Diagonal ≠ 1 (deviation = {deviation:.2e})"
        )


def verify_positive_definite(matrix: np.ndarray, name: str = "",
                             tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie la positivité définie via valeurs propres."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return VerificationResult(
            f"pos_def_{name}",
            False,
            "Not a square matrix"
        )
    
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)
        
        if min_eig > tolerance:
            return VerificationResult(
                f"pos_def_{name}",
                True,
                f"Positive definite (min_λ = {min_eig:.3e})"
            )
        else:
            return VerificationResult(
                f"pos_def_{name}",
                False,
                f"Not positive definite (min_λ = {min_eig:.3e})"
            )
    except np.linalg.LinAlgError:
        return VerificationResult(
            f"pos_def_{name}",
            False,
            "Eigenvalue computation failed"
        )


# ============================================================================
# TESTS STRUCTURE (rang 2)
# ============================================================================

def verify_sparsity(matrix: np.ndarray, expected_sparsity: float,
                   name: str = "", tolerance: float = 0.1,
                   zero_threshold: float = 1e-10) -> VerificationResult:
    """Vérifie le niveau de sparsité."""
    n_elements = matrix.size
    n_zeros = np.sum(np.abs(matrix) < zero_threshold)
    actual_sparsity = n_zeros / n_elements
    
    if abs(actual_sparsity - expected_sparsity) < tolerance:
        return VerificationResult(
            f"sparsity_{name}",
            True,
            f"Sparsity OK: {actual_sparsity:.2%} ≈ {expected_sparsity:.2%}"
        )
    else:
        return VerificationResult(
            f"sparsity_{name}",
            False,
            f"Sparsity mismatch: {actual_sparsity:.2%} vs {expected_sparsity:.2%}"
        )


def verify_banded(matrix: np.ndarray, bandwidth: int, name: str = "",
                 zero_threshold: float = 1e-10) -> VerificationResult:
    """Vérifie la structure en bande."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return VerificationResult(
            f"banded_{name}",
            False,
            "Not a square matrix"
        )
    
    n = matrix.shape[0]
    violations = 0
    
    for i in range(n):
        for j in range(n):
            if abs(i - j) > bandwidth:
                if abs(matrix[i, j]) > zero_threshold:
                    violations += 1
    
    if violations == 0:
        return VerificationResult(
            f"banded_{name}",
            True,
            f"Banded structure (bandwidth={bandwidth})"
        )
    else:
        return VerificationResult(
            f"banded_{name}",
            False,
            f"{violations} elements outside band"
        )


def verify_triangular(matrix: np.ndarray, upper_or_lower: str = "lower",
                     name: str = "", zero_threshold: float = 1e-10) -> VerificationResult:
    """Vérifie la structure triangulaire."""
    if matrix.ndim != 2:
        return VerificationResult(
            f"triangular_{name}",
            False,
            "Not a matrix"
        )
    
    if upper_or_lower == "lower":
        # Vérifie que partie supérieure stricte est nulle
        upper = np.triu(matrix, k=1)
        violations = np.sum(np.abs(upper) > zero_threshold)
    else:
        # Vérifie que partie inférieure stricte est nulle
        lower = np.tril(matrix, k=-1)
        violations = np.sum(np.abs(lower) > zero_threshold)
    
    if violations == 0:
        return VerificationResult(
            f"triangular_{name}",
            True,
            f"{upper_or_lower.capitalize()} triangular"
        )
    else:
        return VerificationResult(
            f"triangular_{name}",
            False,
            f"{violations} non-zero elements in wrong triangle"
        )


# ============================================================================
# TESTS RANG 3
# ============================================================================

def verify_partial_symmetry_23(tensor: np.ndarray, name: str = "",
                               tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie T[i,j,k] = T[i,k,j]."""
    if tensor.ndim != 3:
        return VerificationResult(
            f"partial_sym_{name}",
            False,
            "Not a rank-3 tensor"
        )
    
    # Transpose indices 1 et 2
    T_transposed = np.transpose(tensor, (0, 2, 1))
    deviation = np.linalg.norm(tensor - T_transposed)
    
    if deviation < tolerance:
        return VerificationResult(
            f"partial_sym_{name}",
            True,
            f"Partial symmetry (indices 2-3): ||T-T'|| = {deviation:.2e}"
        )
    else:
        return VerificationResult(
            f"partial_sym_{name}",
            False,
            f"No partial symmetry: ||T-T'|| = {deviation:.2e}"
        )


def verify_full_symmetry_rank3(tensor: np.ndarray, name: str = "",
                               tolerance: float = 1e-6) -> VerificationResult:
    """Vérifie symétrie complète (toutes permutations)."""
    if tensor.ndim != 3:
        return VerificationResult(
            f"full_sym_{name}",
            False,
            "Not a rank-3 tensor"
        )
    
    # Teste toutes les permutations
    permutations = [
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ]
    
    max_deviation = 0.0
    for perm in permutations:
        T_perm = np.transpose(tensor, perm)
        deviation = np.linalg.norm(tensor - T_perm)
        max_deviation = max(max_deviation, deviation)
    
    if max_deviation < tolerance:
        return VerificationResult(
            f"full_sym_{name}",
            True,
            f"Full symmetry: max deviation = {max_deviation:.2e}"
        )
    else:
        return VerificationResult(
            f"full_sym_{name}",
            False,
            f"Not fully symmetric: max deviation = {max_deviation:.2e}"
        )


# ============================================================================
# TESTS DE REPRODUCTIBILITÉ
# ============================================================================

def verify_reproducibility(generator_func, params: dict, name: str = "",
                          n_trials: int = 3) -> VerificationResult:
    """Vérifie que le générateur avec seed fixé donne résultats identiques."""
    
    if 'seed' not in params:
        return VerificationResult(
            f"reproducibility_{name}",
            False,
            "No seed parameter"
        )
    
    # Génère plusieurs fois avec même seed
    results = []
    for _ in range(n_trials):
        result = generator_func(**params)
        results.append(result)
    
    # Vérifie que tous identiques
    max_diff = 0.0
    for i in range(1, n_trials):
        diff = np.linalg.norm(results[i] - results[0])
        max_diff = max(max_diff, diff)
    
    if max_diff < 1e-15:
        return VerificationResult(
            f"reproducibility_{name}",
            True,
            f"Reproducible (max diff = {max_diff:.2e})"
        )
    else:
        return VerificationResult(
            f"reproducibility_{name}",
            False,
            f"Not reproducible (max diff = {max_diff:.2e})"
        )


# ============================================================================
# FONCTION PRINCIPALE DE TEST
# ============================================================================

def run_verification_suite(generator_func, params: dict, 
                          tests_to_run: list,
                          name: str = "") -> Dict[str, VerificationResult]:
    """
    Execute une suite de tests de vérification.
    
    Args:
        generator_func: Fonction générateur à tester
        params: Paramètres pour le générateur
        tests_to_run: Liste de tests à exécuter (strings)
        name: Nom pour identification
    
    Returns:
        Dictionnaire {test_name: VerificationResult}
    
    Exemple:
        results = run_verification_suite(
            create_identity,
            {'n_dof': 50},
            ['shape', 'symmetry', 'diagonal_ones', 'bounds'],
            name="identity_50"
        )
    """
    results = {}
    
    # Génère le tenseur
    try:
        tensor = generator_func(**params)
    except Exception as e:
        results['generation'] = VerificationResult(
            f"generation_{name}",
            False,
            f"Generation failed: {str(e)}"
        )
        return results
    
    # Execute les tests demandés
    n_dof = params.get('n_dof', tensor.shape[0])
    
    for test in tests_to_run:
        if test == 'shape_2d':
            results[test] = verify_shape(tensor, (n_dof, n_dof), name)
        
        elif test == 'shape_3d':
            results[test] = verify_shape(tensor, (n_dof, n_dof, n_dof), name)
        
        elif test == 'bounds':
            results[test] = verify_bounds(tensor, -1.0, 1.0, name)
        
        elif test == 'symmetry':
            results[test] = verify_symmetry(tensor, name)
        
        elif test == 'antisymmetry':
            results[test] = verify_antisymmetry(tensor, name)
        
        elif test == 'diagonal_ones':
            results[test] = verify_diagonal_ones(tensor, name)
        
        elif test == 'positive_definite':
            results[test] = verify_positive_definite(tensor, name)
        
        elif test == 'non_constant':
            results[test] = verify_not_constant(tensor, name)
        
        elif test.startswith('sparsity_'):
            expected = float(test.split('_')[1])
            results[test] = verify_sparsity(tensor, expected, name)
        
        elif test.startswith('banded_'):
            bandwidth = int(test.split('_')[1])
            results[test] = verify_banded(tensor, bandwidth, name)
        
        elif test == 'triangular_lower':
            results[test] = verify_triangular(tensor, 'lower', name)
        
        elif test == 'partial_symmetry_23':
            results[test] = verify_partial_symmetry_23(tensor, name)
        
        elif test == 'full_symmetry_rank3':
            results[test] = verify_full_symmetry_rank3(tensor, name)
        
        elif test == 'reproducibility':
            results[test] = verify_reproducibility(generator_func, params, name)
    
    return results


def print_verification_report(results: Dict[str, VerificationResult],
                             title: str = "Verification Report"):
    """Affiche un rapport de vérification."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)
    
    for result in results.values():
        print(result)
    
    print(f"\n{'-'*60}")
    print(f"Summary: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print(f"{'='*60}\n")